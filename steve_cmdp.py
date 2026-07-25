"""
stEVE to OmniSafe adapter for constrained RL training.

Wraps a stEVE Env as an OmniSafe CMDP, flattening the nested observation
dict to a flat Box and emitting a cost signal alongside reward.

The cost function is injected at construction. First version uses
ZeroCost to validate the plumbing without physics involved.
"""

from __future__ import annotations
import csv
import os
import time
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import torch
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace

import random

import eve
from eve.intervention.vesseltree.aorticarch import ArchType
from force_read import assert_force_available, max_along_device, tip_force, force_N


# ============================================================
# Cost functions
# ============================================================

# Threshold where the hinge starts, and the ceiling that bounds runaway
# buckling forces. These are separate from the episode budget d, and must
# stay separate or the sensitivity analysis is uninterpretable.
# Defined here because MaxAlongDeviceHinge uses them as default arguments,
# which Python evaluates at definition time.
COST_THRESHOLD_N = 0.85
COST_CEILING_N = 5.0


def force_hinge(f: float, threshold: float = COST_THRESHOLD_N,
                ceiling: float = COST_CEILING_N) -> float:
    """Excess force above threshold, with runaway buckling bounded.

    Used in exactly two places, and they must never diverge: the constraint
    cost in MaxAlongDeviceHinge (condition 3), and the R4 reward penalty
    (condition 2). If they diverge, those two conditions stop seeing the same
    force signal and the comparison is no longer single variable.
    """
    return max(0.0, min(f, ceiling) - threshold)


class CostFn:
    """Interface: take simulation, return non negative float."""
    def __call__(self, simulation) -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ZeroCost(CostFn):
    """Constant zero cost. Used to validate plumbing without physics."""
    def __call__(self, simulation) -> float:
        return 0.0


class MaxAlongDeviceHinge(CostFn):
    def __init__(self, threshold: float = COST_THRESHOLD_N,
                 ceiling: float = COST_CEILING_N):
        self.threshold = float(threshold)
        self.ceiling = float(ceiling)
        

    def __call__(self, simulation) -> float:
        return force_hinge(max_along_device(simulation), self.threshold, self.ceiling)

    @property
    def name(self) -> str:
        return "MaxAlongDeviceHinge(t=%.3f,c=%.3f)" % (self.threshold, self.ceiling)


class TipForceHinge(CostFn):
    """Hinge on the tip node force rather than the max along the device.

    Same hinge shape and same shared force_hinge as MaxAlongDeviceHinge, so
    the two cost functions differ only in which force they read. The tip
    signal stays in the regime reported by Robertshaw et al. 2025 (mean tip
    force 0.24 to 0.29 N, max under 1.0 N), whereas max along device is
    dominated by buckling artefacts an order of magnitude larger that pin the
    ceiling in the deep insertion region where the targets are. The tip is
    blind to shaft loading on curved sections, which is a stated limitation,
    and shadow_max is logged alongside on every run so that limitation can be
    quantified.
    """
    def __init__(self, threshold: float = COST_THRESHOLD_N,
                 ceiling: float = COST_CEILING_N):
        self.threshold = float(threshold)
        self.ceiling = float(ceiling)

    def __call__(self, simulation) -> float:
        return force_hinge(tip_force(simulation), self.threshold, self.ceiling)

    @property
    def name(self) -> str:
        return "TipForceHinge(t=%.3f,c=%.3f)" % (self.threshold, self.ceiling)


# ============================================================
# Insertion limit
# ============================================================

# Fraction of the shortest device length at which the episode is truncated.
# The InterventionalRadiologyController segfaults once the inserted length
# approaches the device length, so the episode must end before that point.
MAX_INSERTION_FRACTION = 0.85   # forward translation is clamped here
HARD_INSERTION_FRACTION = 0.95  # backstop truncation, should never fire
MAX_EPISODE_STEPS = 200

# Anatomy variety. Arch morphology is the generalisation axis: the agent
# trains on four arch types and is evaluated on two it has never seen.
# This is what makes the R4 generalisation collapse reproducible.
# Note: AorticArch treats seed 0 as falsy and replaces it, so seeds start at 1.
TRAIN_ARCH_TYPES = [ArchType.I, ArchType.II, ArchType.IV, ArchType.V]
TEST_ARCH_TYPES = [ArchType.VI, ArchType.VII]
ARCH_SEED_RANGE = (1, 10**6)

def insertion_limit(env, fraction: float = MAX_INSERTION_FRACTION) -> float:
    devices = getattr(env.intervention, "devices", None)
    if devices:
        lengths = [float(d.length) for d in devices]
        return min(lengths) * fraction
    # Fallback if the attribute name differs on this stEVE version
    return 450.0 * fraction



# ============================================================
# Env builder
# ============================================================

def build_steve_env(
    seed: Optional[int] = None, arch_type: ArchType = ArchType.I,
) -> eve.Env:
    """
    Build the arch / JShaped stEVE env.
    Kept as a function so the adapter can rebuild on reset if needed
    and so training scripts can share the exact construction.
    """
    if seed is None:
        seed = 30

    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed, arch_type=arch_type, scaling_xyzd=[1.0, 1.0, 1.0, 0.75]
    )

    device = eve.intervention.device.JShaped(velocity_limit=(40, 3.14))
    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation, vessel_tree=vessel_tree,
        image_frequency=7.5, image_rot_zx=[20, 5],
    )
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree, fluoroscopy=fluoroscopy, threshold=5,
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree, devices=[device], simulation=simulation,
        fluoroscopy=fluoroscopy, target=target,
    )
    start = eve.start.MaxDeviceLength(intervention=intervention, max_length=380)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_state = eve.observation.Target2D(intervention=intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(target_state, intervention)
    rotation = eve.observation.Rotations(intervention=intervention)
    state = eve.observation.ObsDict({
        "position": position,
        "target": target_state,
        "rotation": rotation,
    })

    # R1 from Robertshaw et al. 2025, Eq. 1:
    #   R1 = -0.005 - 0.001 * delta_pathlength + 1.0 on target reached
    # PathLengthDelta negates the delta internally, so a positive factor
    # pays for progress. Step emits its factor on every step.
    target_reward = eve.reward.TargetReached(intervention=intervention, factor=1.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.001)
    step_penalty = eve.reward.Step(factor=-0.005)
    reward = eve.reward.Combination([target_reward, path_delta, step_penalty])

    target_reached = eve.terminal.TargetReached(intervention=intervention)
    truncation = eve.truncation.Combination([
        eve.truncation.MaxSteps(MAX_EPISODE_STEPS),
        eve.truncation.SimError(intervention=intervention),
        eve.truncation.VesselEnd(intervention=intervention),
    ])

    env = eve.Env(
        intervention=intervention, observation=state, reward=reward,
        terminal=target_reached, truncation=truncation,
        start=start, pathfinder=pathfinder,
    )
    return env


# ============================================================
# Observation flatten
# ============================================================

OBS_KEY_ORDER = ("position", "target", "rotation")

# Set STEVE_EPISODE_LOG to a csv path to record one row per episode.
# Unset means no logging, so probes and smoke tests stay clean.
EPISODE_LOG = os.environ.get("STEVE_EPISODE_LOG", "")

# OmniSafe builds separate train and eval environments in one process, both
# writing to the same CSV. Tag each instance so rows can be separated.
# Module level so the count is shared across every SteveCMDP created.
_ENV_INSTANCE_COUNTER = [0]


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Concatenate obs dict values in fixed key order to a flat float32 vector."""
    parts = []
    for k in OBS_KEY_ORDER:
        v = np.asarray(obs_dict[k], dtype=np.float32).ravel()
        parts.append(v)
    return np.concatenate(parts, axis=0)


def flat_obs_size(env: eve.Env) -> int:
    total = 0
    for k in OBS_KEY_ORDER:
        total += int(np.prod(env.observation_space[k].shape))
    return total


# ============================================================
# The adapter
# ============================================================

@env_register
class SteveCMDP(CMDP):
    """
    stEVE wrapped as an OmniSafe CMDP.

    Single instance only (SOFA is not safe to parallelise from Python).
    Cost function injected at construction.
    """

    _support_envs: ClassVar[list[str]] = ["SteveNav-v0"]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False  # stEVE already truncates at MaxSteps

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        cost_fn: Optional[CostFn] = None,
        reward_penalty_weight: float = 0.0,
        seed: int = 30,
        train: bool = True,
        vary_anatomy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        assert num_envs == 1, "SteveCMDP does not support vectorised envs (SOFA constraint)"
        assert env_id in self._support_envs, "unknown env_id: %s" % env_id

        self._device = torch.device(device)
        self._cost_fn = cost_fn if cost_fn is not None else ZeroCost()
        # R4 weight on the force hinge in the reward. 0.0 is condition 1,
        # 0.01 is the published Robertshaw value for condition 2.
        self._reward_penalty_weight = float(reward_penalty_weight)
        self._seed = seed
        self._num_envs = 1
        self._train = bool(train)
        self._vary_anatomy = bool(vary_anatomy)
        self._arch_pool = TRAIN_ARCH_TYPES if self._train else TEST_ARCH_TYPES

        # One RNG drives anatomy choice, seeded once so the whole episode
        # sequence is reproducible from the run seed alone.
        self._anatomy_rng = random.Random(seed)
        self._arch_type = self._arch_pool[0]
        self._arch_seed = seed if seed else 1

        # Build the underlying env
        self._env = build_steve_env(seed=self._arch_seed, arch_type=self._arch_type)

        # Seed the target sampler's RNG once. It then advances naturally
        # across episodes, giving a deterministic sequence of different
        # targets. Never pass a seed to env.reset after this, or the RNG
        # is rebuilt and every episode gets the same target.
        self._env.intervention.target._rng = random.Random(seed)

        # Do one reset so the SOFA scene exists and the LCP node is populated
        self._last_obs_dict, _ = self._env.reset()

        # Fail fast if the solver does not expose constraint forces
        assert_force_available(self._env.intervention.simulation)

        # Forward translation is clamped here, backstop truncates above it
        self._insertion_limit = insertion_limit(self._env, MAX_INSERTION_FRACTION)
        self._insertion_backstop = insertion_limit(self._env, HARD_INSERTION_FRACTION)
        print("insertion clamp: %.1f mm, backstop: %.1f mm"
              % (self._insertion_limit, self._insertion_backstop))

        # Which env instance this is: 1 is normally the training env,
        # 2 the evaluation env OmniSafe builds alongside it.
        _ENV_INSTANCE_COUNTER[0] += 1
        self._env_instance = _ENV_INSTANCE_COUNTER[0]

        # Per episode accumulators, zeroed in reset
        self._episode_nr = 0
        self._step_nr = 0
        self._ep_return = 0.0
        self._ep_cost = 0.0
        self._ep_penalty = 0.0
        self._force_trace = []
        self._tip_trace = []
        self._clamp_steps = 0
        self._path_length_at_reset = 0.0

        # Declare flat observation space
        flat_dim = flat_obs_size(self._env)
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32,
        )

        # stEVE action space is (1, 2) with the leading device axis.
        # OmniSafe's Gaussian actor only handles 1D Box, so flatten to (2,).
        # We keep the original for reshape on the way back.
        self._steve_action_space = self._env.action_space
        low = self._env.action_space.low.reshape(-1)
        high = self._env.action_space.high.reshape(-1)
        self._action_space = spaces.Box(
            low=low, high=high, shape=(low.size,), dtype=np.float32,
        )

        # Metadata OmniSafe reads
        self._metadata = {"render_modes": []}

    # ----- OmniSafe required properties -----

    @property
    def observation_space(self) -> OmnisafeSpace:
        return self._observation_space

    @property
    def action_space(self) -> OmnisafeSpace:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        return MAX_EPISODE_STEPS

    # ----- OmniSafe required methods -----

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self._seed = seed
            self._anatomy_rng = random.Random(seed)
            self._env.intervention.target._rng = random.Random(seed)

        self._set_anatomy()
        # No seed passed: the target RNG must advance rather than be rebuilt
        obs_dict, info = self._env.reset()
        self._last_obs_dict = obs_dict

        self._step_nr = 0
        self._ep_return = 0.0
        self._ep_cost = 0.0
        self._ep_penalty = 0.0
        self._force_trace = []
        self._tip_trace = []
        self._clamp_steps = 0
        try:
            self._path_length_at_reset = float(self._env.pathfinder.path_length)
        except Exception:
            self._path_length_at_reset = 0.0

        flat = flatten_obs(obs_dict)
        return torch.as_tensor(flat, dtype=torch.float32, device=self._device), info
    
    def _set_anatomy(self) -> None:
        """Pick a new arch morphology and force regeneration.

        AorticArch.reset only regenerates when branches is None, so nulling
        it is what triggers a new geometry. The SOFA scene rebuild that
        follows is automatic, driven by the changed mesh path and bounds,
        and costs about 1.5 s.
        """
        if not self._vary_anatomy:
            return
        vt = self._env.intervention.vessel_tree
        self._arch_type = self._anatomy_rng.choice(self._arch_pool)
        self._arch_seed = self._anatomy_rng.randint(*ARCH_SEED_RANGE)
        vt.arch_type = self._arch_type
        vt.seed = self._arch_seed
        vt.branches = None

        # Regenerate geometry now, before env.reset runs, so the episode
        # coordinate space can be corrected before the observation wrappers
        # read it. AorticArch.reset is a no op once branches exist, so the
        # call inside env.reset does nothing further.
        vt.reset(self._episode_nr)
        vt.coordinate_space_episode = vt.coordinate_space

    def _clamp_insertion(self, a: np.ndarray) -> tuple[np.ndarray, bool]:
        """Zero forward translation on any device already at its insertion
        limit. Retraction stays allowed, so the agent can always recover.
        This is an environment boundary, not a reward change, so R1 stays
        exactly as the paper defines it."""
        inserted = np.asarray(
            self._env.intervention.device_lengths_inserted, dtype=np.float64
        )
        blocked = (inserted >= self._insertion_limit) & (a[:, 0] > 0.0)
        if not np.any(blocked):
            return a, False
        a = a.copy()
        a[blocked, 0] = 0.0
        return a, True

    def _flush_episode(self, end_reason: str) -> None:
        if not EPISODE_LOG:
            return
        trace = np.asarray(self._force_trace, dtype=np.float64)
        if trace.size:
            excess = np.maximum(np.minimum(trace, COST_CEILING_N) - COST_THRESHOLD_N, 0.0)
        else:
            excess = np.zeros(0)
        tip_trace = np.asarray(self._tip_trace, dtype=np.float64)
        if tip_trace.size:
            tip_excess = np.maximum(
                np.minimum(tip_trace, COST_CEILING_N) - COST_THRESHOLD_N, 0.0)
        else:
            tip_excess = np.zeros(0)
        row = {
            "wall_time": time.time(),
            "episode": self._episode_nr,
            "steps": self._step_nr,
            "end_reason": end_reason,
            "success": int(end_reason == "target_reached"),
            "ep_return": self._ep_return,
            "ep_cost": self._ep_cost,
            "ep_penalty": self._ep_penalty,
            "reward_penalty_weight": self._reward_penalty_weight,
            "force_mean": float(trace.mean()) if trace.size else 0.0,
            "force_max": float(trace.max()) if trace.size else 0.0,
            "force_p95": float(np.percentile(trace, 95)) if trace.size else 0.0,
            "steps_over_threshold": int((trace > COST_THRESHOLD_N).sum()) if trace.size else 0,
            "hinge_cost_shadow": float(excess.sum()) if trace.size else 0.0,
            "shadow_max": float(excess.sum()) if trace.size else 0.0,
            "shadow_tip": float(tip_excess.sum()) if tip_trace.size else 0.0,
            "tip_mean": float(tip_trace.mean()) if tip_trace.size else 0.0,
            "tip_max": float(tip_trace.max()) if tip_trace.size else 0.0,
            "tip_steps_over_threshold": int((tip_trace > COST_THRESHOLD_N).sum()) if tip_trace.size else 0,
            "excess_mean": float(excess.mean()) if trace.size else 0.0,
            "excess_max": float(excess.max()) if trace.size else 0.0,
            "clamp_steps": self._clamp_steps,
            "inserted_final": float(
                np.asarray(self._env.intervention.device_lengths_inserted).max()
            ),
            "path_length_at_reset": self._path_length_at_reset,
            "cost_fn": self._cost_fn.name,
            "seed": self._seed,
            "arch_type": str(self._arch_type),
            "arch_seed": self._arch_seed,
            "train_pool": int(self._train),
            "env_instance": self._env_instance,
        }

        write_header = not os.path.exists(EPISODE_LOG)
        try:
            with open(EPISODE_LOG, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if write_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as exc:
            print("episode log write failed: %s" % exc)
        self._episode_nr += 1

    def step(
        self, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Torch action to numpy in stEVE's expected shape
        a = action.detach().cpu().numpy().astype(np.float32)

        # OmniSafe gives us (2,), stEVE expects (1, 2). Reshape.
        a = a.reshape(self._steve_action_space.shape)

        # Clamp forward translation at the insertion limit before stepping
        a, clamped = self._clamp_insertion(a)
        if clamped:
            self._clamp_steps += 1

        obs_dict, reward, terminated, truncated, info = self._env.step(a)
        self._last_obs_dict = obs_dict
        self._step_nr += 1

        # Cost is computed from the current simulation state after the step
        cost_value = float(self._cost_fn(self._env.intervention.simulation))

        # Log both raw force numbers in info regardless of cost function used
        sim = self._env.intervention.simulation
        f_max = max_along_device(sim)
        f_tip = tip_force(sim)
        info["force_max_along_device"] = f_max
        info["force_tip"] = f_tip
        info["force_solver_max"] = force_N(sim)
        info["cost_fn"] = self._cost_fn.name

        # R4, condition 2: blend the force penalty into the reward at a fixed
        # weight. Weight zero leaves R1 byte identical, so conditions 1 and 2
        # differ by this one number and nothing else. The hinge must read the
        # same force the active constraint reads, so conditions 2 and 3 stay
        # single variable: when the cost function is the tip hinge, R4 penalises
        # tip force too, and when it is the max hinge, R4 penalises max force.
        penalty = 0.0
        if self._reward_penalty_weight != 0.0:
            f_penalty = f_tip if isinstance(self._cost_fn, TipForceHinge) else f_max
            penalty = self._reward_penalty_weight * force_hinge(f_penalty)
            reward = float(reward) - penalty
        info["force_penalty"] = penalty
        self._ep_penalty += penalty

        self._ep_return += float(reward)
        self._ep_cost += cost_value
        self._force_trace.append(float(f_max))
        self._tip_trace.append(float(f_tip))

        # Backstop only. The clamp should stop insertion well before this.
        inserted = np.asarray(
            self._env.intervention.device_lengths_inserted, dtype=np.float64
        )
        backstop_hit = bool(np.any(inserted > self._insertion_backstop))
        if backstop_hit:
            truncated = True
            print("WARNING: insertion backstop fired at %.1f mm, clamp has a bug"
                  % float(inserted.max()))
        info["inserted_max"] = float(inserted.max())
        info["clamped"] = clamped

        if terminated:
            info["end_reason"] = "target_reached"
        elif backstop_hit:
            info["end_reason"] = "insertion_backstop"
        elif truncated:
            info["end_reason"] = "steve_truncation"
        else:
            info["end_reason"] = None

        if terminated or truncated:
            self._flush_episode(info["end_reason"])

        flat = flatten_obs(obs_dict)
        obs_t = torch.as_tensor(flat, dtype=torch.float32, device=self._device)
        reward_t = torch.tensor(float(reward), dtype=torch.float32, device=self._device)
        cost_t = torch.tensor(cost_value, dtype=torch.float32, device=self._device)
        term_t = torch.tensor(bool(terminated), dtype=torch.bool, device=self._device)
        trunc_t = torch.tensor(bool(truncated), dtype=torch.bool, device=self._device)

        return obs_t, reward_t, cost_t, term_t, trunc_t, info

    def set_seed(self, seed: int) -> None:
        """OmniSafe constructs the env with default args and calls this
        afterwards, so this is the only place the run seed actually
        arrives. Reseed every generator here, not just the stored value."""
        self._seed = seed
        self._anatomy_rng = random.Random(seed)
        try:
            self._env.intervention.target._rng = random.Random(seed)
        except Exception as exc:
            print("could not reseed target rng: %s" % exc)

    def sample_action(self) -> torch.Tensor:
        a = self._action_space.sample()
        return torch.as_tensor(a, dtype=torch.float32, device=self._device)

    def render(self) -> Any:
        return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    def spec_log(self, logger: Any) -> None:
        pass
