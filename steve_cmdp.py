"""
stEVE to OmniSafe adapter for constrained RL training.

Wraps a stEVE Env as an OmniSafe CMDP, flattening the nested observation
dict to a flat Box and emitting a cost signal alongside reward.

The cost function is injected at construction. First version uses
ZeroCost to validate the plumbing without physics involved.
"""

from __future__ import annotations
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import torch
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace

import eve
from force_read import assert_force_available, max_along_device, tip_force, force_N


# ============================================================
# Cost functions
# ============================================================

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
    def __init__(self, threshold: float = 0.85, ceiling: float = 5.0):
        self.threshold = float(threshold)
        self.ceiling = float(ceiling)

    def __call__(self, simulation) -> float:
        f = min(max_along_device(simulation), self.ceiling)
        return max(0.0, f - self.threshold)

    @property
    def name(self) -> str:
        return "MaxAlongDeviceHinge(t=%.3f,c=%.3f)" % (self.threshold, self.ceiling)


# ============================================================
# Env builder
# ============================================================

def build_steve_env(seed: Optional[int] = None) -> eve.Env:
    """
    Build the current arch / JShaped stEVE env.
    Kept as a function so the adapter can rebuild on reset if needed
    and so training scripts can share the exact construction.
    """
    if seed is None:
        seed = 30

    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=seed, scaling_xyzd=[1.0, 1.0, 1.0, 0.75]
    )
    device = eve.intervention.device.JShaped()
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
    start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
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

    target_reward = eve.reward.TargetReached(intervention=intervention, factor=1.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01)
    reward = eve.reward.Combination([target_reward, path_delta])

    target_reached = eve.terminal.TargetReached(intervention=intervention)
    max_steps = eve.truncation.MaxSteps(600)

    env = eve.Env(
        intervention=intervention, observation=state, reward=reward,
        terminal=target_reached, truncation=max_steps,
        start=start, pathfinder=pathfinder,
    )
    return env


# ============================================================
# Observation flatten
# ============================================================

OBS_KEY_ORDER = ("position", "target", "rotation")


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
        seed: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        assert num_envs == 1, "SteveCMDP does not support vectorised envs (SOFA constraint)"
        assert env_id in self._support_envs, "unknown env_id: %s" % env_id

        self._device = torch.device(device)
        self._cost_fn = cost_fn if cost_fn is not None else ZeroCost()
        self._seed = seed
        self._num_envs = 1

        # Build the underlying env
        self._env = build_steve_env(seed=seed)

        # Do one reset so the SOFA scene exists and the LCP node is populated
        self._last_obs_dict, _ = self._env.reset()

        # Fail fast if the solver does not expose constraint forces
        assert_force_available(self._env.intervention.simulation)

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
        return 600

    # ----- OmniSafe required methods -----

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self._seed = seed
        obs_dict, info = self._env.reset()
        self._last_obs_dict = obs_dict
        flat = flatten_obs(obs_dict)
        return torch.as_tensor(flat, dtype=torch.float32, device=self._device), info

    def step(
        self, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Torch action to numpy in stEVE's expected shape
        a = action.detach().cpu().numpy().astype(np.float32)
        # OmniSafe gives us (2,), stEVE expects (1, 2). Reshape.
        a = a.reshape(self._steve_action_space.shape)

        obs_dict, reward, terminated, truncated, info = self._env.step(a)
        self._last_obs_dict = obs_dict

        # Cost is computed from the current simulation state after the step
        cost_value = float(self._cost_fn(self._env.intervention.simulation))

        # Log both raw force numbers in info regardless of cost function used
        sim = self._env.intervention.simulation
        info["force_max_along_device"] = max_along_device(sim)
        info["force_tip"] = tip_force(sim)
        info["force_solver_max"] = force_N(sim)
        info["cost_fn"] = self._cost_fn.name

        flat = flatten_obs(obs_dict)
        obs_t = torch.as_tensor(flat, dtype=torch.float32, device=self._device)
        reward_t = torch.tensor(float(reward), dtype=torch.float32, device=self._device)
        cost_t = torch.tensor(cost_value, dtype=torch.float32, device=self._device)
        term_t = torch.tensor(bool(terminated), dtype=torch.bool, device=self._device)
        trunc_t = torch.tensor(bool(truncated), dtype=torch.bool, device=self._device)

        return obs_t, reward_t, cost_t, term_t, trunc_t, info

    def set_seed(self, seed: int) -> None:
        self._seed = seed

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