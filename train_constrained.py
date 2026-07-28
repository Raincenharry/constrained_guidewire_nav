"""
PID Lagrangian SAC with a max along device force hinge as an explicit constraint.
Condition 3 of the experimental design, Experiment A.

Differs from train_baseline.py by the safety mechanism only. Every shared
setting is copied across deliberately, including the actor learning rate,
which OmniSafe sets to 0.0001 for SACPID and 0.0003 for SAC. Leaving that
default would change two variables at once and break the comparison.
"""

import os
import sys
import argparse

# Must run BEFORE steve_cmdp is imported, because EPISODE_LOG is read at
# module import time. Setting it later has no effect and the log stays empty.
def _preset_episode_log() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    known, _ = ap.parse_known_args()
    path = os.path.abspath("./runs_%s/episodes_seed%d.csv" % (known.tag, known.seed))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.environ["STEVE_EPISODE_LOG"] = path
    return path

EPISODE_LOG_PATH = _preset_episode_log()

import torch
import omnisafe

import steve_cmdp
from steve_cmdp import SteveCMDP, MaxAlongDeviceHinge, TipForceHinge


_ACTIVE_COST_FN = None
_ACTIVE_PENALTY_WEIGHT = 0.0
_original_init = SteveCMDP.__init__


def _patched_init(self, env_id, num_envs=1, device=torch.device("cpu"),
                  cost_fn=None, reward_penalty_weight=None, seed=30, **kwargs):
    if cost_fn is None:
        cost_fn = _ACTIVE_COST_FN
    if reward_penalty_weight is None:
        reward_penalty_weight = _ACTIVE_PENALTY_WEIGHT
    _original_init(self, env_id, num_envs=num_envs, device=device,
                   cost_fn=cost_fn, reward_penalty_weight=reward_penalty_weight,
                   seed=seed, **kwargs)


SteveCMDP.__init__ = _patched_init


def main():
    global _ACTIVE_COST_FN, _ACTIVE_PENALTY_WEIGHT

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="sacpid")
    p.add_argument("--steps_per_epoch", type=int, default=6000,
                   help="must divide --steps exactly, OmniSafe asserts this")
    p.add_argument("--cost_limit", type=float, required=True,
                   help="d, the episode cost budget in newton steps. "
                        "No default on purpose, this is the experimental variable.")
    p.add_argument("--warmup_epochs", type=int, required=True,
                   help="epochs before the PID controller starts. OmniSafe "
                        "defaults to 100, which exceeds the 50 epochs of a "
                        "300k run, so lambda would never move and the run "
                        "would be a silent SAC baseline. No default on purpose.")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="force hinge threshold in newtons")
    p.add_argument("--ceiling", type=float, default=5.0,
                   help="raw force ceiling before the hinge, bounds buckling artefacts")
    p.add_argument("--cost_fn", type=str, default="max", choices=["max", "tip"],
                   help="which force the constraint hinges on: max along device or tip node")
    p.add_argument("--start_learning_steps", type=int, default=10000,
                   help="matches train_baseline.py. Lower it only for local "
                        "wiring checks that are too short to reach it.")
    p.add_argument("--anneal_from", type=float, default=None,
                   help="If set, d starts here and descends linearly to "
                        "--cost_limit, reaching it at --anneal_end_epoch. "
                        "If unset, d is fixed and behaviour is identical to "
                        "every run before 27 July.")
    p.add_argument("--anneal_end_epoch", type=int, default=None,
                   help="Epoch at which d reaches --cost_limit. Required "
                        "when --anneal_from is set.")
    p.add_argument("--pid_kp", type=float, default=None,
                   help="proportional gain. Unset leaves the OmniSafe "
                        "default of 1e-6.")
    p.add_argument("--pid_ki", type=float, default=None,
                   help="integral gain. Unset leaves the OmniSafe default "
                        "of 1e-7, so every run before 28 July reproduces "
                        "byte identically.")
    p.add_argument("--pid_kd", type=float, default=None,
                   help="derivative gain. Unset leaves the OmniSafe "
                        "default of 1e-7.")
    args = p.parse_args()

    if args.steps % args.steps_per_epoch != 0:
        raise SystemExit(
            "steps (%d) must be divisible by steps_per_epoch (%d)"
            % (args.steps, args.steps_per_epoch)
        )

    n_epochs = args.steps // args.steps_per_epoch
    if args.warmup_epochs >= n_epochs:
        raise SystemExit(
            "warmup_epochs (%d) must be less than the total epochs (%d), "
            "otherwise the multiplier never leaves warmup and this run is "
            "an unconstrained baseline wearing a SACPID tag"
            % (args.warmup_epochs, n_epochs)
        )
    if (args.anneal_from is None) != (args.anneal_end_epoch is None):
        raise SystemExit("--anneal_from and --anneal_end_epoch must be given together")
    if args.anneal_from is not None:
        if args.anneal_end_epoch <= args.warmup_epochs:
            raise SystemExit(
                "--anneal_end_epoch (%d) must exceed --warmup_epochs (%d), "
                "otherwise the descent finishes before the controller is ever "
                "called and this is a fixed budget run at --cost_limit"
                % (args.anneal_end_epoch, args.warmup_epochs))
        if args.anneal_from <= args.cost_limit:
            raise SystemExit(
                "--anneal_from (%g) must exceed --cost_limit (%g). The point "
                "is to descend onto the budget from above."
                % (args.anneal_from, args.cost_limit))

    _COST_CLASSES = {"max": MaxAlongDeviceHinge, "tip": TipForceHinge}
    _ACTIVE_COST_FN = _COST_CLASSES[args.cost_fn](threshold=args.threshold,
                                                  ceiling=args.ceiling)
    _ACTIVE_PENALTY_WEIGHT = 0.0

    custom_cfgs = {
        "seed": args.seed,
        "train_cfgs": {
            "total_steps": args.steps,
            "vector_env_nums": 1,
            "parallel": 1,
            "torch_threads": 8,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },

        "algo_cfgs": {
            "steps_per_epoch": args.steps_per_epoch,
            "update_iters": 1,
            "batch_size": 256,
            "gamma": 0.99,
            "polyak": 0.005,
            "start_learning_steps": args.start_learning_steps,
            # The safety mechanism. This is the only intended difference
            # from condition 1.
            "use_cost": True,
            "warmup_epochs": args.warmup_epochs,
            # OmniSafe defaults this to True for SACPID, which rescales the
            # cost before the critic sees it and destroys the physical
            # meaning of d in newton steps. That meaning is the central
            # claim of the project, so this is not optional.
            "cost_normalize": False,
            # Identical to train_baseline.py, see the derivation there.
            "auto_alpha": False,
            "alpha": 0.0005,
        },
        "model_cfgs": {
            "actor": {
                # SACPID.yaml sets 0.0001 while SAC.yaml sets 0.0003.
                # Pinned to the SAC value so the baseline and the
                # constrained agent share an optimiser setting.
                "lr": 0.0003,
            },
        },
        "lagrange_cfgs": {
            "cost_limit": args.cost_limit,
            "lagrangian_multiplier_init": 0.001,
            # PID gains default to kp 1e-6, ki 1e-7, kd 1e-7. The 21 July
            # note said not to retune pre emptively. That no longer holds.
            # ddpg_pid.py calls pid_update from _update, which runs once
            # per gradient step rather than once per epoch, so the integral
            # accumulates thousands of times against one epoch level Jc.
            # The 28 July anneal run showed lambda still climbing while the
            # error shrank to 2, which is windup, not proportional
            # response. Gains are now settable via --pid_kp/ki/kd.
            # Unset means default, so earlier runs reproduce exactly.
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": True,
            "save_model_freq": 10,
            "log_dir": "./runs_" + args.tag,
            "window_lens": 10,
        },
    }

    for _name, _val in (("pid_kp", args.pid_kp),
                        ("pid_ki", args.pid_ki),
                        ("pid_kd", args.pid_kd)):
        if _val is not None:
            custom_cfgs["lagrange_cfgs"][_name] = _val

    print("=" * 60)
    print("Condition 3: PID Lagrangian SAC, %s force hinge" % args.cost_fn)
    print("PID gains: kp=%s  ki=%s  kd=%s"
          % (custom_cfgs["lagrange_cfgs"].get("pid_kp", "default 1e-6"),
             custom_cfgs["lagrange_cfgs"].get("pid_ki", "default 1e-7"),
             custom_cfgs["lagrange_cfgs"].get("pid_kd", "default 1e-7")))
    print("d (cost_limit): %g newton steps   warmup_epochs: %d of %d"
          % (args.cost_limit, args.warmup_epochs, n_epochs))
    print("cost fn: %s  threshold %g N  ceiling %g N"
          % (_ACTIVE_COST_FN.name, args.threshold, args.ceiling))
    print("steps: %d   seed: %d   tag: %s" % (args.steps, args.seed, args.tag))
    print("device: %s" % custom_cfgs["train_cfgs"]["device"])
    print("cost_normalize: False   actor lr: 0.0003 (pinned to the SAC value)")
    print("=" * 60)
    print("episode log: %s" % EPISODE_LOG_PATH)
    sys.stdout.flush()

    agent = omnisafe.Agent(
        algo="SACPID",
        env_id="SteveNav-v0",
        custom_cfgs=custom_cfgs,
    )

    alg = agent.agent
    lag = alg._lagrange
    _orig_pid_update = lag.pid_update
    _state = {"epoch": -1, "calls": 0, "prev": 0}
    w = args.warmup_epochs

    if args.anneal_from is not None:
        e_end = args.anneal_end_epoch
        d0 = args.anneal_from
        d1 = args.cost_limit

        def _d_for_epoch(e):
            if e <= w:
                return d0
            if e >= e_end:
                return d1
            frac = (e - w) / float(e_end - w)
            return d0 + (d1 - d0) * frac
    else:
        def _d_for_epoch(e):
            return args.cost_limit

    def _monitored_pid_update(ep_cost_avg):
        e = alg._epoch
        d = _d_for_epoch(e)
        lag._cost_limit = d
        if e != _state["epoch"]:
            _state["prev"] = _state["calls"]
            _state["calls"] = 0
            _state["epoch"] = e
            if args.anneal_from is not None:
                # Format unchanged so read_anneal.py keeps working.
                print("ANNEAL epoch %d  d=%.2f  ep_cost=%.2f  lambda=%.6f"
                      % (e, d, ep_cost_avg, lag.lagrangian_multiplier),
                      flush=True)
            else:
                print("LAG epoch %d  d=%.2f  ep_cost=%.2f  lambda=%.6f  "
                      "updates_prev_epoch=%d"
                      % (e, d, ep_cost_avg, lag.lagrangian_multiplier,
                         _state["prev"]),
                      flush=True)
        _state["calls"] += 1
        return _orig_pid_update(ep_cost_avg)

    lag.pid_update = _monitored_pid_update

    if args.anneal_from is not None:
        print("ANNEALING ON: d %g -> %g, held through warmup epoch %d, "
              "reaching %g at epoch %d"
              % (args.anneal_from, args.cost_limit, w,
                 args.cost_limit, args.anneal_end_epoch))
    else:
        print("ANNEALING OFF: d fixed at %g" % args.cost_limit)

    agent.learn()
    print("done: %s" % args.tag)


if __name__ == "__main__":
    main()
