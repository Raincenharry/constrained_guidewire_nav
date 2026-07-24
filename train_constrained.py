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
from steve_cmdp import SteveCMDP, MaxAlongDeviceHinge


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
    p.add_argument("--start_learning_steps", type=int, default=10000,
                   help="matches train_baseline.py. Lower it only for local "
                        "wiring checks that are too short to reach it.")
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

    _ACTIVE_COST_FN = MaxAlongDeviceHinge(threshold=args.threshold,
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
            # PID gains left at the OmniSafe defaults. They were validated
            # in the 20 July smoke run against episode costs in the
            # hundreds, which is the regime here. Do not retune pre emptively.
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": True,
            "save_model_freq": 10,
            "log_dir": "./runs_" + args.tag,
            "window_lens": 10,
        },
    }

    print("=" * 60)
    print("Condition 3: PID Lagrangian SAC, max along device force hinge")
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
    agent.learn()
    print("done: %s" % args.tag)


if __name__ == "__main__":
    main()
