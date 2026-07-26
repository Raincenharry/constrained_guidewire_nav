"""
Plain SAC baseline with R1 reward and no safety mechanism.
Condition 1 of the experimental design.

Purpose of this run: find where success plateaus on the arch task,
to set the training budget for all later conditions.
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
from steve_cmdp import SteveCMDP, ZeroCost, TipForceHinge


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
    p.add_argument("--tag", type=str, default="r1_baseline")
    p.add_argument("--steps_per_epoch", type=int, default=6000,
                   help="must divide --steps exactly, OmniSafe asserts this")
    p.add_argument("--force_penalty", type=float, default=0.0,
                   help="R4 weight on the force hinge in the reward. "
                        "0.0 is condition 1 (R1, no force awareness). "
                        "Nonzero is condition 2 (R4). The signal it hinges on "
                        "is chosen by --cost_fn.")
    p.add_argument("--cost_fn", type=str, default="max", choices=["max", "tip"],
                   help="Which force signal the R4 reward penalty hinges on. "
                        "SAC never uses the cost, so this selects the penalty "
                        "signal only. Must match the constrained condition or "
                        "the comparison stops being single variable.")
    args = p.parse_args()

    if args.steps % args.steps_per_epoch != 0:
        raise SystemExit(
            "steps (%d) must be divisible by steps_per_epoch (%d)"
            % (args.steps, args.steps_per_epoch)
        )

    # TipForceHinge here constrains nothing, SAC ignores the cost. It is what
    # makes steve_cmdp route the R4 penalty through tip force instead of max
    # along device, via the isinstance check in step().
    _ACTIVE_COST_FN = TipForceHinge() if args.cost_fn == "tip" else ZeroCost()
    _ACTIVE_PENALTY_WEIGHT = args.force_penalty

    print("condition: %s | penalty weight %.4f | penalty signal %s"
          % ("2 (R4)" if args.force_penalty else "1 (R1)",
             args.force_penalty, args.cost_fn))

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
            "start_learning_steps": 10000,
            "use_cost": False,
            # Entropy coefficient set by arithmetic rather than left to
            # auto_alpha, which initialises at 1.0 and burns roughly ten
            # epochs descending. Task return is order 1.0 per episode; at
            # gamma 0.99 the entropy term is about alpha * 175, so alpha
            # 0.0005 puts entropy near 10 percent of the task scale.
            # The 21 July auto_alpha run converged to 0.000147, which is
            # independent confirmation of this order of magnitude.
            "auto_alpha": False,
            "alpha": 0.0005,
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
    if args.force_penalty == 0.0:
        print("Condition 1: R1 baseline, plain SAC, no safety")
    else:
        print("Condition 2: R4, plain SAC, force penalty weight %g"
              % args.force_penalty)
    print("steps: %d   seed: %d   tag: %s" % (args.steps, args.seed, args.tag))
    print("device: %s" % custom_cfgs["train_cfgs"]["device"])
    print("=" * 60)
    print("episode log: %s" % EPISODE_LOG_PATH)
    sys.stdout.flush()

    agent = omnisafe.Agent(
        algo="SAC",
        env_id="SteveNav-v0",
        custom_cfgs=custom_cfgs,
        
    )
    agent.learn()
    print("done: %s" % args.tag)


if __name__ == "__main__":
    main()
