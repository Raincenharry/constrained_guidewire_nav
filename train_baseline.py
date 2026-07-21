"""
Plain SAC baseline with R1 reward and no safety mechanism.
Condition 1 of the experimental design.

Purpose of this run: find where success plateaus on the arch task,
to set the training budget for all later conditions.
"""

import os
import sys
import argparse
import torch
import omnisafe

import steve_cmdp
from steve_cmdp import SteveCMDP, ZeroCost


_ACTIVE_COST_FN = None
_original_init = SteveCMDP.__init__


def _patched_init(self, env_id, num_envs=1, device=torch.device("cpu"),
                  cost_fn=None, seed=30, **kwargs):
    if cost_fn is None:
        cost_fn = _ACTIVE_COST_FN
    _original_init(self, env_id, num_envs=num_envs, device=device,
                   cost_fn=cost_fn, seed=seed, **kwargs)


SteveCMDP.__init__ = _patched_init


def main():
    global _ACTIVE_COST_FN

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="r1_baseline")
    args = p.parse_args()

    _ACTIVE_COST_FN = ZeroCost()

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
            "steps_per_epoch": 6000,
            "update_iters": 1,
            "batch_size": 256,
            "gamma": 0.99,
            "polyak": 0.005,
            "start_learning_steps": 10000,
            "use_cost": False,
            "auto_alpha": True,
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
    print("R1 baseline, plain SAC, no safety")
    print("steps: %d   seed: %d   tag: %s" % (args.steps, args.seed, args.tag))
    print("device: %s" % custom_cfgs["train_cfgs"]["device"])
    print("=" * 60)
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
