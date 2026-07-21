"""
OmniSafe SACPID smoke test.

Runs a very short training loop to verify the adapter integrates with
OmniSafe. Not a real training run; the goal is only to confirm the pipeline
does not crash and the multiplier behaves.

Two runs:
  1. ZeroCost: lambda should stay near zero, cost near zero.
  2. MaxAlongDeviceHinge: lambda should react to cost.
"""

import os
import sys
import torch
import omnisafe

# Import the adapter module so the @env_register decorator fires
import steve_cmdp
from steve_cmdp import SteveCMDP, ZeroCost, MaxAlongDeviceHinge

# ============================================================
# We need OmniSafe to instantiate SteveCMDP with a specific cost fn.
# The @env_register system builds env from env_id alone by default.
# The simplest robust hook: set a module level cost fn that SteveCMDP
# reads at construction. Do this by patching SteveCMDP.__init__ to
# pick up a module attribute if no cost_fn was passed.
# ============================================================

_ACTIVE_COST_FN = None

_original_init = SteveCMDP.__init__

def _patched_init(self, env_id, num_envs=1, device=torch.device("cpu"),
                  cost_fn=None, seed=30, **kwargs):
    if cost_fn is None:
        cost_fn = _ACTIVE_COST_FN
    _original_init(self, env_id, num_envs=num_envs, device=device,
                   cost_fn=cost_fn, seed=seed, **kwargs)

SteveCMDP.__init__ = _patched_init

# ============================================================
# Config for a very short run
# ============================================================

custom_cfgs = {
    "train_cfgs": {
        "total_steps": 5000,
        "vector_env_nums": 1,
        "parallel": 1,
        "device": "cpu",
    },
    "algo_cfgs": {
        "steps_per_epoch": 500,
        "update_iters": 1,
        "batch_size": 64,
        "gamma": 0.99,
        "polyak": 0.005,
        "start_learning_steps": 200,
        "warmup_epochs": 1,
        "cost_normalize": False,
    },
    "lagrange_cfgs": {
        "cost_limit": 25.0,
        "lagrangian_multiplier_init": 0.001,
    },
    "logger_cfgs": {
        "use_wandb": False,
        "use_tensorboard": True,
        "save_model_freq": 1000,
        "log_dir": "./omnisafe_runs",
        "window_lens": 100,
    },
}


def run(cost_fn, tag):
    global _ACTIVE_COST_FN
    _ACTIVE_COST_FN = cost_fn
    print("=" * 60)
    print("SACPID smoke run: %s" % tag)
    print("cost fn: %s" % cost_fn.name)
    print("=" * 60)

    agent = omnisafe.Agent(
        algo="SACPID",
        env_id="SteveNav-v0",
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
    print("done: %s" % tag)
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "zero"
    if which == "zero":
        run(ZeroCost(), "zero_cost")
    elif which == "hinge":
        run(MaxAlongDeviceHinge(threshold=0.85, ceiling=5.0), "hinge_cost")
    else:
        print("usage: python train_smoke.py [zero|hinge]")
        sys.exit(1)