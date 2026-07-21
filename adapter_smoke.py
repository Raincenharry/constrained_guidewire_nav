"""
Adapter smoke test. No OmniSafe training loop, no SACPID, no learning.
Just verify: construct the adapter, reset, step, get sensible shapes and dtypes.
"""

import numpy as np
import torch
from steve_cmdp import SteveCMDP, ZeroCost, MaxAlongDeviceHinge, flatten_obs, OBS_KEY_ORDER

print("=== CONSTRUCT ADAPTER WITH ZERO COST ===")
env = SteveCMDP(env_id="SteveNav-v0", cost_fn=ZeroCost(), seed=30)
print("obs space:", env.observation_space)
print("action space:", env.action_space)
print("cost fn:", env._cost_fn.name)
print()

print("=== RESET ===")
obs, info = env.reset()
print("obs type:", type(obs), "shape:", tuple(obs.shape), "dtype:", obs.dtype)
print("obs min:", float(obs.min()), "max:", float(obs.max()))
print()

print("=== FLATTEN ROUND TRIP ===")
raw = env._last_obs_dict
flat = flatten_obs(raw)
print("raw keys:", list(raw.keys()))
for k in OBS_KEY_ORDER:
    print("  ", k, raw[k].shape)
print("flat shape:", flat.shape, "sum:", float(flat.sum()))
print("torch obs matches numpy flatten:", np.allclose(obs.cpu().numpy(), flat))
print()

print("=== 5 RANDOM STEPS (zero cost) ===")
for i in range(5):
    a = env.sample_action()
    obs, rew, cost, term, trunc, info = env.step(a)
    print("step %d  rew=%+.4f  cost=%.4f  fmax=%.4f  ftip=%.4f  term=%s trunc=%s" % (
        i, float(rew), float(cost),
        info["force_max_along_device"], info["force_tip"],
        bool(term), bool(trunc),
    ))
print()

print("=== NOW DRIVE INTO WALL WITH HINGE COST ===")
env.close()
env2 = SteveCMDP(env_id="SteveNav-v0", cost_fn=MaxAlongDeviceHinge(threshold=0.85), seed=30)
obs, info = env2.reset()
print("cost fn:", env2._cost_fn.name)

# Constant hard push
push = torch.zeros((2,), dtype=torch.float32)
push[0] = 40.0
for i in range(60):
    obs, rew, cost, term, trunc, info = env2.step(push)
    if i % 5 == 0 or float(cost) > 0:
        print("step %2d  rew=%+.4f  cost=%.4f  fmax=%.4f  ftip=%.4f" % (
            i, float(rew), float(cost),
            info["force_max_along_device"], info["force_tip"],
        ))
    if bool(term) or bool(trunc):
        print("episode ended, term=%s trunc=%s" % (bool(term), bool(trunc)))
        break

env2.close()
print()
print("adapter smoke test done")