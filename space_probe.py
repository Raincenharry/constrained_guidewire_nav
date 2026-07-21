import numpy as np
import eve

vessel_tree = eve.intervention.vesseltree.AorticArch(seed=30, scaling_xyzd=[1.0, 1.0, 1.0, 0.75])
device = eve.intervention.device.JShaped()
simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(simulation=simulation, vessel_tree=vessel_tree, image_frequency=7.5, image_rot_zx=[20, 5])
target = eve.intervention.target.CenterlineRandom(vessel_tree=vessel_tree, fluoroscopy=fluoroscopy, threshold=5, branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"])
intervention = eve.intervention.MonoPlaneStatic(vessel_tree=vessel_tree, devices=[device], simulation=simulation, fluoroscopy=fluoroscopy, target=target)
start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
target_state = eve.observation.Target2D(intervention=intervention)
target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(target_state, intervention)
rotation = eve.observation.Rotations(intervention=intervention)
state = eve.observation.ObsDict({"position": position, "target": target_state, "rotation": rotation})
target_reward = eve.reward.TargetReached(intervention=intervention, factor=1.0)
path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.01)
reward = eve.reward.Combination([target_reward, path_delta])
target_reached = eve.terminal.TargetReached(intervention=intervention)
max_steps = eve.truncation.MaxSteps(600)
env = eve.Env(intervention=intervention, observation=state, reward=reward, terminal=target_reached, truncation=max_steps, start=start, pathfinder=pathfinder)

print("=== OBSERVATION SPACE ===")
print(type(env.observation_space))
print(env.observation_space)
print()

print("=== ACTION SPACE ===")
print(type(env.action_space))
print(env.action_space)
print()

obs, info = env.reset()
print("=== OBS AFTER RESET ===")
print("type:", type(obs))
if isinstance(obs, dict):
    for k, v in obs.items():
        a = np.asarray(v)
        print("  key=%-10s shape=%-12s dtype=%-10s min=%.4f max=%.4f" % (k, str(a.shape), str(a.dtype), a.min(), a.max()))
else:
    a = np.asarray(obs)
    print("  shape=%s dtype=%s" % (a.shape, a.dtype))
print()

a = env.action_space.sample()
print("=== ACTION SAMPLE ===")
print("type:", type(a), "shape:", np.asarray(a).shape, "dtype:", np.asarray(a).dtype)
print(a)
print()

out = env.step(np.asarray(a))
print("=== STEP RETURN ===")
print("length:", len(out))
for i, item in enumerate(out):
    print("  [%d] %s" % (i, type(item)))
print()

print("=== INFO KEYS ===")
print(list(out[-1].keys()) if isinstance(out[-1], dict) else out[-1])

print("=== INFO KEYS ===")
print(list(out[-1].keys()) if isinstance(out[-1], dict) else out[-1])
print()

print("=== DRIVING TO CONTACT ===")
push_action = np.zeros_like(a)
push_action[0, 0] = 40.0
push_action[0, 1] = 0.0
contact_step = None
for i in range(120):
    out = env.step(push_action)
    mo = simulation._instruments_combined.DOFs
    rows = mo.constraint.value.splitlines()
    if rows:
        contact_step = i
        print("first contact at step %d, %d rows" % (i, len(rows)))
        break
if contact_step is None:
    print("no contact after 120 steps")

print()
print("=== CONSTRAINT MATRIX INSPECTION ===")
mo = simulation._instruments_combined.DOFs
rows = mo.constraint.value.splitlines()
print("num rows:", len(rows))
for i, r in enumerate(rows[:5]):
    parts = r.split()
    print("row %d: len=%d" % (i, len(parts)))
    print("       first 12 tokens:", parts[:12])
print()

print("=== LAMBDA VECTOR ===")
lam = np.array(simulation.root.LCP.constraintForces.value)
print("shape:", lam.shape, "dtype:", lam.dtype)
print("nonzero count:", int(np.sum(lam != 0)))
print("max abs:", float(np.max(np.abs(lam))) if lam.size else 0.0)
print()

print("=== DT ===")
print("dt:", float(simulation.root.dt.value))

env.close()