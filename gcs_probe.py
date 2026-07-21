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
max_steps = eve.truncation.MaxSteps(400)
env = eve.Env(intervention=intervention, observation=state, reward=reward, terminal=target_reached, truncation=max_steps, start=start, pathfinder=pathfinder)

def cf():
    s = simulation.root.LCP
    d = getattr(s, "constraintForces", None)
    if d is None:
        return None
    return np.array(d.value)

def tiprow():
    m = simulation._instruments_combined.collision_monitor.sortedCollisionMatrix
    act = [i for i,r in enumerate(m) if np.linalg.norm(r) > 1e-6]
    if not act:
        return None, 0.0
    idx = max(act)
    return idx, float(np.linalg.norm(m[idx]))

print("resetting...")
env.reset()
env.step(np.array([[0.0, 0.0]]))
print("solver type:", simulation.root.LCP.getClassName())
c = cf()
print("constraintForces present:", c is not None, " len:", (len(c) if c is not None else "n/a"))

print("=== creep to contact ===")
for step in range(250):
    env.step(np.array([[2.0, 0.0]]))
    idx, tn = tiprow()
    c = cf()
    cnorm = float(np.linalg.norm(c)) if c is not None and len(c) else 0.0
    cmax = float(np.max(np.abs(c))) if c is not None and len(c) else 0.0
    if idx is not None:
        print("FIRST CONTACT step %d  tipnode=%s tiprow=%.4f  |cf|=%.4f  max|cf|=%.4f  ncf=%s" % (step, idx, tn, cnorm, cmax, (len(c) if c is not None else 0)))
        break

print("=== gentle ramp, does |cf| scale ===")
for v in [1.0, 2.0, 3.0, 5.0, 8.0]:
    for _ in range(4):
        env.step(np.array([[v, 0.0]]))
        idx, tn = tiprow()
        c = cf()
        cnorm = float(np.linalg.norm(c)) if c is not None and len(c) else 0.0
        cmax = float(np.max(np.abs(c))) if c is not None and len(c) else 0.0
        print("v=%5.1f  tipnode=%s tiprow=%7.4f  |cf|=%9.4f  max|cf|=%8.4f" % (v, idx, (tn or 0.0), cnorm, cmax))

env.close()
