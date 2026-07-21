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

def mat():
    return simulation._instruments_combined.collision_monitor.sortedCollisionMatrix

print("resetting...")
env.reset()
env.step(np.array([[0.0, 0.0]]))
print("reset ok, array length =", len(mat()))

def active_nodes(m):
    out = []
    for i, row in enumerate(m):
        n = float(np.linalg.norm(row))
        if n > 1e-6:
            out.append((i, round(n, 4)))
    return out

print("=== gentle creep, report which nodes are active on contact ===")
first_seen = None
for step in range(250):
    env.step(np.array([[2.0, 0.0]]))
    m = mat()
    act = active_nodes(m)
    if act:
        if first_seen is None:
            first_seen = step
            L = len(m)
            print("FIRST CONTACT at step", step, " array len", L)
            print("  active (index, |row|):", act)
            print("  last index L-1 =", L-1, " its norm =", round(float(np.linalg.norm(m[-1])), 4))
        if step - first_seen >= 0 and step - first_seen <= 8:
            print("step %3d  active=%s" % (step, act))
        if step - first_seen > 8:
            break

print("=== now ramp gently on the node that actually carries contact ===")
# track the highest-index active node (closest to tip) and watch it scale
for v in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
    for _ in range(4):
        env.step(np.array([[v, 0.0]]))
        m = mat()
        act = active_nodes(m)
        if act:
            tip_idx = max(a[0] for a in act)
            tip_norm = round(float(np.linalg.norm(m[tip_idx])), 4)
            print("v=%5.1f  tipnode=%3d  |row|=%7.4f  n_active=%d" % (v, tip_idx, tip_norm, len(act)))
        else:
            print("v=%5.1f  (no contact)" % v)

env.close()
