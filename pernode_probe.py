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
env.reset()

def per_node_force():
    mo = simulation._instruments_combined.DOFs
    ndofs = 6
    dt = float(simulation.root.dt.value)
    lam = np.array(simulation.root.LCP.constraintForces.value)
    if lam.size == 0:
        return None, 0.0
    rows = mo.constraint.value.splitlines()
    node_force = {}
    for r in rows:
        p = r.split()
        if len(p) < 6:
            continue
        p = [float(x) for x in p]
        cid = int(p[0])
        ne = int(p[1])
        if cid >= lam.size:
            continue
        L = lam[cid]
        offs = []
        base = 2
        for _ in range(ne):
            node = int(p[base])
            d = np.array(p[base+1:base+4])
            offs.append((node, d))
            base += 1 + ndofs
        for node, d in offs:
            f = node_force.get(node, np.zeros(3))
            node_force[node] = f + L * d
    mags = {n: np.linalg.norm(v)/dt for n, v in node_force.items()}
    return mags, (max(abs(lam))/dt)

for i in range(1600):
    env.step(np.array([[20.0, 0.0]]))
    if i % 100 == 0 or (i > 1450):
        mags, gmax = per_node_force()
        if mags:
            top = sorted(mags.items(), key=lambda kv: -kv[1])[:3]
            top_str = "  ".join("n%d=%.3f" % (n, v) for n, v in top)
            join_max = max(mags.values())
            print("step %4d  global_max=%.3f  join_max=%.3f  top: %s" % (i, gmax, join_max, top_str))
        else:
            print("step %4d  no contact" % i)
env.close()
