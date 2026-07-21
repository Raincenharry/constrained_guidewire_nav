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

def cf_max():
    d = getattr(simulation.root.LCP, "constraintForces", None)
    if d is None:
        return 0.0
    a = np.array(d.value)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a)))

print("resetting...")
env.reset()
env.step(np.array([[0.0, 0.0]]))
dt = float(simulation.root.dt.value)
print("dt =", dt)

print("=== creep to contact ===")
contacted = False
for step in range(300):
    env.step(np.array([[2.0, 0.0]]))
    if cf_max() > 1e-9:
        print("first contact at step", step)
        contacted = True
        break
if not contacted:
    print("no contact reached"); env.close(); raise SystemExit

print("=== hold steady loads, read lambda and conversions ===")
print("%-6s %12s %12s %12s" % ("push", "lambda", "lambda/dt", "lambda*dt"))
for v in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
    vals = []
    for _ in range(8):
        env.step(np.array([[v, 0.0]]))
        lam = cf_max()
        if lam > 1e-9:
            vals.append(lam)
    if vals:
        lam = float(np.median(vals))
        print("%-6.1f %12.5f %12.5f %12.5f" % (v, lam, lam/dt, lam*dt))
    else:
        print("%-6.1f   (contact lost)" % v)

env.close()
