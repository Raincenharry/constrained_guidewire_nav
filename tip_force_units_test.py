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
max_steps = eve.truncation.MaxSteps(200)
env = eve.Env(intervention=intervention, observation=state, reward=reward, terminal=target_reached, truncation=max_steps, start=start, pathfinder=pathfinder)

def read_tip_force(sim):
    """Report tip (node 0) force and the max force over all nodes."""
    try:
        monitor = sim._instruments_combined.collision_monitor
        m = monitor.sortedCollisionMatrix
        if m is None or len(m) == 0:
            return 0.0, 0.0, -1
        import numpy as _np
        mags = [float(_np.linalg.norm(row)) for row in m]
        tip = mags[-1]
        mx = max(mags)
        mx_node = mags.index(mx)
        return tip, mx, mx_node
    except AttributeError:
        return 0.0, 0.0, -1
        return float(np.linalg.norm(m[0]))
    except AttributeError:
        return 0.0

print("resetting (builds SOFA scene)...")
obs, info = env.reset()
print("reset ok")

print("=== creep to first contact (gentle, trans=2.0) ===")
step = 0
contact = None
for _ in range(200):
    obs, reward, terminal, truncated, info = env.step(np.array([[2.0, 0.0]]))
    tip, mx, mxn = read_tip_force(simulation)
    print("creep %3d  tip=%7.4f  max=%7.4f @node%3d" % (step, tip, mx, mxn))
    step += 1
    if tip > 1e-6:
        contact = tip
        print(">>> FIRST CONTACT (gentle) tip=%.4f  max=%.4f @node%d" % (tip, mx, mxn))
        break

print("=== ramp the push, watch tip force grow or pin ===")
for v in [2.0, 5.0, 10.0, 20.0, 40.0]:
    for _ in range(6):
        obs, reward, terminal, truncated, info = env.step(np.array([[v, 0.0]]))
        tip, mx, mxn = read_tip_force(simulation)
        print("ramp  %3d  v=%5.1f  tip=%7.4f  max=%7.4f @node%3d" % (step, v, tip, mx, mxn))
        step += 1

print("=== done === gentle first contact tip force:", contact)
env.close()
