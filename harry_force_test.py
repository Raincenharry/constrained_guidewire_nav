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

def harry_read():
    dev_forces = simulation._instruments_combined.collision_monitor.sortedCollisionMatrix
    dev_tip_force = dev_forces[-1]
    s = np.array([0.0, 0.0, 0.0])
    for i in range(len(dev_forces)):
        s[0] += dev_forces[i][0]
        s[1] += dev_forces[i][1]
        s[2] += dev_forces[i][2]
    tip_mag = float(np.linalg.norm(dev_tip_force))
    sum_mag = float(np.linalg.norm(s))
    return dev_tip_force, tip_mag, sum_mag

print("resetting (builds SOFA scene)...")
obs, info = env.reset()
print("reset ok")

print("=== creep gently to first contact (trans=2.0) ===")
step = 0
for _ in range(200):
    obs, reward, terminal, truncated, info = env.step(np.array([[2.0, 0.0]]))
    tipvec, tipmag, summag = harry_read()
    print("creep %3d  tip_vec=%s  |tip|=%7.4f  |sum|=%7.4f" % (step, np.round(tipvec, 4), tipmag, summag))
    step += 1
    if tipmag > 1e-6:
        print(">>> FIRST CONTACT  |tip|=%.4f  tip_vec=%s" % (tipmag, np.round(tipvec, 4)))
        break

print("=== ramp the push, watch |tip| grow or pin ===")
for v in [2.0, 5.0, 10.0, 20.0, 40.0]:
    for _ in range(6):
        obs, reward, terminal, truncated, info = env.step(np.array([[v, 0.0]]))
        tipvec, tipmag, summag = harry_read()
        print("ramp  %3d  v=%5.1f  |tip|=%7.4f  |sum|=%7.4f" % (step, v, tipmag, summag))
        step += 1

print("=== done ===")
env.close()
