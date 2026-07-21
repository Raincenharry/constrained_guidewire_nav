import numpy as np
import eve
import Sofa.Gui
from force_read import force_N

def per_node_force():
    mo = simulation._instruments_combined.DOFs
    ndofs = 6
    dt = float(simulation.root.dt.value)
    lam = np.array(simulation.root.LCP.constraintForces.value)
    if lam.size == 0:
        return {}
    node_force = {}
    for r in mo.constraint.value.splitlines():
        p = r.split()
        if len(p) < 6:
            continue
        p = [float(x) for x in p]
        cid = int(p[0]); ne = int(p[1])
        if cid >= lam.size:
            continue
        L = lam[cid]
        base = 2
        for _ in range(ne):
            node = int(p[base])
            d = np.array(p[base+1:base+4])
            f = node_force.get(node, np.zeros(3))
            node_force[node] = f + L * d
            base += 1 + ndofs
    return {n: np.linalg.norm(v)/dt for n, v in node_force.items()}

vessel_tree = eve.intervention.vesseltree.AorticArch(seed=30, scaling_xyzd=[1.0, 1.0, 1.0, 0.75])
device = eve.intervention.device.JShaped()
simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
simulation.init_visual_nodes = True
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

class Pusher(Sofa.Core.Controller):
    def __init__(self, *a, **kw):
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.i = 0
        self.mode = "push"
        self.hold = 0
    def onAnimateBeginEvent(self, event):
        f = force_N(simulation)
        if self.mode == "push" and f > 0.6:
            self.mode = "rotate"
            self.hold = 300
        elif self.mode == "rotate":
            self.hold -= 1
            if self.hold <= 0:
                self.mode = "push"
        trans = 10.0 if self.mode == "push" else 0.0
        rot = 0.0 if self.mode == "push" else 90.0
        ctrl = simulation._instruments_combined.m_ircontroller
        dt = float(simulation.root.dt.value)
        x_tip = ctrl.xtip
        tip_rot = ctrl.rotationInstrument
        x_tip[0] += trans * dt
        tip_rot[0] += rot * dt
        ctrl.xtip = x_tip
        ctrl.rotationInstrument = tip_rot
    def onAnimateEndEvent(self, event):
        if self.i % 50 == 0:
            mags = per_node_force()
            if mags:
                top = sorted(mags.items(), key=lambda kv: -kv[1])[:3]
                tip = mags.get(61, 0.0)
                top_str = "  ".join("n%d=%.2f" % (n, v) for n, v in top)
                print("step %5d  mode=%-6s  tip(n61)=%.2f  loaded: %s" % (self.i, self.mode, tip, top_str))
            else:
                print("step %5d  mode=%-6s  no contact" % (self.i, self.mode))
        self.i += 1

simulation.root.addObject(Pusher(name="pusher"))

Sofa.Gui.GUIManager.Init("gui_run", "qt")
Sofa.Gui.GUIManager.createGUI(simulation.root, __file__)
Sofa.Gui.GUIManager.SetDimension(1200, 800)
Sofa.Gui.GUIManager.MainLoop(simulation.root)
Sofa.Gui.GUIManager.closeGUI()
