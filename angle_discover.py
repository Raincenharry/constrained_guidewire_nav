"""Discovery only. Prints what geometry is readable, so the real angle probe
can be written against facts rather than assumptions. Writes nothing."""
import numpy as np
from steve_cmdp import build_steve_env
from eve.intervention.vesseltree.aorticarch import ArchType
from force_read import per_node_forces

env = build_steve_env(seed=1, arch_type=ArchType.I)
env.reset()
sim = env.intervention.simulation

# push forward a few steps so something is in the vessel
for _ in range(30):
    env.step(np.array([[30.0, 0.0]], dtype=np.float32))

print("=" * 60)
print("1. MechanicalObject DOFs")
mo = sim._instruments_combined.DOFs
pos = np.array(mo.position.value)
print("   position.value shape:", pos.shape)
print("   first row:", np.round(pos[0], 3))
print("   last  row:", np.round(pos[-1], 3))
print("   second last:", np.round(pos[-2], 3))

print()
print("2. Force node indices, for the phantom node check")
f = per_node_forces(sim)
print("   nodes with force:", sorted(f.keys())[:5], "...", sorted(f.keys())[-5:] if f else "none")
print("   n position rows:", pos.shape[0])

print()
print("3. Inserted length")
print("   device_lengths_inserted:", env.intervention.device_lengths_inserted)

print()
print("4. Fluoroscopy attributes")
fl = env.intervention.fluoroscopy
for a in sorted(x for x in dir(fl) if not x.startswith("_")):
    try:
        v = getattr(fl, a)
    except Exception as e:
        print("   %-28s ERROR %s" % (a, type(e).__name__)); continue
    if isinstance(v, np.ndarray):
        print("   %-28s ndarray %s" % (a, v.shape))
    elif isinstance(v, (int, float, str, bool, tuple, list)):
        print("   %-28s %s" % (a, str(v)[:60]))

print()
print("5. Vessel tree attributes")
vt = env.intervention.vessel_tree
for a in sorted(x for x in dir(vt) if not x.startswith("_")):
    try:
        v = getattr(vt, a)
    except Exception:
        continue
    if isinstance(v, np.ndarray):
        print("   %-28s ndarray %s" % (a, v.shape))

print()
print("6. Centerline sample, if present")
for name in ("centerline_coordinates", "centerlines"):
    if hasattr(vt, name):
        c = getattr(vt, name)
        c = np.asarray(c[0] if isinstance(c, (list, tuple)) else c)
        print("   %s -> %s" % (name, c.shape))
        print("   first three points:\n", np.round(c[:3], 2))
        break
else:
    print("   no centerline attribute found")
print("=" * 60)
