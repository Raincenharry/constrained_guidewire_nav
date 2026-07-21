import numpy as np

NDOFS = 6

def _read_lambda(simulation):
    return np.array(simulation.root.LCP.constraintForces.value)

def _read_dt(simulation):
    return float(simulation.root.dt.value)

def _per_node_vectors(simulation):
    mo = simulation._instruments_combined.DOFs
    lam = _read_lambda(simulation)
    if lam.size == 0:
        return {}
    node_force = {}
    for r in mo.constraint.value.splitlines():
        parts = r.split()
        if len(parts) < 2:
            continue
        cid = int(parts[0])
        ne = int(parts[1])
        if cid >= lam.size:
            continue
        L = lam[cid]
        base = 2
        for _ in range(ne):
            if base + 1 + NDOFS > len(parts):
                break
            node = int(parts[base])
            d = np.array([float(parts[base + 1 + k]) for k in range(3)])
            v = node_force.get(node, np.zeros(3))
            node_force[node] = v + L * d
            base += 1 + NDOFS
    return node_force

def per_node_forces(simulation):
    dt = _read_dt(simulation)
    vecs = _per_node_vectors(simulation)
    return {n: float(np.linalg.norm(v) / dt) for n, v in vecs.items()}

def tip_force(simulation):
    mags = per_node_forces(simulation)
    if not mags:
        return 0.0
    tip_node = max(mags.keys())
    return mags[tip_node]

def max_along_device(simulation):
    mags = per_node_forces(simulation)
    if not mags:
        return 0.0
    return max(mags.values())

def force_N(simulation):
    dt = _read_dt(simulation)
    lam = _read_lambda(simulation)
    if lam.size == 0:
        return 0.0
    return float(np.max(np.abs(lam)) / dt)

def assert_force_available(simulation):
    root = simulation.root
    if not hasattr(root, "LCP"):
        raise RuntimeError("LCP node not present on simulation root; GenericConstraintSolver required")
    try:
        _ = root.LCP.constraintForces.value
    except Exception as e:
        raise RuntimeError("LCP.constraintForces not readable; check computeConstraintForces=True") from e