"""Per step log of tip angle against force, to test the Experiment B premise.

Writes one row per simulation step. The question this answers is whether tip
angle rises before force does. If it does not, condition 4 is dropped.

Two angle definitions, both logged, because they mean different things:
  angle_bend       local curvature at the tip, device geometry only
  angle_centerline tip heading against the local lumen direction, needs anatomy

Both are privileged information, which is fine: like force, they are only used
in the cost during training and never at inference.
"""
import argparse
import csv
import numpy as np
import torch

from steve_cmdp import SteveCMDP
from force_read import max_along_device, tip_force, per_node_forces

import evaluate as ev


def tip_index(xyz: np.ndarray) -> int:
    """Highest index that is not a duplicate of its neighbour.

    The nx+1 topology can leave a phantom node at the end. Never use [-1]
    blindly, which is the rule already learned on the force side."""
    i = len(xyz) - 1
    while i > 3 and np.linalg.norm(xyz[i] - xyz[i - 1]) < 1e-9:
        i -= 1
    return i


def angle_between(u: np.ndarray, v: np.ndarray, fold: bool = False) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    if fold:
        c = abs(c)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def geometry(sim, centerline: np.ndarray) -> dict:
    pos = np.array(sim._instruments_combined.DOFs.position.value)
    xyz = pos[:, :3]
    i = tip_index(xyz)

    # Local bend, one segment either side of the last node
    bend = angle_between(xyz[i] - xyz[i - 1], xyz[i - 1] - xyz[i - 2])

    # Same thing over a longer baseline, less sensitive to node spacing noise
    j = max(i - 5, 1)
    k = max(i - 10, 0)
    bend_span = angle_between(xyz[i] - xyz[j], xyz[j] - xyz[k])

    # Tip heading against the local centerline tangent.
    # fold=True because the centerline has no inherent direction, so only
    # the acute angle is meaningful, giving a range of 0 to 90 degrees.
    heading = xyz[i] - xyz[max(i - 3, 0)]
    d = np.linalg.norm(centerline - xyz[i], axis=1)
    n = int(np.argmin(d))
    a = max(n - 3, 0)
    b = min(n + 3, len(centerline) - 1)
    tangent = centerline[b] - centerline[a]
    vs_centerline = angle_between(heading, tangent, fold=True)

    return {
        "angle_bend": bend,
        "angle_bend_span": bend_span,
        "angle_centerline": vs_centerline,
        "dist_to_centerline": float(d[n]),
        "tip_index": i,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ckpt/torch_save/epoch-50.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", default="angle_trace.csv")
    args = p.parse_args()

    cfg = ev.find_config(args.checkpoint)
    env = SteveCMDP("SteveNav-v0", train=True, seed=args.seed)
    env.set_seed(args.seed)

    # Same hand rescale as evaluate.py. Outside OmniSafe there is no
    # ActionScale wrapper, so a full push of 1.0 would reach stEVE as 1 mm/s.
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    print("action bounds: %s to %s" % (env.action_space.low, env.action_space.high))

    def scale_action(a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(-1.0, 1.0)
        return act_low + (a + 1.0) * 0.5 * (act_high - act_low)

    actor = ev.build_actor(env, cfg)
    ev.load_actor_weights(actor, args.checkpoint)



    fields = ["episode", "step", "force_max", "force_tip", "angle_bend",
              "angle_bend_span", "angle_centerline", "dist_to_centerline",
              "tip_index",  "tip_node_used", "n_contact_nodes","inserted", "arch_type"]
    fh = open(args.out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        centerline = np.asarray(
            env._env.intervention.vessel_tree.centerline_coordinates, dtype=np.float64)
        for t in range(env.max_episode_steps):
            action = scale_action(ev.deterministic_action(actor, obs))
            obs, reward, cost, term, trunc, info = env.step(action)
            sim = env._env.intervention.simulation
            row = geometry(sim, centerline)
            pnf = per_node_forces(sim)
            row.update({
                "episode": ep,
                "step": t,
                "force_max": max_along_device(sim),
                "force_tip": tip_force(sim),
                "inserted": float(
                    np.asarray(env._env.intervention.device_lengths_inserted).max()),
                "arch_type": str(env._arch_type),
                "tip_node_used": max(pnf.keys()) if pnf else -1,
                "n_contact_nodes": len(pnf),
            })
            w.writerow(row)
            if bool(term) or bool(trunc):
                break
        fh.flush()
        print("episode %d done" % ep)

    fh.close()
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
