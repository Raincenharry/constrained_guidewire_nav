"""
check_pid_gate.py

Wiring check for the gated pid_update change.

What this proves, and it is deliberately narrow:

  A. Call count. After warmup, pid_update fires exactly once per epoch.
     During warmup it fires zero times.

  B. Input identity. The Jc value handed to pid_update is exactly the mean of
     the logger deque for Metrics/EpCost at the moment of the call. This is
     checked to machine precision against a value computed by a second,
     independent read of that deque inside the wrapper.

  C. Output identity. The multiplier trajectory produced during the run is
     reproduced to machine precision by replaying the logged Jc sequence
     through a fresh PIDLagrangian built from the run's own config.json.
     If this passes, the logged Jc sequence is the complete and only input
     to the multiplier, and nothing else in the code path is touching it.

Usage, on WSL:

    python check_pid_gate.py <path to slurm or console log> <path to config.json>

Input format. The training wrapper must print one line per pid_update call:

    PIDCALL epoch=%d call=%d jc=%.17g win=%.17g n=%d d=%.17g lam=%.17g

where
    epoch  alg._epoch at the moment of the call
    call   running count of pid_update calls since the run started
    jc     the value actually passed to pid_update
    win    mean of alg._logger._data['Metrics/EpCost'], computed separately
    n      len of that deque at the moment of the call
    d      lag._cost_limit at the moment of the call, so annealed runs replay
    lam    lag.lagrangian_multiplier read immediately AFTER the call

All floats are printed with %.17g so the replay is exact rather than close.
"""

import json
import re
import sys

from omnisafe.common.pid_lagrange import PIDLagrangian

LINE = re.compile(
    r"PIDCALL\s+epoch=(\d+)\s+call=(\d+)\s+jc=(\S+)\s+win=(\S+)\s+"
    r"n=(\d+)\s+d=(\S+)\s+lam=(\S+)"
)


def parse(log_path):
    rows = []
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                rows.append(
                    {
                        "epoch": int(m.group(1)),
                        "call": int(m.group(2)),
                        "jc": float(m.group(3)),
                        "win": float(m.group(4)),
                        "n": int(m.group(5)),
                        "d": float(m.group(6)),
                        "lam": float(m.group(7)),
                    }
                )
    return rows


def find_lagrange_cfgs(obj):
    """Return the lagrange_cfgs dict from a saved omnisafe config, at any depth."""
    if isinstance(obj, dict):
        if "lagrange_cfgs" in obj and isinstance(obj["lagrange_cfgs"], dict):
            return obj["lagrange_cfgs"]
        for v in obj.values():
            found = find_lagrange_cfgs(v)
            if found is not None:
                return found
    return None


def find_warmup(obj):
    if isinstance(obj, dict):
        if "warmup_epochs" in obj:
            return obj["warmup_epochs"]
        for v in obj.values():
            found = find_warmup(v)
            if found is not None:
                return found
    return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    log_path, cfg_path = sys.argv[1], sys.argv[2]

    rows = parse(log_path)
    if not rows:
        print("FAIL: no PIDCALL lines found in", log_path)
        sys.exit(1)

    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)

    lag_cfgs = find_lagrange_cfgs(cfg)
    if lag_cfgs is None:
        print("FAIL: no lagrange_cfgs found in", cfg_path)
        sys.exit(1)
    warmup = find_warmup(cfg)

    print("config     :", cfg_path)
    print("warmup     :", warmup)
    print("gains      : kp=%g ki=%g kd=%g d_delay=%s" % (
        lag_cfgs["pid_kp"], lag_cfgs["pid_ki"], lag_cfgs["pid_kd"],
        lag_cfgs["pid_d_delay"]))
    print("init       :", lag_cfgs["lagrangian_multiplier_init"])
    print("cost_limit :", lag_cfgs["cost_limit"])
    print("PIDCALL rows:", len(rows))
    print()

    # ---------------------------------------------------------------
    # Check A. One call per epoch after warmup, zero during warmup.
    # ---------------------------------------------------------------
    per_epoch = {}
    for r in rows:
        per_epoch[r["epoch"]] = per_epoch.get(r["epoch"], 0) + 1

    bad_count = {e: c for e, c in per_epoch.items() if c != 1}
    epochs_seen = sorted(per_epoch)
    early = [e for e in epochs_seen if warmup is not None and e <= warmup]

    print("CHECK A  call count per epoch")
    print("  epochs with a call :", epochs_seen[0], "to", epochs_seen[-1],
          "count", len(epochs_seen))
    print("  epochs with count != 1 :", bad_count if bad_count else "none")
    print("  calls at or before warmup epoch :", early if early else "none")
    a_ok = (not bad_count) and (not early)
    print("  ->", "PASS" if a_ok else "FAIL")
    print()

    # Contiguity. Every epoch from first to last should have exactly one call.
    missing = [e for e in range(epochs_seen[0], epochs_seen[-1] + 1)
               if e not in per_epoch]
    print("  epochs in range with no call :", missing if missing else "none")
    print()

    # ---------------------------------------------------------------
    # Check B. jc equals the independently read deque mean.
    # ---------------------------------------------------------------
    # Note on the threshold. omnisafe's get_stats builds torch.tensor(vals)
    # from a list of Python floats, which defaults to float32, so Jc is a
    # float32 mean widened back to a Python float. The independent read in
    # the wrapper sums in float64. So the two agree to float32 relative
    # precision, about 6e-8, and not to 1e-16. Anything larger than 1e-6
    # relative means the gate is reading the wrong thing, not rounding.
    worst_b_abs = 0.0
    worst_b_rel = 0.0
    for r in rows:
        a = abs(r["jc"] - r["win"])
        worst_b_abs = max(worst_b_abs, a)
        if r["win"] != 0.0:
            worst_b_rel = max(worst_b_rel, a / abs(r["win"]))
    print("CHECK B  jc against independent deque mean")
    print("  max abs mismatch : %.3e" % worst_b_abs)
    print("  max rel mismatch : %.3e   (float32 floor is about 6e-8)"
          % worst_b_rel)
    print("  deque length range : %d to %d"
          % (min(r["n"] for r in rows), max(r["n"] for r in rows)))
    b_ok = worst_b_rel < 1e-6
    print("  ->", "PASS" if b_ok else "FAIL")
    print()

    # ---------------------------------------------------------------
    # Check C. Replay the Jc sequence through a fresh PIDLagrangian.
    # ---------------------------------------------------------------
    replay = PIDLagrangian(**lag_cfgs)
    worst_c = 0.0
    worst_at = None
    for r in rows:
        # Honour an annealed budget. For fixed d runs this is a no op.
        replay._cost_limit = r["d"]
        replay.pid_update(r["jc"])
        diff = abs(replay.lagrangian_multiplier - r["lam"])
        if diff > worst_c:
            worst_c = diff
            worst_at = r["epoch"]
    print("CHECK C  multiplier replay")
    print("  max abs mismatch : %.3e  (worst at epoch %s)" % (worst_c, worst_at))
    print("  final logged lambda : %.10g" % rows[-1]["lam"])
    print("  final replay lambda : %.10g" % replay.lagrangian_multiplier)
    c_ok = worst_c < 1e-12
    print("  ->", "PASS" if c_ok else "FAIL")
    print()

    # ---------------------------------------------------------------
    # Trace, for eyeballing against the collapse boundary of 0.009.
    # ---------------------------------------------------------------
    if len(rows) > 60:
        print("trace, first call of each epoch (%d rows total)" % len(rows))
        seen = set()
        shown = []
        for r in rows:
            if r["epoch"] not in seen:
                seen.add(r["epoch"])
                shown.append(r)
    else:
        print("trace, every call")
        shown = rows
    print("epoch      jc         d        lambda")
    for r in shown:
        print("%5d  %9.4f  %8.2f  %.8f" % (r["epoch"], r["jc"], r["d"], r["lam"]))
    print()

    all_ok = a_ok and b_ok and c_ok
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
