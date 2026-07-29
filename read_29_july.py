#!/usr/bin/env python
"""
Reads for the 29 July evening session. Login node safe, CSV only, no SOFA.

Part A  ki insertion depth, closes the confound that low ep_cost may only mean
        a policy that stopped inserting.
Part B  per arch success across R4 at four weights and SACPID d30, tests whether
        ArchType II is unsolved under every condition rather than only under R4.
Part C  force safety at 0.85 N inside a matched insertion band, so the 211 mm
        against 90 mm insertion gap cannot drive the comparison.

Run from the project root:
    python read_29_july.py
    python read_29_july.py --part A
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

STEPS_PER_EPOCH = 6000
COST_THRESHOLD_N = 0.85

R4_PREFIXES = [
    "r4tip_w0.0003",
    "r4tip_w0.001",
    "r4tip_w0.003",
    "r4tip_w0.01",
    "r4tip_w0.03",          # picked up automatically once its evals land
]
SACPID_PREFIXES = [
    "sacpid_tip_w20_d30",
    "sacpid_tip_w20_d150",  # picked up automatically once its evals land
]

KI_DIRS = [
    ("ki8 s0", "runs_sacpid_tip_ki8_d20_s0", "episodes_seed0.csv"),
    ("ki8 s1", "runs_sacpid_tip_ki8_d20_s1", "episodes_seed1.csv"),
    ("ki8 s2", "runs_sacpid_tip_ki8_d20_s2", "episodes_seed2.csv"),
]


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def eval_files(prefix, pool="test"):
    """Test pool epoch 50 evals only. Excludes any _ep<N> override tag, which
    would otherwise break the epoch matching every comparison depends on."""
    found = []
    pattern = "evals/%s_s*_%s_seed100.csv" % (prefix, pool)
    for path in sorted(glob.glob(pattern)):
        tail = os.path.basename(path).replace(prefix, "", 1)
        if "_ep" in tail:
            continue
        found.append(path)
    return found


def load_condition(prefix, pool="test"):
    frames = []
    for path in eval_files(prefix, pool):
        df = pd.read_csv(path)
        df["source"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- part A

def part_a():
    rule("PART A   ki arm insertion depth against epoch")
    print("If insertion holds up while ep_cost falls, the constraint is working.")
    print("If insertion collapses alongside cost, the low cost is a stalled policy.\n")

    for label, rundir, fname in KI_DIRS:
        path = os.path.join(rundir, fname)
        if not os.path.exists(path):
            print("%-8s  missing %s" % (label, path))
            continue

        df = pd.read_csv(path)
        if "inserted_final" not in df.columns:
            print("%-8s  no inserted_final column, columns are:" % label)
            print("   ", list(df.columns))
            continue

        # Epoch from pooled cumulative environment steps, ordered by wall_time,
        # because parallel env_instance values make a per instance cumsum wrong.
        order = "wall_time" if "wall_time" in df.columns else "episode"
        df = df.sort_values(order).reset_index(drop=True)
        df["cum_steps"] = df["steps"].cumsum()
        df["epoch"] = (df["cum_steps"] // STEPS_PER_EPOCH).astype(int)

        n_inst = df["env_instance"].nunique() if "env_instance" in df.columns else 1
        print("%s   %d episodes, %d env_instance value(s), epochs 0 to %d"
              % (label, len(df), n_inst, df["epoch"].max()))

        cost_col = None
        for cand in ("shadow_tip", "hinge_cost_shadow", "tip_mean"):
            if cand in df.columns:
                cost_col = cand
                break

        rows = []
        for ep in [20, 21, 22, 24, 26, 30, 35, 39, 44, 49]:
            sub = df[df["epoch"] == ep]
            if sub.empty:
                continue
            rows.append((
                ep,
                len(sub),
                sub["inserted_final"].mean(),
                sub["success"].mean() if "success" in sub.columns else float("nan"),
                sub[cost_col].mean() if cost_col else float("nan"),
            ))

        if not rows:
            print("   no episodes in the sampled epochs\n")
            continue

        print("   %-7s %5s %12s %9s %12s" % ("epoch", "n", "insert_mm", "success", cost_col or "cost"))
        for ep, n, ins, suc, cost in rows:
            print("   %-7d %5d %12.1f %9.3f %12.2f" % (ep, n, ins, suc, cost))

        early = df[(df["epoch"] >= 20) & (df["epoch"] <= 24)]["inserted_final"].mean()
        late = df[df["epoch"] >= 40]["inserted_final"].mean()
        if np.isfinite(early) and np.isfinite(late) and early > 0:
            print("   insertion epochs 20 to 24 %.1f mm, epochs 40 plus %.1f mm, ratio %.2f"
                  % (early, late, late / early))
        print()


# ----------------------------------------------------------------- part B

def part_b():
    rule("PART B   per arch success, R4 and SACPID, test pool, epoch 50")

    conditions = []
    for prefix in R4_PREFIXES + SACPID_PREFIXES:
        df = load_condition(prefix)
        if df is None:
            print("no evals yet for %s, skipped" % prefix)
            continue
        conditions.append((prefix, df))

    if not conditions:
        print("nothing to read")
        return

    print()
    archs = sorted(set().union(*[set(df["arch_type"].unique()) for _, df in conditions]))

    header = "%-24s" % "condition" + "".join("%10s" % ("arch %s" % a) for a in archs) + "%10s" % "all"
    print(header)
    print("-" * len(header))

    for prefix, df in conditions:
        line = "%-24s" % prefix
        for a in archs:
            sub = df[df["arch_type"] == a]
            line += "%10s" % ("%.3f" % sub["success"].mean() if len(sub) else "   .")
        line += "%10.3f" % df["success"].mean()
        print(line)

    # The claim under test: is any arch zero across every condition.
    print("\nzero success check, pooled over all conditions above")
    pooled = pd.concat([df for _, df in conditions], ignore_index=True)
    for a in archs:
        sub = pooled[pooled["arch_type"] == a]
        wins = int(sub["success"].sum())
        flag = "   NEVER SOLVED" if wins == 0 else ""
        print("   arch %-4s  %4d / %4d episodes%s" % (a, wins, len(sub), flag))

    print("\nsame check inside each condition separately")
    for prefix, df in conditions:
        never = [a for a in archs
                 if len(df[df["arch_type"] == a]) and df[df["arch_type"] == a]["success"].sum() == 0]
        print("   %-24s never solved: %s" % (prefix, never if never else "none"))


# ----------------------------------------------------------------- part C

def part_c():
    rule("PART C   force safety at %.2f N inside a matched insertion band" % COST_THRESHOLD_N)
    print("steps_over_threshold counts steps above %.2f N, which is COST_THRESHOLD_N," % COST_THRESHOLD_N)
    print("not the 500 N buckling rule. A 500 N rate is not derivable from these CSVs.\n")

    groups = {}
    r4 = [load_condition(p) for p in R4_PREFIXES]
    r4 = [d for d in r4 if d is not None]
    if r4:
        groups["R4 all weights"] = pd.concat(r4, ignore_index=True)

    d30 = load_condition("sacpid_tip_w20_d30")
    if d30 is not None:
        groups["SACPID d30"] = d30

    d150 = load_condition("sacpid_tip_w20_d150")
    if d150 is not None:
        groups["SACPID d150"] = d150

    if len(groups) < 2:
        print("need at least two conditions, have %d" % len(groups))
        return

    force_cols = [c for c in ("steps_over_threshold", "tip_steps_over_threshold")
                  if all(c in df.columns for df in groups.values())]

    print("unmatched, for reference only, insertion differs so this is not a fair read")
    print("%-18s %6s %12s %10s" % ("condition", "n", "insert_mm", "success")
          + "".join("%14s" % c[:14] for c in force_cols))
    for name, df in groups.items():
        line = "%-18s %6d %12.1f %10.3f" % (
            name, len(df), df["inserted_final"].mean(), df["success"].mean())
        for c in force_cols:
            line += "%14.2f" % df[c].mean()
        print(line)

    # Matched band: intersect the central 80 percent of each condition's
    # insertion distribution, so neither tail sets the window.
    lo = max(np.percentile(df["inserted_final"], 10) for df in groups.values())
    hi = min(np.percentile(df["inserted_final"], 90) for df in groups.values())
    print("\nmatched band from the p10 to p90 overlap: %.1f mm to %.1f mm" % (lo, hi))

    if hi <= lo:
        print("NO OVERLAP. The insertion distributions do not share a common band,")
        print("so a matched comparison is not possible and must not be reported.")
    else:
        print("%-18s %6s %12s %10s" % ("condition", "n", "insert_mm", "success")
              + "".join("%14s" % c[:14] for c in force_cols))
        for name, df in groups.items():
            sub = df[(df["inserted_final"] >= lo) & (df["inserted_final"] <= hi)]
            if len(sub) < 10:
                print("%-18s %6d   too few episodes in band, do not report" % (name, len(sub)))
                continue
            line = "%-18s %6d %12.1f %10.3f" % (
                name, len(sub), sub["inserted_final"].mean(), sub["success"].mean())
            for c in force_cols:
                line += "%14.2f" % sub[c].mean()
            print(line)

    # Binned, so the reader can see the result is not an artefact of band choice.
    print("\nbinned by insertion, 50 mm bins, mean %s per episode" % (force_cols[0] if force_cols else "force"))
    if force_cols:
        edges = np.arange(0, 351, 50)
        header = "%-18s" % "condition" + "".join(
            "%12s" % ("%d-%d" % (edges[i], edges[i + 1])) for i in range(len(edges) - 1))
        print(header)
        for name, df in groups.items():
            line = "%-18s" % name
            for i in range(len(edges) - 1):
                sub = df[(df["inserted_final"] >= edges[i]) & (df["inserted_final"] < edges[i + 1])]
                line += "%12s" % ("%.2f" % sub[force_cols[0]].mean() if len(sub) >= 5 else ".")
            print(line)
        print("dot means fewer than 5 episodes in that bin")


# ----------------------------------------------------------------- part D

def part_d():
    rule("PART D   d150 seed 2 provenance, does it join the control")
    print("Seed 2 ran pre push code with no LAG logging. It is usable only if its")
    print("config matches seed 3 apart from the seed, and its multiplier is zero.\n")

    cfgs = {}
    for s in (2, 3, 4):
        hits = glob.glob("runs_sacpid_tip_w20_d150_s%d/**/config.json" % s, recursive=True)
        if not hits:
            print("seed %d: no config.json found" % s)
            continue
        with open(hits[0]) as fh:
            cfgs[s] = json.load(fh)
        print("seed %d config: %s" % (s, hits[0]))

    if 2 in cfgs and 3 in cfgs:
        flat2 = json.loads(json.dumps(cfgs[2]))
        flat3 = json.loads(json.dumps(cfgs[3]))

        def flatten(d, pre=""):
            out = {}
            for k, v in d.items():
                key = pre + k
                if isinstance(v, dict):
                    out.update(flatten(v, key + "."))
                else:
                    out[key] = v
            return out

        f2, f3 = flatten(flat2), flatten(flat3)
        diffs = [(k, f2.get(k), f3.get(k)) for k in sorted(set(f2) | set(f3))
                 if f2.get(k) != f3.get(k)]
        print("\nconfig differences, seed 2 against seed 3")
        if not diffs:
            print("   none at all, which is suspicious, the seed itself should differ")
        for k, a, b in diffs:
            print("   %-40s s2=%-18s s3=%s" % (k, a, b))
        benign = all(("seed" in k.lower() or "time" in k.lower() or "dir" in k.lower()
                      or "name" in k.lower()) for k, _, _ in diffs)
        print("\n   VERDICT: %s" % ("all differences are seed or path only, seed 2 joins the control"
                                    if benign and diffs else
                                    "review the differences above before using seed 2"))

    # Multiplier trace from the CSV, since seed 2 has no stdout LAG lines.
    print("\nLagrange multiplier from progress.csv")
    for s in (2, 3, 4):
        hits = glob.glob("runs_sacpid_tip_w20_d150_s%d/**/progress.csv" % s, recursive=True)
        if not hits:
            print("   seed %d: no progress.csv" % s)
            continue
        df = pd.read_csv(hits[0])
        cols = [c for c in df.columns if "lagrang" in c.lower() or "multiplier" in c.lower()
                or c.lower().endswith("lambda")]
        if not cols:
            print("   seed %d: no multiplier column. columns containing 'Metrics' or 'Lag':" % s)
            print("     ", [c for c in df.columns if "Lag" in c or "Metrics" in c][:12])
            continue
        c = cols[0]
        print("   seed %d  %s  min %.3e  max %.3e  final %.3e"
              % (s, c, df[c].min(), df[c].max(), df[c].iloc[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ABCD", help="subset of ABCD, default all")
    args = ap.parse_args()
    if "A" in args.part:
        part_a()
    if "B" in args.part:
        part_b()
    if "C" in args.part:
        part_c()
    if "D" in args.part:
        part_d()
    print()


if __name__ == "__main__":
    main()
