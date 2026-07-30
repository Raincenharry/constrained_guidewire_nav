import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
CLAMP_MM = 382.5
REF = "r1_baseline"

def load(pool):
    frames = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(path))
        if m is None or m.group("pool") != pool or m.group("cond") in DROP:
            continue
        d = pd.read_csv(path)
        d["cond"] = m.group("cond")
        d["seed"] = int(m.group("seed"))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def section_a(df, pool):
    print("=== A. what kind of quantity is path_length_at_reset, pool %s ===" % pool)
    col = "path_length_at_reset"
    print("  overall: min %.2f  median %.2f  max %.2f  distinct values %d"
          % (df[col].min(), df[col].median(), df[col].max(), df[col].nunique()))
    d = df[df["cond"] == REF]
    g = d.groupby(KEY)[col].agg(["mean", "std", "size"])
    print("  within anatomy std across r1_baseline seeds: median %.4f, max %.4f"
          % (g["std"].median(), g["std"].max()))
    print("  if that is near zero it is a property of the anatomy, so a path to target.")
    print("  if it is large it is a property of the episode, so wire fed, and section")
    print("  C and D below do not mean what they are labelled as meaning.")
    h = df.groupby(["cond"])[col].mean()
    print("\n  mean by condition, a second check on the same question:")
    print(h.round(2).to_string())
    print("  if conditions differ a lot it tracks the policy, not the anatomy.")
    print()

def section_b(df, pool):
    print("=== B. path length by arch_type, one row per anatomy, pool %s ===" % pool)
    col = "path_length_at_reset"
    g = df.groupby(KEY)[col].mean().reset_index()
    print(g.groupby("arch_type")[col].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]).round(1).to_string())
    g["over"] = (g[col] > CLAMP_MM).astype(float)
    print("\n  fraction of anatomies whose path exceeds the %.1f mm clamp:" % CLAMP_MM)
    print(g.groupby("arch_type")["over"].mean().round(3).to_string())
    print()

def section_c(df, pool):
    print("=== C. fraction of the path the device covers, r1_baseline, pool %s ===" % pool)
    d = df[(df["cond"] == REF) & (df["path_length_at_reset"] > 0)].copy()
    d["frac"] = d["inserted_final"] / d["path_length_at_reset"]
    print(d.groupby("arch_type")["frac"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]).round(3).to_string())
    best = df.groupby(KEY)["success"].max().reset_index().rename(
        columns={"success": "best"})
    d = d.merge(best, on=KEY)
    print("\n  split by whether the anatomy is ever solved by any condition:")
    print(d.groupby(["arch_type", "best"])["frac"].agg(
        ["size", "mean", "median", "max"]).round(3).to_string())
    print("\n  split by outcome of the individual episode:")
    print(d.groupby(["arch_type", "success"])["frac"].agg(
        ["size", "mean", "median"]).round(3).to_string())
    print()

def section_d(df, pool):
    print("=== D. does path length predict solvability, pool %s ===" % pool)
    col = "path_length_at_reset"
    g = df.groupby(KEY).agg(best=("success", "max"),
                            plen=(col, "mean"),
                            ins=("inserted_final", "mean")).reset_index()
    for label, sub in [("solved by something", g[g["best"] > 0]),
                       ("never solved", g[g["best"] == 0])]:
        if len(sub) == 0:
            continue
        print("  %-22s n=%3d  path mean %7.1f  median %7.1f  over clamp %.3f  ins median %6.1f"
              % (label, len(sub), sub["plen"].mean(), sub["plen"].median(),
                 float((sub["plen"] > CLAMP_MM).mean()), sub["ins"].median()))
    print("\n  by arch_type:")
    for at, sub in g.groupby("arch_type"):
        print("    %-14s %3d of %3d solved, path median %7.1f, over clamp %.3f"
              % (str(at), int((sub["best"] > 0).sum()), len(sub),
                 sub["plen"].median(), float((sub["plen"] > CLAMP_MM).mean())))
    print("\n  Spearman correlation, anatomy path length against solved by anything:")
    print("    %.3f" % g["plen"].corr(g["best"], method="spearman"))
    print()

def section_e(df, pool):
    print("=== E. success restricted to the solvable subset, pool %s ===" % pool)
    best = df.groupby(KEY)["success"].max().reset_index().rename(
        columns={"success": "best"})
    d = df.merge(best, on=KEY)
    s = d[d["best"] > 0]
    print("  %d solvable anatomies, %d episodes" % (s.groupby(KEY).ngroups, len(s)))
    print("%-24s %10s %10s %10s %10s" %
          ("cond", "succ_solv", "succ_all", "ins_solv", "tip_cost"))
    for c in sorted(df["cond"].unique()):
        a = s[s["cond"] == c]
        b = df[df["cond"] == c]
        tc = a["tip_steps_over_threshold"].mean()
        print("%-24s %10.3f %10.3f %10.1f %10.2f" %
              (c, a["success"].mean(), b["success"].mean(),
               a["inserted_final"].mean(), tc))
    print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        section_a(df, pool)
        section_b(df, pool)
        section_c(df, pool)
        section_d(df, pool)
        section_e(df, pool)