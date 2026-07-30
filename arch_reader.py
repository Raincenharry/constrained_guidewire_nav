import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
NO_PROGRESS_MM = 5.0
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")

def load(pool):
    frames = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(path))
        if m is None or m.group("pool") != pool:
            continue
        if m.group("cond") in DROP:
            continue
        d = pd.read_csv(path)
        d["cond"] = m.group("cond")
        d["seed"] = int(m.group("seed"))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def arch_census(df, pool):
    print("--- anatomy census, pool %s ---" % pool)
    a = df.groupby("arch_type")["arch_seed"].nunique()
    n = df.groupby("arch_type").size()
    for k in a.index:
        print("  arch_type %-6s  %3d distinct anatomies, %5d episodes"
              % (str(k), a[k], n[k]))
    print()

def by_arch(df, pool):
    print("--- success and insertion by arch_type, pool %s ---" % pool)
    g = df.groupby(["cond", "arch_type"]).agg(
        n=("success", "size"),
        ins=("inserted_final", "mean"),
        ins_med=("inserted_final", "median"),
        succ=("success", "mean"),
        noprog=("inserted_final", lambda s: float((s < NO_PROGRESS_MM).mean())),
        tipcost=("tip_steps_over_threshold", "mean"),
    ).reset_index()
    print(g.pivot(index="cond", columns="arch_type", values="succ").round(3))
    print()
    print("insertion mean")
    print(g.pivot(index="cond", columns="arch_type", values="ins").round(1))
    print()
    print("no progress fraction below %.0f mm" % NO_PROGRESS_MM)
    print(g.pivot(index="cond", columns="arch_type", values="noprog").round(3))
    print()

def r1_per_seed(df, pool):
    print("--- r1_baseline by arch_type and seed, pool %s ---" % pool)
    d = df[df["cond"] == "r1_baseline"]
    if len(d) == 0:
        print("  r1_baseline absent")
        return
    g = d.groupby(["arch_type", "seed"]).agg(
        n=("success", "size"),
        ins=("inserted_final", "mean"),
        succ=("success", "mean"),
    ).reset_index()
    print(g.to_string(index=False))
    print()

def anatomy_universality(df, pool):
    print("--- anatomies never solved by ANY condition, pool %s ---" % pool)
    g = df.groupby(["arch_type", "arch_seed"]).agg(
        conds=("cond", "nunique"),
        succ=("success", "mean"),
        best=("success", "max"),
        ins=("inserted_final", "mean"),
    ).reset_index()
    dead = g[g["best"] == 0]
    tot = g.groupby("arch_type").size()
    ded = dead.groupby("arch_type").size()
    for k in tot.index:
        d = int(ded.get(k, 0))
        print("  arch_type %-6s  %3d of %3d anatomies never solved by any condition, "
              "their mean insertion %.1f mm"
              % (str(k), d, tot[k],
                 dead[dead["arch_type"] == k]["ins"].mean() if d else float("nan")))
    print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("==================== pool %s ====================" % pool)
        df = load(pool)
        arch_census(df, pool)
        by_arch(df, pool)
        r1_per_seed(df, pool)
        anatomy_universality(df, pool)