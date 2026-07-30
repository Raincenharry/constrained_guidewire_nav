import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
REF = "r1_baseline"
NO_PROGRESS_MM = 5.0
BAND_LO = 125.4
NBOOT = 5000
RNG = np.random.default_rng(0)
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]

def load(pool):
    frames = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(path))
        if m is None or m.group("pool") != pool:
            continue
        d = pd.read_csv(path)
        d["cond"] = m.group("cond")
        d["seed"] = int(m.group("seed"))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def anatomy_table(df, cond, cost_col):
    d = df[df["cond"] == cond]
    if len(d) == 0 or d[cost_col].isna().all():
        return None
    g = d.groupby(KEY).agg(
        ins=("inserted_final", "mean"),
        cost=(cost_col, "mean"),
        succ=("success", "mean"),
        nseed=("seed", "nunique"),
    ).reset_index()
    return g

def boot_ratio(cost, ins):
    n = len(cost)
    point = cost.sum() / ins.sum()
    idx = RNG.integers(0, n, size=(NBOOT, n))
    samp = cost.values[idx].sum(axis=1) / ins.values[idx].sum(axis=1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return point, lo, hi

def boot_mean(x):
    n = len(x)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    samp = np.asarray(x)[idx].mean(axis=1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return float(np.mean(x)), lo, hi

def run(pool, cost_col):
    df = load(pool)
    conds = sorted(df["cond"].unique())
    tabs = {c: anatomy_table(df, c, cost_col) for c in conds}
    tabs = {c: t for c, t in tabs.items() if t is not None}
    print("=== pool %s, cost column %s ===" % (pool, cost_col))
    print("\n--- frontier, paired over anatomies ---")
    print("%-24s %5s %8s %8s %22s %8s" % ("cond", "nanat", "ins", "succ", "cost per mm 95pct", "noprog"))
    for c, t in tabs.items():
        p, lo, hi = boot_ratio(t["cost"], t["ins"])
        noprog = float((t["ins"] < NO_PROGRESS_MM).mean())
        print("%-24s %5d %8.1f %8.3f  %6.4f [%6.4f %6.4f] %8.3f"
              % (c, len(t), t["ins"].mean(), t["succ"].mean(), p, lo, hi, noprog))
    if REF not in tabs:
        print("reference %s absent, skipping paired differences" % REF)
        return
    ref = tabs[REF]
    print("\n--- paired against %s, all anatomies ---" % REF)
    print("%-24s %5s %24s %24s %8s" % ("cond", "npair", "d ins mm 95pct", "d cost 95pct", "frac<"))
    for c, t in tabs.items():
        if c == REF:
            continue
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        di = j["ins"] - j["ins_ref"]
        dc = j["cost"] - j["cost_ref"]
        mi, ilo, ihi = boot_mean(di)
        mc, clo, chi = boot_mean(dc)
        frac = float((dc < 0).mean())
        print("%-24s %5d %8.2f [%7.2f %7.2f] %8.2f [%7.2f %7.2f] %8.3f"
              % (c, len(j), mi, ilo, ihi, mc, clo, chi, frac))
    print("\n--- paired, both above %.1f mm on the same anatomy ---" % BAND_LO)
    print("%-24s %5s %24s %24s %22s" % ("cond", "npair", "d ins mm 95pct", "d cost 95pct", "cost per mm 95pct"))
    for c, t in tabs.items():
        if c == REF:
            continue
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        j = j[(j["ins"] >= BAND_LO) & (j["ins_ref"] >= BAND_LO)]
        if len(j) < 8:
            print("%-24s %5d  too few paired anatomies" % (c, len(j)))
            continue
        mi, ilo, ihi = boot_mean(j["ins"] - j["ins_ref"])
        mc, clo, chi = boot_mean(j["cost"] - j["cost_ref"])
        p, lo, hi = boot_ratio(j["cost"], j["ins"])
        print("%-24s %5d %8.2f [%7.2f %7.2f] %8.2f [%7.2f %7.2f]  %6.4f [%6.4f %6.4f]"
              % (c, len(j), mi, ilo, ihi, mc, clo, chi, p, lo, hi))
    print()

if __name__ == "__main__":
    for pool in ["test", "train"]:
        run(pool, "tip_steps_over_threshold")
        run(pool, "steps_over_threshold")