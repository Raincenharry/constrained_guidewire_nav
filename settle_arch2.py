import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
BAND_LO = 125.4
BUCKLE_N = 500.0
EXCLUDE_ARCH = "ArchType.II"
STOCH = ["cmp_r1_stoch_train_seed100.csv", "cmp_sacpid_stoch_train_seed100.csv"]
NBOOT = 5000
RNG = np.random.default_rng(0)

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
    df = pd.concat(frames, ignore_index=True)
    df["buckled"] = (df["force_max"] >= BUCKLE_N).astype(float)
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    return df

def boot_mean(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    s = x[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(x.mean()), lo, hi

def section_a():
    print("=== A. stochastic action selection, does ArchType II ever succeed ===")
    for fn in STOCH:
        p = os.path.join(EVAL_DIR, fn)
        if not os.path.exists(p):
            print("  MISSING %s" % fn)
            continue
        d = pd.read_csv(p)
        print("\n  file %s, %d episodes" % (fn, len(d)))
        if "arch_type" not in d.columns:
            print("    no arch_type column, cannot answer")
            continue
        g = d.groupby("arch_type").agg(
            n=("success", "size"),
            anat=("arch_seed", "nunique"),
            ins=("inserted_final", "mean"),
            succ=("success", "mean"),
        )
        print(g.round(3).to_string())
        ii = d[d["arch_type"] == EXCLUDE_ARCH]
        if len(ii):
            print("    %s: %d episodes, %d successes" %
                  (EXCLUDE_ARCH, len(ii), int(ii["success"].sum())))
    print()

def section_b(df, pool):
    print("=== B. pool %s, headline with and without %s ===" % (pool, EXCLUDE_ARCH))
    if EXCLUDE_ARCH not in set(df["arch_type"]):
        print("  %s not present in this pool, nothing to exclude\n" % EXCLUDE_ARCH)
        return
    print("%-24s %8s %8s %10s %8s" %
          ("cond", "succ_all", "succ_ex", "ins_all", "ins_ex"))
    ex = df[df["arch_type"] != EXCLUDE_ARCH]
    for c in sorted(df["cond"].unique()):
        a = df[df["cond"] == c]
        b = ex[ex["cond"] == c]
        print("%-24s %8.3f %8.3f %10.1f %8.1f" %
              (c, a["success"].mean(), b["success"].mean(),
               a["inserted_final"].mean(), b["inserted_final"].mean()))
    print()

def section_c(df, pool):
    print("=== C. pool %s, is solvability a property of the anatomy ===" % pool)
    g = df.groupby(KEY).agg(
        nc=("cond", "nunique"),
        best=("success", "max"),
        mean_succ=("success", "mean"),
    ).reset_index()
    solv = g[g["best"] > 0]
    print("  %d of %d anatomies solved by at least one condition" % (len(solv), len(g)))
    piv = df.pivot_table(index=KEY, columns="cond", values="success", aggfunc="mean")
    piv = piv.dropna(axis=1, how="all")
    print("\n  anatomy level success correlation between conditions (Spearman):")
    corr = piv.corr(method="spearman")
    print(corr.round(2).to_string())
    print()

def section_d(df, pool):
    print("=== D. pool %s, where do never solved anatomies stop ===" % pool)
    g = df.groupby(KEY)["success"].max().reset_index().rename(columns={"success": "best"})
    d = df.merge(g, on=KEY)
    ref = d[d["cond"] == "r1_baseline"]
    if len(ref) == 0:
        print("  r1_baseline absent\n")
        return
    for label, sub in [("never solved", ref[ref["best"] == 0]),
                       ("solved by something", ref[ref["best"] > 0])]:
        if len(sub) == 0:
            continue
        q = sub["inserted_final"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print("  %-20s n=%4d  p10 %6.1f  p25 %6.1f  med %6.1f  p75 %6.1f  p90 %6.1f"
              % (label, len(sub), q[0.1], q[0.25], q[0.5], q[0.75], q[0.9]))
    print("\n  r1_baseline insertion histogram on never solved anatomies:")
    sub = ref[ref["best"] == 0]["inserted_final"]
    bins = list(range(0, 401, 25))
    h = pd.cut(sub, bins=bins).value_counts().sort_index()
    for k, v in h.items():
        print("    %-16s %4d %s" % (str(k), v, "#" * int(v / 2)))
    print()

def section_e(df, pool):
    print("=== E. pool %s, buckling paired at matched insertion, unclamped only ===" % pool)
    un = df[df["clamped"] == 0]
    tabs = {}
    for c, d in un.groupby("cond"):
        tabs[c] = d.groupby(KEY).agg(b=("buckled", "mean"),
                                     ins=("inserted_final", "mean")).reset_index()
    if "r1_baseline" not in tabs:
        print("  reference absent\n")
        return
    ref = tabs["r1_baseline"]
    print("%-24s %6s %26s %24s" %
          ("cond", "npair", "d buckle rate 95pct", "d ins mm"))
    for c, t in tabs.items():
        if c == "r1_baseline":
            continue
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        j = j[(j["ins"] >= BAND_LO) & (j["ins_ref"] >= BAND_LO)]
        if len(j) < 8:
            print("%-24s %6d  too few paired anatomies" % (c, len(j)))
            continue
        mb, blo, bhi = boot_mean(j["b"] - j["b_ref"])
        mi, ilo, ihi = boot_mean(j["ins"] - j["ins_ref"])
        print("%-24s %6d  %6.3f [%6.3f %6.3f] %8.1f [%6.1f %6.1f]" %
              (c, len(j), mb, blo, bhi, mi, ilo, ihi))
    print()

if __name__ == "__main__":
    section_a()
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        section_b(df, pool)
        section_c(df, pool)
        section_d(df, pool)
        section_e(df, pool)