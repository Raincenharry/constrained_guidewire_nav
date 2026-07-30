import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
REF = "r1_baseline"
DROP = ["r1"]              # r1 is bit identical to r1_baseline, keeping both double counts
BUCKLE_N = 500.0           # device level force threshold
CLAMP_MM = 382.5           # insertion clamp
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
        if m.group("cond") in DROP:
            continue
        d = pd.read_csv(path)
        d["cond"] = m.group("cond")
        d["seed"] = int(m.group("seed"))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["buckled"] = (df["force_max"] >= BUCKLE_N).astype(float)
    df["clamped_flag"] = (df["clamp_steps"] > 0).astype(float)
    df["clamped_mm"] = (df["inserted_final"] >= CLAMP_MM).astype(float)
    return df

def boot_mean(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    samp = x[idx].mean(axis=1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return float(x.mean()), lo, hi

def definitions_agree(df):
    print("--- do the two clamp definitions agree ---")
    ct = pd.crosstab(df["clamped_flag"], df["clamped_mm"])
    print(ct)
    dis = int((df["clamped_flag"] != df["clamped_mm"]).sum())
    print("episodes where they disagree: %d of %d" % (dis, len(df)))
    if dis:
        d = df[df["clamped_flag"] != df["clamped_mm"]]
        print("their insertion range: %.1f to %.1f mm"
              % (d["inserted_final"].min(), d["inserted_final"].max()))
    print()

def confound(df):
    print("--- the confound, reproducing the earlier reading ---")
    print("%-24s %6s %8s %10s %12s %12s" %
          ("cond", "n", "b_all", "frac_clamp", "b_if_clamped", "b_if_not"))
    for c, d in df.groupby("cond"):
        cl = d[d["clamped_flag"] == 1]
        un = d[d["clamped_flag"] == 0]
        print("%-24s %6d %8.3f %10.3f %12s %12s" % (
            c, len(d), d["buckled"].mean(), d["clamped_flag"].mean(),
            ("%.3f" % cl["buckled"].mean()) if len(cl) else "n/a",
            ("%.3f" % un["buckled"].mean()) if len(un) else "n/a"))
    print()

def unclamped_rate(df):
    print("--- buckling on UNCLAMPED episodes only, 95pct interval ---")
    print("%-24s %6s %8s %26s %10s" %
          ("cond", "n_unc", "ins", "buckle rate 95pct", "fmax_p95"))
    un = df[df["clamped_flag"] == 0]
    for c, d in un.groupby("cond"):
        m, lo, hi = boot_mean(d["buckled"])
        print("%-24s %6d %8.1f  %6.3f [%6.3f %6.3f] %10.1f" % (
            c, len(d), d["inserted_final"].mean(), m, lo, hi,
            d["force_max"].quantile(0.95)))
    print()

def unclamped_paired(df):
    print("--- unclamped, paired by anatomy against %s ---" % REF)
    un = df[df["clamped_flag"] == 0]
    tabs = {}
    for c, d in un.groupby("cond"):
        g = d.groupby(KEY).agg(b=("buckled", "mean"),
                               ins=("inserted_final", "mean")).reset_index()
        tabs[c] = g
    if REF not in tabs:
        print("reference absent, skipping")
        return
    ref = tabs[REF]
    print("%-24s %6s %26s %22s" %
          ("cond", "npair", "d buckle rate 95pct", "d ins mm"))
    for c, t in tabs.items():
        if c == REF:
            continue
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        if len(j) < 8:
            print("%-24s %6d  too few paired anatomies" % (c, len(j)))
            continue
        mb, blo, bhi = boot_mean(j["b"] - j["b_ref"])
        mi, ilo, ihi = boot_mean(j["ins"] - j["ins_ref"])
        print("%-24s %6d  %6.3f [%6.3f %6.3f] %8.1f [%6.1f %6.1f]" %
              (c, len(j), mb, blo, bhi, mi, ilo, ihi))
    print()

def threshold_sweep(df):
    print("--- unclamped buckle rate against threshold, is 500 N a cliff or a knob ---")
    un = df[df["clamped_flag"] == 0]
    ths = [100, 200, 300, 400, 500, 750, 1000]
    header = "%-24s" % "cond" + "".join("%9d" % t for t in ths)
    print(header)
    for c, d in un.groupby("cond"):
        row = "%-24s" % c
        for t in ths:
            row += "%9.3f" % (d["force_max"] >= t).mean()
        print(row)
    print()

if __name__ == "__main__":
    for pool in ["test", "train"]:
        print("==================== pool %s ====================" % pool)
        df = load(pool)
        definitions_agree(df)
        confound(df)
        unclamped_rate(df)
        unclamped_paired(df)
        threshold_sweep(df)