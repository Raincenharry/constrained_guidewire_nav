import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1", "r4_w0.0003"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
PLEN = "path_length_at_reset"
COST = "shadow_tip"
REF = "r1_baseline"
NBOOT = 5000
RNG = np.random.default_rng(0)

def load(pool):
    frames = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(path))
        if m is None or m.group("pool") != pool or m.group("cond") in DROP:
            continue
        d = pd.read_csv(path)
        d["cond"] = m.group("cond"); d["seed"] = int(m.group("seed"))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df[COST].notna()].copy()
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    df["prog"] = df["frac"].clip(upper=1.0)
    df["prog_z"] = np.where(df["jam"] == 1, 0.0, df["prog"])
    return df

def boot_mean(x):
    x = np.asarray(x, float); n = len(x)
    if n < 5:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    s = x[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(x.mean()), lo, hi

def boot_ratio(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float); n = len(num)
    if n < 5 or abs(den.mean()) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    d = den[idx].mean(axis=1); ok = np.abs(d) > 1e-9
    s = num[idx].mean(axis=1)[ok] / d[ok]
    if len(s) < 100:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(num.mean() / den.mean()), lo, hi

def s1(df, pool):
    print("=== 1. what a jam looks like, pool %s ===" % pool)
    j = df[df["jam"] == 1]; n = df[df["jam"] == 0]
    print("  %-14s %7s %9s %9s %9s %9s %9s" %
          ("group", "n", "frac_med", "cost", "f_max", "clamped", "steps"))
    for lab, s in [("jam", j), ("not jam", n)]:
        print("  %-14s %7d %9.2f %9.2f %9.1f %9.3f %9.0f" % (
            lab, len(s), s["frac"].median(), s[COST].mean(),
            s["force_max"].mean(), s["clamped"].mean(), s["steps"].mean()))
    print("\n  jam rate by path length decile, is it a property of the anatomy:")
    g = df.groupby(KEY).agg(p=(PLEN, "mean"), jam=("jam", "mean")).reset_index()
    g["dec"] = pd.qcut(g["p"], 10, labels=False, duplicates="drop")
    print(g.groupby("dec").agg(n=("jam", "size"), p_med=("p", "median"),
                               jam=("jam", "mean")).round(3).to_string())
    print("\n  jam rate by arch_type:")
    print(df.groupby("arch_type")["jam"].agg(["size", "mean"]).round(3).to_string())
    print()

def s2(df, pool):
    print("=== 2. jam rate as the safety endpoint, paired, pool %s ===" % pool)
    ref = df[df["cond"] == REF].groupby(KEY).agg(
        jam=("jam", "mean"), p=("prog_z", "mean")).reset_index()
    print("  %-24s %7s %9s %24s %26s" %
          ("cond", "n", "jam_rate", "d jam 95pct", "jam removed per prog"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        if c == REF:
            print("  %-24s %7d %9.3f %24s %26s" %
                  (c, len(d), d["jam"].mean(), "reference", ""))
            continue
        t = d.groupby(KEY).agg(jam=("jam", "mean"), p=("prog_z", "mean")).reset_index()
        k = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        m, lo, hi = boot_mean(k["jam"] - k["jam_ref"])
        e, elo, ehi = boot_ratio((k["jam_ref"] - k["jam"]).values,
                                 (k["p_ref"] - k["p"]).values)
        print("  %-24s %7d %9.3f %6.3f [%6.3f %6.3f] %8.3f [%7.3f %7.3f]" %
              (c, len(d), d["jam"].mean(), m, lo, hi, e, elo, ehi))
    print()

def s3(df, pool):
    print("=== 3. is integrated cost just a jam proxy, pool %s ===" % pool)
    rows = []
    for (c, s), d in df.groupby(["cond", "seed"]):
        rows.append(dict(cond=c, seed=int(s), jam=d["jam"].mean(),
                         cost=d[COST].mean(),
                         cost_nj=d[d["jam"] == 0][COST].mean(),
                         prog=d["prog_z"].mean()))
    t = pd.DataFrame(rows)
    print("  across %d seeds:" % len(t))
    print("    Spearman jam against cost         %.3f" %
          t["jam"].corr(t["cost"], method="spearman"))
    print("    Spearman jam against cost_no_jam  %.3f" %
          t["jam"].corr(t["cost_nj"], method="spearman"))
    print("    Spearman cost against prog        %.3f" %
          t["cost"].corr(t["prog"], method="spearman"))
    print("    Spearman jam against prog         %.3f" %
          t["jam"].corr(t["prog"], method="spearman"))
    print("\n  if jam and cost correlate near 1.0 and cost_no_jam does not track jam,")
    print("  then the cost metric is largely counting jams and jam rate is the")
    print("  cleaner endpoint to report.\n")
    print(t.round(3).to_string(index=False))
    print()

def s4(df, pool):
    print("=== 4. at matched progress, does jam still separate, pool %s ===" % pool)
    for lo, hi in [(0.60, 0.80), (0.80, 0.95)]:
        d = df[(df["prog"] >= lo) & (df["prog"] < hi)]
        print("  prog in [%.2f, %.2f), %d episodes" % (lo, hi, len(d)))
        print("  %-24s %7s %9s %9s %9s" % ("cond", "n", "jam", "cost", "f_max"))
        for c in sorted(d["cond"].unique()):
            s = d[d["cond"] == c]
            if len(s) < 15:
                continue
            print("  %-24s %7d %9.3f %9.2f %9.1f" %
                  (c, len(s), s["jam"].mean(), s[COST].mean(), s["force_max"].mean()))
        print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        s1(df, pool); s2(df, pool); s3(df, pool); s4(df, pool)