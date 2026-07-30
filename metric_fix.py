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
UNCON = ["r1_baseline", "sacpid_tip_w20_d150"]
NBOOT = 5000
RNG = np.random.default_rng(0)
VARIANTS = ["prog_raw", "prog_zero", "prog_nojam"]

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
    df = df[df[COST].notna()].copy()
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    df["prog_raw"] = df["frac"].clip(upper=1.0)
    df["prog_zero"] = np.where(df["jam"] == 1, 0.0, df["prog_raw"])
    df["prog_nojam"] = np.where(df["jam"] == 1, np.nan, df["prog_raw"])
    df["clean"] = ((df["prog_raw"] >= 0.95) & (df["jam"] == 0)).astype(float)
    return df

def boot_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    s = x[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(x.mean()), lo, hi

def boot_ratio(num, den):
    num = np.asarray(num, dtype=float); den = np.asarray(den, dtype=float)
    ok = ~(np.isnan(num) | np.isnan(den))
    num, den = num[ok], den[ok]
    n = len(num)
    if n < 5 or abs(den.mean()) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    d = den[idx].mean(axis=1)
    good = np.abs(d) > 1e-9
    s = num[idx].mean(axis=1)[good] / d[good]
    if len(s) < 100:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(num.mean() / den.mean()), lo, hi

def s1(df, pool):
    print("=== 1. how much does the jam inflate progress, pool %s ===" % pool)
    print("  prog_raw scores a jam as 1.000. prog_zero scores it 0. prog_nojam drops it.\n")
    print("  %-24s %7s %8s %10s %10s %11s %9s %9s" %
          ("cond", "n", "jam", "prog_raw", "prog_zero", "prog_nojam", "clean", "succ"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        print("  %-24s %7d %8.3f %10.3f %10.3f %11.3f %9.3f %9.3f" % (
            c, len(d), d["jam"].mean(), d["prog_raw"].mean(),
            d["prog_zero"].mean(), d["prog_nojam"].mean(),
            d["clean"].mean(), d["success"].mean()))
    print("\n  inflation = prog_raw minus prog_zero, which is exactly the jam rate")
    print("  weighted by 1.0, so it hits the high jam conditions hardest.\n")

def s2(df, pool):
    print("=== 2. paired against %s under each variant, pool %s ===" % (REF, pool))
    for v in VARIANTS:
        print("\n  --- %s ---" % v)
        ref = df[df["cond"] == REF].groupby(KEY).agg(
            p=(v, "mean"), cost=(COST, "mean")).reset_index()
        print("  %-24s %6s %22s %24s" % ("cond", "npair", "d prog 95pct", "d cost 95pct"))
        for c in sorted(df["cond"].unique()):
            if c == REF:
                continue
            t = df[df["cond"] == c].groupby(KEY).agg(
                p=(v, "mean"), cost=(COST, "mean")).reset_index()
            j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
            j = j[j["p"].notna() & j["p_ref"].notna()]
            if len(j) < 8:
                print("  %-24s %6d  too few" % (c, len(j)))
                continue
            mp, plo, phi = boot_mean(j["p"] - j["p_ref"])
            mc, clo, chi = boot_mean(j["cost"] - j["cost_ref"])
            print("  %-24s %6d %6.3f [%6.3f %6.3f] %7.2f [%7.2f %7.2f]" %
                  (c, len(j), mp, plo, phi, mc, clo, chi))
    print()

def s3(df, pool):
    print("=== 3. efficiency under each variant, pool %s ===" % pool)
    print("  cost removed per unit of progress given up. higher is better.\n")
    for v in VARIANTS:
        print("  --- %s ---" % v)
        ref = df[df["cond"] == REF].groupby(KEY).agg(
            p=(v, "mean"), cost=(COST, "mean")).reset_index()
        print("  %-24s %6s %28s %10s" % ("cond", "npair", "eff 95pct", "d prog"))
        for c in sorted(df["cond"].unique()):
            if c == REF:
                continue
            t = df[df["cond"] == c].groupby(KEY).agg(
                p=(v, "mean"), cost=(COST, "mean")).reset_index()
            j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
            j = j[j["p"].notna() & j["p_ref"].notna()]
            if len(j) < 8:
                print("  %-24s %6d  too few" % (c, len(j)))
                continue
            m, lo, hi = boot_ratio((j["cost_ref"] - j["cost"]).values,
                                   (j["p_ref"] - j["p"]).values)
            print("  %-24s %6d %10.1f [%8.1f %8.1f] %10.3f" %
                  (c, len(j), m, lo, hi, float((j["p_ref"] - j["p"]).mean())))
        print()

def s4(df, pool):
    print("=== 4. head to head under each variant, fixed solvable set, pool %s ===" % pool)
    u = df[df["cond"].isin([c for c in UNCON if c in set(df["cond"])])]
    fx = u.groupby(KEY)["success"].max().reset_index()
    fx = fx[fx["success"] > 0][KEY]
    s = df.merge(fx, on=KEY, how="inner")
    pairs = [("r4tip_w0.01", "sacpid_tip_w20_d30"),
             ("r4tip_w0.003", "sacpid_tip_w20_d20"),
             ("r4tip_w0.001", "sacpid_tip_w20_d20")]
    for a, b in pairs:
        if a not in set(s["cond"]) or b not in set(s["cond"]):
            continue
        print("\n  %s minus %s" % (a, b))
        for v in VARIANTS + ["clean"]:
            ta = s[s["cond"] == a].groupby(KEY).agg(p=(v, "mean")).reset_index()
            tb = s[s["cond"] == b].groupby(KEY).agg(p=(v, "mean")).reset_index()
            j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
            j = j[j["p_a"].notna() & j["p_b"].notna()]
            m, lo, hi = boot_mean(j["p_a"] - j["p_b"])
            print("    d %-12s %8.3f [%8.3f %8.3f]  n=%d" % (v, m, lo, hi, len(j)))
        ta = s[s["cond"] == a].groupby(KEY).agg(x=(COST, "mean")).reset_index()
        tb = s[s["cond"] == b].groupby(KEY).agg(x=(COST, "mean")).reset_index()
        j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
        m, lo, hi = boot_mean(j["x_a"] - j["x_b"])
        print("    d %-12s %8.2f [%8.2f %8.2f]" % ("cost", m, lo, hi))
    print()

def s5(df, pool):
    print("=== 5. does the condition ordering survive the metric change, pool %s ===" % pool)
    cols = VARIANTS + ["clean", "success"]
    g = df.groupby("cond")[cols].mean()
    print(g.round(3).to_string())
    print("\n  Spearman between metrics across conditions:")
    print(g.corr(method="spearman").round(3).to_string())
    print("\n  a value near 1.0 means the ranking is unchanged and the metric flaw")
    print("  affects magnitudes only. below about 0.9 means it changes conclusions.\n")

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        s1(df, pool); s2(df, pool); s3(df, pool); s4(df, pool); s5(df, pool)