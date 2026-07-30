import re, glob, os, itertools
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
PLEN = "path_length_at_reset"
REF = "r1_baseline"
COST = "shadow_tip"
PROXY = "tip_steps_over_threshold"
UNCON = ["r1_baseline", "sacpid_tip_w20_d150"]
BUDGET = {"sacpid_tip_w20_d20": 20.0, "sacpid_tip_w20_d30": 30.0,
          "sacpid_tip_w20_d150": 150.0, "sacpid_tip_ki8_d20": 20.0}
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
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["prog"] = df["frac"].clip(upper=1.0)
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    return df

def has_cost(df, c):
    d = df[df["cond"] == c]
    return len(d) > 0 and not d[COST].isna().all()

def boot_mean(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    s = x[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(x.mean()), lo, hi

def boot_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(num)
    if n < 5 or abs(den.mean()) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(NBOOT, n))
    d = den[idx].mean(axis=1)
    ok = np.abs(d) > 1e-9
    s = num[idx].mean(axis=1)[ok] / d[ok]
    if len(s) < 100:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(num.mean() / den.mean()), lo, hi

def s1(df, pool):
    print("=== 1. cost signal audit, pool %s ===" % pool)
    print("  %s is the constrained quantity in newton steps." % COST)
    print("  %s is a step count and is what every earlier table used.\n" % PROXY)
    print("%-24s %7s %8s %9s %9s %9s %9s %8s" %
          ("cond", "n", "haveC", "cost_mn", "cost_md", "cost_p95", "proxy_mn", "rho"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        nn = int(d[COST].notna().sum())
        if nn == 0:
            print("%-24s %7d %8s %9s %9s %9s %9.2f %8s" %
                  (c, len(d), "0", "n/a", "n/a", "n/a", d[PROXY].mean(), "n/a"))
            continue
        v = d[d[COST].notna()]
        rho = v[COST].corr(v[PROXY], method="spearman")
        print("%-24s %7d %8d %9.2f %9.2f %9.2f %9.2f %8.3f" %
              (c, len(d), nn, v[COST].mean(), v[COST].median(),
               v[COST].quantile(0.95), v[PROXY].mean(), rho))
    print("\n  does each constrained arm meet its budget at evaluation:")
    print("  %-24s %8s %10s %10s %10s %10s" %
          ("cond", "d", "cost_mean", "ratio", "frac_over", "cost_med"))
    for c, b in BUDGET.items():
        if not has_cost(df, c):
            continue
        d = df[(df["cond"] == c) & (df[COST].notna())]
        print("  %-24s %8.0f %10.2f %10.2f %10.3f %10.2f" %
              (c, b, d[COST].mean(), d[COST].mean() / b,
               float((d[COST] > b).mean()), d[COST].median()))
    print("\n  a ratio below 1.0 means the budget is met on average.")
    print("  compare against the unconstrained reference:")
    if has_cost(df, REF):
        d = df[(df["cond"] == REF) & (df[COST].notna())]
        print("  %-24s %8s %10.2f" % (REF, "none", d[COST].mean()))
    print()

def s2(df, pool):
    print("=== 2. frontier on the ACTUAL cost, pool %s ===" % pool)
    print("%-24s %7s %9s %9s %13s %13s" %
          ("cond", "nanat", "prog", "cost", "cost_per_mm", "cost_per_prog"))
    for c in sorted(df["cond"].unique()):
        if not has_cost(df, c):
            continue
        d = df[df["cond"] == c]
        g = d.groupby(KEY).agg(prog=("prog", "mean"), cost=(COST, "mean"),
                               ins=("inserted_final", "mean")).reset_index()
        cpm = g["cost"].sum() / g["ins"].sum()
        cpp = g["cost"].sum() / g["prog"].sum()
        print("%-24s %7d %9.3f %9.2f %13.4f %13.2f" %
              (c, len(g), g["prog"].mean(), g["cost"].mean(), cpm, cpp))
    print("\n  paired against %s:" % REF)
    if not has_cost(df, REF):
        print("  reference has no cost column\n")
        return
    ref = df[df["cond"] == REF].groupby(KEY).agg(
        prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
    print("  %-24s %6s %22s %24s" %
          ("cond", "npair", "d prog 95pct", "d cost 95pct"))
    for c in sorted(df["cond"].unique()):
        if c == REF or not has_cost(df, c):
            continue
        t = df[df["cond"] == c].groupby(KEY).agg(
            prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        mp, plo, phi = boot_mean(j["prog"] - j["prog_ref"])
        mc, clo, chi = boot_mean(j["cost"] - j["cost_ref"])
        print("  %-24s %6d %6.3f [%6.3f %6.3f] %7.2f [%7.2f %7.2f]" %
              (c, len(j), mp, plo, phi, mc, clo, chi))
    print()

def s3(df, pool):
    print("=== 3. efficiency, cost removed per unit of progress given up, pool %s ===" % pool)
    print("  higher is better. this is the single number the whole comparison reduces to.\n")
    if not has_cost(df, REF):
        print("  reference has no cost column\n")
        return
    ref = df[df["cond"] == REF].groupby(KEY).agg(
        prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
    print("  %-24s %6s %28s %10s" %
          ("cond", "npair", "cost per prog 95pct", "d prog"))
    for c in sorted(df["cond"].unique()):
        if c == REF or not has_cost(df, c):
            continue
        t = df[df["cond"] == c].groupby(KEY).agg(
            prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        dc = (j["cost_ref"] - j["cost"]).values
        dp = (j["prog_ref"] - j["prog"]).values
        m, lo, hi = boot_ratio(dc, dp)
        print("  %-24s %6d %10.1f [%8.1f %8.1f] %10.3f" %
              (c, len(j), m, lo, hi, float(dp.mean())))
    print()

def s4(df, pool):
    print("=== 4. head to head on the actual cost, fixed solvable set, pool %s ===" % pool)
    have = [c for c in UNCON if c in set(df["cond"])]
    u = df[df["cond"].isin(have)]
    fx = u.groupby(KEY)["success"].max().reset_index()
    fx = fx[fx["success"] > 0][KEY]
    s = df.merge(fx, on=KEY, how="inner")
    print("  fixed set %d anatomies\n" % s.groupby(KEY).ngroups)
    pairs = [("r4tip_w0.01", "sacpid_tip_w20_d30"),
             ("r4tip_w0.003", "sacpid_tip_w20_d20"),
             ("r4tip_w0.001", "sacpid_tip_w20_d20")]
    for a, b in pairs:
        if not has_cost(s, a) or not has_cost(s, b):
            print("  %s against %s: cost column missing\n" % (a, b))
            continue
        ta = s[s["cond"] == a].groupby(KEY).agg(
            ins=("inserted_final", "mean"), cost=(COST, "mean"),
            prog=("prog", "mean"), suc=("success", "mean")).reset_index()
        tb = s[s["cond"] == b].groupby(KEY).agg(
            ins=("inserted_final", "mean"), cost=(COST, "mean"),
            prog=("prog", "mean"), suc=("success", "mean")).reset_index()
        j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
        print("  %s minus %s over %d anatomies" % (a, b, len(j)))
        for lab, col in [("insertion mm", "ins"), ("cost N.steps", "cost"),
                         ("progress", "prog"), ("success", "suc")]:
            m, lo, hi = boot_mean(j["%s_a" % col] - j["%s_b" % col])
            print("    d %-14s %8.2f [%8.2f %8.2f]" % (lab, m, lo, hi))
        print()

def s5(df, pool):
    print("=== 5. tail against mean, does the Lagrangian control peaks, pool %s ===" % pool)
    print("  a penalty on a sum should cut the mean. a budget on a sum might still")
    print("  differ in how it treats the tail. matched on progress so reach is fixed.\n")
    bands = [(0.60, 0.80), (0.80, 0.95), (0.95, 1.01)]
    for lo, hi in bands:
        d = df[(df["prog"] >= lo) & (df["prog"] < hi)]
        print("  prog in [%.2f, %.2f), %d episodes" % (lo, hi, len(d)))
        print("  %-24s %6s %9s %9s %9s %9s %9s" %
              ("cond", "n", "cost", "tip_mean", "tip_max", "exc_max", "f_max"))
        for c in sorted(d["cond"].unique()):
            x = d[d["cond"] == c]
            if len(x) < 15 or x[COST].isna().all():
                continue
            print("  %-24s %6d %9.2f %9.3f %9.2f %9.2f %9.1f" % (
                c, len(x), x[COST].mean(), x["tip_mean"].mean(),
                x["tip_max"].mean(), x["excess_max"].mean(),
                x["force_max"].mean()))
        print()

def s6(df, pool):
    print("=== 6. the jam, frac above 1.15 and failed, pool %s ===" % pool)
    d = df.copy()
    print("  jam episodes overall: %d of %d" % (int(d["jam"].sum()), len(d)))
    j = d[d["jam"] == 1]
    n = d[d["jam"] == 0]
    print("  %-16s %7s %9s %9s %9s %9s" %
          ("group", "n", "cost", "f_max", "clamped", "prog"))
    for lab, s in [("jam", j), ("not jam", n)]:
        if len(s) == 0:
            continue
        print("  %-16s %7d %9.2f %9.1f %9.3f %9.3f" % (
            lab, len(s),
            s[COST].mean() if not s[COST].isna().all() else float("nan"),
            s["force_max"].mean(), s["clamped"].mean(), s["prog"].mean()))
    print("\n  jam rate by condition, paired against %s:" % REF)
    ref = d[d["cond"] == REF].groupby(KEY)["jam"].mean().reset_index()
    print("  %-24s %7s %9s %24s" % ("cond", "n", "jam_rate", "d jam 95pct"))
    for c in sorted(d["cond"].unique()):
        x = d[d["cond"] == c]
        t = x.groupby(KEY)["jam"].mean().reset_index()
        if c == REF:
            print("  %-24s %7d %9.3f %24s" % (c, len(x), x["jam"].mean(), "reference"))
            continue
        k = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        m, lo, hi = boot_mean(k["jam"] - k["jam_ref"])
        print("  %-24s %7d %9.3f  %6.3f [%6.3f %6.3f]" %
              (c, len(x), x["jam"].mean(), m, lo, hi))
    print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        s1(df, pool)
        s2(df, pool)
        s3(df, pool)
        s4(df, pool)
        s5(df, pool)
        s6(df, pool)