import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1", "r4_w0.0003"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
PLEN, COST, REF = "path_length_at_reset", "shadow_tip", "r1_baseline"
UNCON = ["r1_baseline", "sacpid_tip_w20_d150"]
BUDGET = {"sacpid_tip_w20_d20": 20.0, "sacpid_tip_w20_d30": 30.0,
          "sacpid_tip_w20_d150": 150.0}
WEIGHT = {"r4tip_w0.0003": 0.0003, "r4tip_w0.001": 0.001, "r4tip_w0.003": 0.003,
          "r4tip_w0.01": 0.01, "r4tip_w0.03": 0.03}
NBOOT = 5000
RNG = np.random.default_rng(0)

def load(pool):
    fr = []
    for p in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(p))
        if m is None or m.group("pool") != pool or m.group("cond") in DROP:
            continue
        d = pd.read_csv(p)
        d["cond"] = m.group("cond"); d["seed"] = int(m.group("seed"))
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df[df[COST].notna()].copy()
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    df["prog"] = np.where(df["jam"] == 1, np.nan, df["frac"].clip(upper=1.0))
    return df

def bm(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    i = RNG.integers(0, len(x), size=(NBOOT, len(x)))
    s = x[i].mean(axis=1)
    return float(x.mean()), *np.percentile(s, [2.5, 97.5])

def t1(df, pool):
    print("=== TABLE 1. conditions, pool %s ===" % pool)
    print("  progress excludes jam episodes. cost is %s in newton steps." % COST)
    print("  two intervals: anatomies (n=100) and seeds (n as shown).\n")
    print("  %-22s %5s %26s %24s %8s %8s" %
          ("cond", "seeds", "prog by anatomy", "prog by seed", "cost", "jam"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        g = d.groupby(KEY)["prog"].mean()
        m, lo, hi = bm(g.values)
        sv = d.groupby("seed")["prog"].mean().values
        sm, slo, shi = bm(sv)
        print("  %-22s %5d %7.3f [%6.3f %6.3f] %7.3f [%6.3f %6.3f] %8.2f %8.3f" %
              (c, d["seed"].nunique(), m, lo, hi, sm, slo, shi,
               d[COST].mean(), d["jam"].mean()))
    print()

def t2(df, pool):
    print("=== TABLE 2. paired against %s, pool %s ===" % (REF, pool))
    ref = df[df["cond"] == REF].groupby(KEY).agg(
        p=("prog", "mean"), c=(COST, "mean")).reset_index()
    print("  %-22s %6s %22s %24s %20s" %
          ("cond", "npair", "d prog 95pct", "d cost 95pct", "cost per prog"))
    for c in sorted(df["cond"].unique()):
        if c == REF:
            continue
        t = df[df["cond"] == c].groupby(KEY).agg(
            p=("prog", "mean"), c=(COST, "mean")).reset_index()
        j = t.merge(ref, on=KEY, suffixes=("", "_r"))
        j = j[j["p"].notna() & j["p_r"].notna()]
        if len(j) < 8:
            print("  %-22s %6d  too few" % (c, len(j))); continue
        mp, plo, phi = bm(j["p"] - j["p_r"])
        mc, clo, chi = bm(j["c"] - j["c_r"])
        dp = (j["p_r"] - j["p"]).mean(); dc = (j["c_r"] - j["c"]).mean()
        eff = dc / dp if abs(dp) > 0.02 else np.nan
        print("  %-22s %6d %6.3f [%6.3f %6.3f] %7.2f [%7.2f %7.2f] %14s" %
              (c, len(j), mp, plo, phi, mc, clo, chi,
               ("%.0f" % eff) if not np.isnan(eff) else "unstable"))
    print("\n  cost per prog is suppressed where the progress difference is under")
    print("  0.02, because the ratio is not estimable there.\n")

def t3(df, pool):
    print("=== TABLE 3. the knob, pool %s ===" % pool)
    g = df.groupby("cond").agg(cost=(COST, "mean")).reset_index()
    for fam, keys, lab, desc in [("SACPID", BUDGET, "d", "ascending"),
                                 ("R4", WEIGHT, "w", "descending")]:
        d = g[g["cond"].isin(keys)].copy()
        if len(d) < 2:
            continue
        d["k"] = d["cond"].map(keys); d = d.sort_values("k")
        v = d["cost"].values
        mono = bool(np.all(np.diff(v) >= 0)) if desc == "ascending" \
            else bool(np.all(np.diff(v) <= 0))
        print("  %-8s %s ascending -> cost %s" % (fam, lab, np.round(v, 2)))
        print("           expected %s, monotone: %s" % (desc, mono))
    print("\n  per seed budget compliance:")
    for c, b in sorted(BUDGET.items(), key=lambda kv: kv[1]):
        d = df[df["cond"] == c]
        if len(d) == 0:
            continue
        s = d.groupby("seed")[COST].mean()
        print("    %-24s d %5.0f  seeds %s  over budget %d of %d" %
              (c, b, np.round(s.values, 1), int((s > b).sum()), len(s)))
    print()

def t4(df, pool):
    print("=== TABLE 4. head to head, fixed solvable set, pool %s ===" % pool)
    u = df[df["cond"].isin([c for c in UNCON if c in set(df["cond"])])]
    fx = u.groupby(KEY)["success"].max().reset_index()
    fx = fx[fx["success"] > 0][KEY]
    s = df.merge(fx, on=KEY, how="inner")
    print("  fixed set %d anatomies, defined from the unconstrained arms only\n"
          % s.groupby(KEY).ngroups)
    for a, b in [("r4tip_w0.01", "sacpid_tip_w20_d30"),
                 ("r4tip_w0.003", "sacpid_tip_w20_d20"),
                 ("r4tip_w0.001", "sacpid_tip_w20_d20")]:
        if a not in set(s["cond"]) or b not in set(s["cond"]):
            continue
        print("  %s minus %s  (%d against %d seeds)" %
              (a, b, df[df["cond"] == a]["seed"].nunique(),
               df[df["cond"] == b]["seed"].nunique()))
        for lab, col in [("progress", "prog"), ("cost", COST),
                         ("insertion", "inserted_final"), ("success", "success")]:
            ta = s[s["cond"] == a].groupby(KEY)[col].mean().reset_index()
            tb = s[s["cond"] == b].groupby(KEY)[col].mean().reset_index()
            j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
            j = j[j["%s_a" % col].notna() & j["%s_b" % col].notna()]
            m, lo, hi = bm(j["%s_a" % col] - j["%s_b" % col])
            print("    d %-11s %8.3f [%8.3f %8.3f]  n=%d" % (lab, m, lo, hi, len(j)))
        print()

if __name__ == "__main__":
    for pool in ["test", "train"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        t1(df, pool); t2(df, pool); t3(df, pool); t4(df, pool)