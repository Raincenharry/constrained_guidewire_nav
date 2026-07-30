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
BUDGET = {"sacpid_tip_w20_d20": 20.0, "sacpid_tip_w20_d30": 30.0,
          "sacpid_tip_w20_d150": 150.0}
WEIGHT = {"r4tip_w0.0003": 0.0003, "r4tip_w0.001": 0.001,
          "r4tip_w0.003": 0.003, "r4tip_w0.01": 0.01, "r4tip_w0.03": 0.03}
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
    df = df[df[COST].notna()]
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["prog"] = df["frac"].clip(upper=1.0)
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
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
    print("=== 1. how much of the cost is the jam, pool %s ===" % pool)
    tot = df[COST].sum()
    jt = df[df["jam"] == 1][COST].sum()
    print("  jam episodes %d of %d (%.1f pct), carrying %.1f pct of all cost"
          % (int(df["jam"].sum()), len(df),
             100.0 * df["jam"].mean(), 100.0 * jt / tot))
    print("\n  %-24s %7s %8s %10s %10s %10s" %
          ("cond", "n", "jam_rt", "cost_all", "cost_nojam", "pct_jam"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        nj = d[d["jam"] == 0]
        t = d[COST].sum()
        pj = 100.0 * d[d["jam"] == 1][COST].sum() / t if t > 0 else float("nan")
        print("  %-24s %7d %8.3f %10.2f %10.2f %10.1f" %
              (c, len(d), d["jam"].mean(), d[COST].mean(), nj[COST].mean(), pj))
    print()

def s2(df, pool):
    print("=== 2. efficiency EXCLUDING jam episodes, pool %s ===" % pool)
    print("  if the ordering survives here, safety is not merely jam avoidance.\n")
    nj = df[df["jam"] == 0]
    ref = nj[nj["cond"] == REF].groupby(KEY).agg(
        prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
    print("  %-24s %6s %28s %10s %10s" %
          ("cond", "npair", "cost per prog 95pct", "d prog", "d cost"))
    for c in sorted(nj["cond"].unique()):
        if c == REF:
            continue
        t = nj[nj["cond"] == c].groupby(KEY).agg(
            prog=("prog", "mean"), cost=(COST, "mean")).reset_index()
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        if len(j) < 8:
            print("  %-24s %6d  too few paired anatomies" % (c, len(j)))
            continue
        dc = (j["cost_ref"] - j["cost"]).values
        dp = (j["prog_ref"] - j["prog"]).values
        m, lo, hi = boot_ratio(dc, dp)
        print("  %-24s %6d %10.1f [%8.1f %8.1f] %10.3f %10.2f" %
              (c, len(j), m, lo, hi, float(dp.mean()), float(dc.mean())))
    print()

def s3(df, pool):
    print("=== 3. one point per SEED, immune to seed composition, pool %s ===" % pool)
    rows = []
    for (c, s), d in df.groupby(["cond", "seed"]):
        rows.append(dict(cond=c, seed=s, n=len(d),
                         prog=d["prog"].mean(), cost=d[COST].mean(),
                         ins=d["inserted_final"].mean(),
                         suc=d["success"].mean(), jam=d["jam"].mean()))
    t = pd.DataFrame(rows).sort_values(["cond", "seed"])
    print(t.round(3).to_string(index=False))
    r4 = t[t["cond"].str.startswith("r4tip")]
    sp = t[t["cond"].str.startswith("sacpid")]
    print("\n  seed level dominance: an R4 seed dominates a SACPID seed if it has")
    print("  strictly more progress AND strictly less cost.\n")
    print("  %-28s %8s %10s %s" %
          ("sacpid seed", "prog", "cost", "dominated by"))
    for _, b in sp.iterrows():
        dom = r4[(r4["prog"] > b["prog"]) & (r4["cost"] < b["cost"])]
        names = ", ".join("%s s%d" % (r["cond"].replace("r4tip_w", "w"), r["seed"])
                          for _, r in dom.iterrows())
        print("  %-28s %8.3f %10.2f %s" %
              ("%s s%d" % (b["cond"].replace("sacpid_tip_", ""), b["seed"]),
               b["prog"], b["cost"], names if names else "NONE"))
    print("\n  and the reverse, SACPID seeds dominating R4 seeds:")
    any_rev = False
    for _, a in r4.iterrows():
        dom = sp[(sp["prog"] > a["prog"]) & (sp["cost"] < a["cost"])]
        if len(dom):
            any_rev = True
            print("  %s s%d dominated by %s" %
                  (a["cond"], a["seed"],
                   ", ".join("%s s%d" % (r["cond"], r["seed"])
                             for _, r in dom.iterrows())))
    if not any_rev:
        print("    none")
    print()
    return t

def s4(t, pool):
    print("=== 4. is d30 dominated by d20 a seed artefact, pool %s ===" % pool)
    for c in ["sacpid_tip_w20_d20", "sacpid_tip_w20_d30"]:
        d = t[t["cond"] == c]
        if len(d) == 0:
            continue
        print("  %-24s seeds %s" % (c, list(d["seed"])))
        print("    prog %s" % np.round(d["prog"].values, 3))
        print("    cost %s" % np.round(d["cost"].values, 2))
    a = t[t["cond"] == "sacpid_tip_w20_d20"]
    b = t[t["cond"] == "sacpid_tip_w20_d30"]
    if len(a) and len(b):
        print("\n  pooled: d20 prog %.3f cost %.2f, d30 prog %.3f cost %.2f"
              % (a["prog"].mean(), a["cost"].mean(),
                 b["prog"].mean(), b["cost"].mean()))
        ov = b[b["prog"] > b["prog"].min()]
        print("  d30 dropping its worst seed: prog %.3f cost %.2f (n=%d)"
              % (ov["prog"].mean(), ov["cost"].mean(), len(ov)))
        print("  best seed each: d20 prog %.3f cost %.2f, d30 prog %.3f cost %.2f"
              % (a.loc[a["prog"].idxmax(), "prog"], a.loc[a["prog"].idxmax(), "cost"],
                 b.loc[b["prog"].idxmax(), "prog"], b.loc[b["prog"].idxmax(), "cost"]))
    print()

def s5(t, pool):
    print("=== 5. is d predictive of realised cost, per seed, pool %s ===" % pool)
    print("  the interpretability claim in its testable form. no fitting is used:")
    print("  the prediction for SACPID is simply cost = d.\n")
    print("  %-28s %8s %10s %10s %10s" %
          ("seed", "d", "realised", "ratio", "abs err"))
    errs = []
    for _, r in t.iterrows():
        b = BUDGET.get(r["cond"])
        if b is None:
            continue
        e = abs(r["cost"] - b) / b
        errs.append(e)
        print("  %-28s %8.0f %10.2f %10.2f %10.2f" %
              ("%s s%d" % (r["cond"].replace("sacpid_tip_", ""), r["seed"]),
               b, r["cost"], r["cost"] / b, e))
    if errs:
        print("\n  mean relative error of the identity prediction: %.2f" % np.mean(errs))
        print("  fraction of seeds under budget: %.2f" %
              np.mean([1.0 if x <= 0 else 0.0 for x in
                       [t.loc[i, "cost"] - BUDGET[t.loc[i, "cond"]]
                        for i in t.index if t.loc[i, "cond"] in BUDGET]]))
    print("\n  the same question for R4, where the knob has no physical units.")
    print("  leave one weight out, fit log cost against log w on the rest,")
    print("  predict the held out weight. this is what tuning would have to do.\n")
    r4 = t[t["cond"].isin(WEIGHT)].copy()
    r4["w"] = r4["cond"].map(WEIGHT)
    g = r4.groupby("w").agg(cost=("cost", "mean")).reset_index()
    g = g[g["cost"] > 0]
    print("  %-12s %12s %12s %10s" % ("w", "realised", "predicted", "rel err"))
    errs2 = []
    for i in range(len(g)):
        tr = g.drop(g.index[i])
        te = g.iloc[i]
        if len(tr) < 2:
            continue
        k = np.polyfit(np.log(tr["w"]), np.log(tr["cost"]), 1)
        pred = float(np.exp(np.polyval(k, np.log(te["w"]))))
        e = abs(pred - te["cost"]) / te["cost"]
        errs2.append(e)
        print("  %-12.4f %12.2f %12.2f %10.2f" % (te["w"], te["cost"], pred, e))
    if errs2:
        print("\n  mean relative error of the fitted prediction: %.2f" % np.mean(errs2))
    print("\n  if the identity error is comparable to or lower than the fitted error,")
    print("  the budget is a usable knob and the fixed weight is not.\n")

def s6(df, pool):
    print("=== 6. what a target cost would take under each method, pool %s ===" % pool)
    print("  for each target, the SACPID answer is d = target, one run.")
    print("  the R4 answer is whichever weight lands nearest, found by sweeping.\n")
    g = df[df["cond"].isin(WEIGHT)].groupby("cond").agg(
        cost=(COST, "mean"), prog=("prog", "mean")).reset_index()
    sp = df[df["cond"].isin(BUDGET)].groupby("cond").agg(
        cost=(COST, "mean"), prog=("prog", "mean")).reset_index()
    print("  %8s %28s %10s %10s" % ("target", "nearest R4", "its cost", "its prog"))
    for tgt in [5, 10, 20, 30, 50]:
        if len(g) == 0:
            continue
        i = (g["cost"] - tgt).abs().idxmin()
        print("  %8d %28s %10.2f %10.3f" %
              (tgt, g.loc[i, "cond"], g.loc[i, "cost"], g.loc[i, "prog"]))
    print("\n  SACPID arms actually run, for comparison:")
    for _, r in sp.iterrows():
        print("  %-28s d %5.0f  realised %8.2f  prog %6.3f" %
              (r["cond"], BUDGET[r["cond"]], r["cost"], r["prog"]))
    print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        s1(df, pool)
        s2(df, pool)
        t = s3(df, pool)
        s4(t, pool)
        s5(t, pool)
        s6(df, pool)