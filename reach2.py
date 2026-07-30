import re, glob, os, itertools
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
PLEN = "path_length_at_reset"
UNCON = ["r1_baseline", "sacpid_tip_w20_d150"]
BUCKLE_N = 500.0
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
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["prog"] = df["frac"].clip(upper=1.0)
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

def solvable_sets(df):
    have = [c for c in UNCON if c in set(df["cond"])]
    u = df[df["cond"].isin(have)]
    fixed = u.groupby(KEY)["success"].max().reset_index()
    fixed = fixed[fixed["success"] > 0][KEY]
    allc = df.groupby(KEY)["success"].max().reset_index()
    allc = allc[allc["success"] > 0][KEY]
    return have, fixed, allc

def section_a(df, pool):
    print("=== A. solvable set defined WITHOUT the conditions under test, pool %s ===" % pool)
    have, fixed, allc = solvable_sets(df)
    print("  defined from: %s" % ", ".join(have))
    print("  fixed set %d anatomies, old all condition set %d anatomies" %
          (len(fixed), len(allc)))
    f = set(map(tuple, fixed.values))
    a = set(map(tuple, allc.values))
    print("  in old but not fixed: %d   in fixed but not old: %d" %
          (len(a - f), len(f - a)))
    g = df.groupby(KEY)[PLEN].mean().reset_index()
    g["in_fixed"] = g.set_index(KEY).index.isin(f)
    print("\n  path length by membership of the fixed set:")
    print(g.groupby("in_fixed")[PLEN].describe(
        percentiles=[0.25, 0.5, 0.75]).round(1).to_string())
    print()
    return fixed

def section_b(df, fixed, pool):
    print("=== B. success on the FIXED solvable set, pool %s ===" % pool)
    s = df.merge(fixed, on=KEY, how="inner")
    print("  %d anatomies, %d episodes" % (s.groupby(KEY).ngroups, len(s)))
    print("%-24s %6s %10s %10s %10s %10s %10s" %
          ("cond", "seeds", "succ_fix", "succ_all", "ins", "prog", "tip"))
    for c in sorted(df["cond"].unique()):
        a = s[s["cond"] == c]
        b = df[df["cond"] == c]
        if len(a) == 0:
            continue
        print("%-24s %6d %10.3f %10.3f %10.1f %10.3f %10.2f" % (
            c, a["seed"].nunique(), a["success"].mean(), b["success"].mean(),
            a["inserted_final"].mean(), a["prog"].mean(),
            a["tip_steps_over_threshold"].mean()))
    print()

def section_c(df, pool):
    print("=== C. is success just reaching far enough, episode level, pool %s ===" % pool)
    d = df[df["cond"].isin(UNCON)]
    if len(d) == 0:
        d = df
    print("  reference arms only, %d episodes" % len(d))
    print("\n  %8s %8s %8s %8s %8s %8s" %
          ("t_frac", "n_pred", "prec", "recall", "acc", "base"))
    base = float(d["success"].mean())
    for t in [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05]:
        pred = (d["frac"] >= t)
        tp = float((pred & (d["success"] > 0)).sum())
        npred = float(pred.sum())
        prec = tp / npred if npred else float("nan")
        rec = tp / max(float((d["success"] > 0).sum()), 1.0)
        acc = float((pred.astype(int) == (d["success"] > 0).astype(int)).mean())
        print("  %8.2f %8d %8.3f %8.3f %8.3f %8.3f" %
              (t, int(npred), prec, rec, acc, max(base, 1 - base)))
    print("\n  the residual: episodes that reached the path length but failed anyway")
    r = d[(d["frac"] >= 1.0) & (d["success"] == 0)]
    q = d[(d["frac"] >= 1.0) & (d["success"] > 0)]
    for lab, s in [("reached and failed", r), ("reached and succeeded", q)]:
        if len(s) == 0:
            continue
        print("    %-22s n=%4d  frac med %.2f  clamped %.3f  buckle %.3f  tip %6.2f" %
              (lab, len(s), s["frac"].median(), s["clamped"].mean(),
               s["buckled"].mean(), s["tip_steps_over_threshold"].mean()))
    print()

def section_d(df, pool):
    print("=== D. normalised progress as the headline metric, pool %s ===" % pool)
    print("  prog = min(insertion / path length, 1), so anatomy difficulty divides out.\n")
    print("%-24s %8s %24s %12s %10s" %
          ("cond", "nanat", "prog 95pct", "tip_per_mm", "succ"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        g = d.groupby(KEY).agg(prog=("prog", "mean"),
                               ins=("inserted_final", "mean"),
                               tip=("tip_steps_over_threshold", "mean"),
                               suc=("success", "mean")).reset_index()
        m, lo, hi = boot_mean(g["prog"])
        tpm = g["tip"].sum() / g["ins"].sum() if g["ins"].sum() > 0 else float("nan")
        print("%-24s %8d  %6.3f [%6.3f %6.3f] %12.4f %10.3f" %
              (c, len(g), m, lo, hi, tpm, g["suc"].mean()))
    print("\n  paired against r1_baseline on progress:")
    tabs = {}
    for c, d in df.groupby("cond"):
        tabs[c] = d.groupby(KEY).agg(prog=("prog", "mean"),
                                     tip=("tip_steps_over_threshold", "mean")).reset_index()
    if "r1_baseline" not in tabs:
        print("  reference absent\n")
        return
    ref = tabs["r1_baseline"]
    print("  %-24s %6s %24s %24s" % ("cond", "npair", "d prog 95pct", "d tip 95pct"))
    for c, t in tabs.items():
        if c == "r1_baseline":
            continue
        j = t.merge(ref, on=KEY, suffixes=("", "_ref"))
        mp, plo, phi = boot_mean(j["prog"] - j["prog_ref"])
        mt, tlo, thi = boot_mean(j["tip"] - j["tip_ref"])
        print("  %-24s %6d  %6.3f [%6.3f %6.3f]  %7.2f [%7.2f %7.2f]" %
              (c, len(j), mp, plo, phi, mt, tlo, thi))
    print()

def section_e(df, fixed, pool):
    print("=== E. head to head on the FIXED set, per seed pair, pool %s ===" % pool)
    s = df.merge(fixed, on=KEY, how="inner")
    pairs = [("r4tip_w0.01", "sacpid_tip_w20_d30"),
             ("r4tip_w0.003", "sacpid_tip_w20_d20"),
             ("r4tip_w0.001", "sacpid_tip_w20_d20")]
    for a, b in pairs:
        if a not in set(s["cond"]) or b not in set(s["cond"]):
            print("  %s against %s: one is missing\n" % (a, b))
            continue
        da, db = s[s["cond"] == a], s[s["cond"] == b]
        ta = da.groupby(KEY).agg(ins=("inserted_final", "mean"),
                                 tip=("tip_steps_over_threshold", "mean"),
                                 suc=("success", "mean")).reset_index()
        tb = db.groupby(KEY).agg(ins=("inserted_final", "mean"),
                                 tip=("tip_steps_over_threshold", "mean"),
                                 suc=("success", "mean")).reset_index()
        j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
        mi, ilo, ihi = boot_mean(j["ins_a"] - j["ins_b"])
        mt, tlo, thi = boot_mean(j["tip_a"] - j["tip_b"])
        ms, slo, shi = boot_mean(j["suc_a"] - j["suc_b"])
        print("\n  %s minus %s, pooled over %d anatomies" % (a, b, len(j)))
        print("    d insertion %8.1f mm [%7.1f %7.1f]" % (mi, ilo, ihi))
        print("    d tip cost  %8.2f    [%7.2f %7.2f]" % (mt, tlo, thi))
        print("    d success   %8.3f    [%7.3f %7.3f]" % (ms, slo, shi))
        print("\n    every seed pair, to check one seed is not carrying it:")
        print("    %6s %6s %10s %10s %10s" %
              ("seed_a", "seed_b", "d_ins", "d_tip", "d_succ"))
        for sa, sb in itertools.product(sorted(da["seed"].unique()),
                                        sorted(db["seed"].unique())):
            xa = da[da["seed"] == sa].groupby(KEY).agg(
                ins=("inserted_final", "mean"),
                tip=("tip_steps_over_threshold", "mean"),
                suc=("success", "mean")).reset_index()
            xb = db[db["seed"] == sb].groupby(KEY).agg(
                ins=("inserted_final", "mean"),
                tip=("tip_steps_over_threshold", "mean"),
                suc=("success", "mean")).reset_index()
            k = xa.merge(xb, on=KEY, suffixes=("_a", "_b"))
            if len(k) == 0:
                continue
            print("    %6d %6d %10.1f %10.2f %10.3f" % (
                sa, sb, (k["ins_a"] - k["ins_b"]).mean(),
                (k["tip_a"] - k["tip_b"]).mean(),
                (k["suc_a"] - k["suc_b"]).mean()))
    print()

def section_f(df, pool):
    print("=== F. cost at matched progress rather than matched insertion, pool %s ===" % pool)
    print("  bands on prog, so anatomy difficulty is held fixed as well as reach.\n")
    bands = [(0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    for lo, hi in bands:
        d = df[(df["prog"] >= lo) & (df["prog"] < hi)]
        print("  prog in [%.2f, %.2f), %d episodes" % (lo, hi, len(d)))
        print("  %-24s %7s %9s %9s %9s" % ("cond", "n", "tip", "dev", "succ"))
        for c in sorted(d["cond"].unique()):
            s = d[d["cond"] == c]
            if len(s) < 15:
                continue
            print("  %-24s %7d %9.2f %9.2f %9.3f" % (
                c, len(s), s["tip_steps_over_threshold"].mean(),
                s["steps_over_threshold"].mean(), s["success"].mean()))
        print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        fixed = section_a(df, pool)
        section_b(df, fixed, pool)
        section_c(df, pool)
        section_d(df, pool)
        section_e(df, fixed, pool)
        section_f(df, pool)