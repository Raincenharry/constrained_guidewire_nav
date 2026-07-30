import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
REF = "r1_baseline"
PLEN = "path_length_at_reset"
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

def section_a(df, pool):
    print("=== A. is reach anatomy independent, pool %s ===" % pool)
    print("  if the agent adapted to the anatomy, insertion would rise with path length.")
    print("  a slope near zero means a fixed reach regardless of what is required.\n")
    print("%-24s %8s %10s %10s %10s" %
          ("cond", "nanat", "rho", "slope", "ins_sd"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        g = d.groupby(KEY).agg(ins=("inserted_final", "mean"),
                               p=(PLEN, "mean")).reset_index()
        rho = g["ins"].corr(g["p"], method="spearman")
        slope = np.polyfit(g["p"], g["ins"], 1)[0]
        print("%-24s %8d %10.3f %10.3f %10.1f" %
              (c, len(g), rho, slope, g["ins"].std()))
    print("\n  slope of 1.0 would mean perfect adaptation, 0.0 a fixed reach.\n")

def section_b(df, pool):
    print("=== B. does path length alone predict solvability, pool %s ===" % pool)
    g = df.groupby(KEY).agg(best=("success", "max"), p=(PLEN, "mean")).reset_index()
    base = float((g["best"] > 0).mean())
    print("  base rate solved: %.3f over %d anatomies" % (base, len(g)))
    print("\n  threshold on path length, predict solved if path below t:")
    print("  %8s %8s %8s %8s %8s" % ("t_mm", "n_below", "solved", "acc", "recall"))
    best_acc, best_t = -1.0, None
    for t in range(180, 341, 20):
        below = g["p"] < t
        pred = below.astype(int)
        acc = float((pred == (g["best"] > 0).astype(int)).mean())
        nb = int(below.sum())
        sr = float(g[below]["best"].gt(0).mean()) if nb else float("nan")
        rec = float((below & (g["best"] > 0)).sum() / max((g["best"] > 0).sum(), 1))
        print("  %8d %8d %8.3f %8.3f %8.3f" % (t, nb, sr, acc, rec))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print("\n  best single threshold %d mm at accuracy %.3f, against %.3f for always"
          " predicting the majority class" % (best_t, best_acc, max(base, 1 - base)))
    print("\n  solve rate by path length decile:")
    g["dec"] = pd.qcut(g["p"], 10, labels=False, duplicates="drop")
    h = g.groupby("dec").agg(n=("best", "size"), p_med=("p", "median"),
                             solved=("best", lambda s: float((s > 0).mean())))
    print(h.round(3).to_string())
    print()

def section_c(df, pool):
    print("=== C. does ArchType II survive matching on path length, pool %s ===" % pool)
    g = df.groupby(KEY).agg(best=("success", "max"), p=(PLEN, "mean"),
                            ins=("inserted_final", "mean")).reset_index()
    if "ArchType.II" not in set(g["arch_type"]):
        print("  ArchType.II not in this pool\n")
        return
    g["is2"] = (g["arch_type"] == "ArchType.II")
    print("  ArchType.II path range %.1f to %.1f, others %.1f to %.1f" % (
        g[g["is2"]]["p"].min(), g[g["is2"]]["p"].max(),
        g[~g["is2"]]["p"].min(), g[~g["is2"]]["p"].max()))
    lo = g[g["is2"]]["p"].min()
    ov = g[(~g["is2"]) & (g["p"] >= lo)]
    print("\n  non II anatomies in the ArchType.II path range (>= %.1f mm):" % lo)
    print("    n = %d, solved %d, rate %.3f" %
          (len(ov), int((ov["best"] > 0).sum()),
           float((ov["best"] > 0).mean()) if len(ov) else float("nan")))
    print("    ArchType.II: n = %d, solved %d, rate %.3f" %
          (int(g["is2"].sum()), int(g[g["is2"]]["best"].gt(0).sum()),
           float(g[g["is2"]]["best"].gt(0).mean())))
    print("\n  solve rate by path bin, ArchType.II against the rest:")
    bins = list(range(175, 361, 25))
    g["bin"] = pd.cut(g["p"], bins)
    t = g.groupby(["bin", "is2"], observed=True).agg(
        n=("best", "size"), solved=("best", lambda s: float((s > 0).mean())))
    print(t.round(3).to_string())
    print("\n  mean insertion by path bin, is ArchType.II reach unusual:")
    t2 = g.groupby(["bin", "is2"], observed=True)["ins"].agg(["size", "mean"])
    print(t2.round(1).to_string())
    print()

def section_d(df, pool):
    print("=== D. overshoot, frac above 1 means the device exceeded the path, pool %s ===" % pool)
    d = df[df["cond"] == REF]
    print("  r1_baseline, %d episodes, frac > 1.0 in %.3f of them"
          % (len(d), float((d["frac"] > 1.0).mean())))
    d = d.copy()
    d["ov"] = (d["frac"] > 1.0)
    print("\n%-10s %6s %8s %10s %10s %10s %10s" %
          ("overshoot", "n", "succ", "tip_cost", "dev_cost", "buckle", "clamped"))
    for k, s in d.groupby("ov"):
        print("%-10s %6d %8.3f %10.2f %10.2f %10.3f %10.3f" % (
            str(k), len(s), s["success"].mean(),
            s["tip_steps_over_threshold"].mean(), s["steps_over_threshold"].mean(),
            s["buckled"].mean(), s["clamped"].mean()))
    print("\n  same split excluding clamped episodes, so it is not just the clamp:")
    u = d[d["clamped"] == 0]
    for k, s in u.groupby("ov"):
        print("%-10s %6d %8.3f %10.2f %10.2f %10.3f" % (
            str(k), len(s), s["success"].mean(),
            s["tip_steps_over_threshold"].mean(), s["steps_over_threshold"].mean(),
            s["buckled"].mean()))
    print("\n  overshoot rate by condition, unclamped only:")
    u2 = df[df["clamped"] == 0]
    for c in sorted(u2["cond"].unique()):
        s = u2[u2["cond"] == c]
        print("    %-24s %6d %8.3f" % (c, len(s), float((s["frac"] > 1.0).mean())))
    print()

def section_e(df, pool):
    print("=== E. is lost success fully explained by lost reach, pool %s ===" % pool)
    g = df.groupby(KEY)[PLEN].mean().reset_index()
    print("  predicted success uses each condition's own median reach and asks what")
    print("  fraction of anatomies have a shorter path. No policy quality involved.\n")
    print("%-24s %10s %12s %12s %10s" %
          ("cond", "reach_med", "pred_succ", "actual", "ratio"))
    for c in sorted(df["cond"].unique()):
        d = df[df["cond"] == c]
        reach = float(d["inserted_final"].median())
        pred = float((g[PLEN] < reach).mean())
        act = float(d["success"].mean())
        print("%-24s %10.1f %12.3f %12.3f %10s" % (
            c, reach, pred, act,
            ("%.2f" % (act / pred)) if pred > 0 else "n/a"))
    print("\n  a ratio near a shared constant means reach explains the ordering and")
    print("  the residual is the same for everyone. A ratio that varies means some")
    print("  conditions convert reach into success better than others.\n")

def section_f(df, pool):
    print("=== F. head to head at matched success and matched cost, pool %s ===" % pool)
    best = df.groupby(KEY)["success"].max().reset_index().rename(
        columns={"success": "best"})
    d = df.merge(best, on=KEY)
    s = d[d["best"] > 0]
    pairs = [("r4tip_w0.01", "sacpid_tip_w20_d30"),
             ("r4tip_w0.003", "sacpid_tip_w20_d20"),
             ("r4tip_w0.001", "sacpid_tip_w20_d20")]
    for a, b in pairs:
        if a not in set(s["cond"]) or b not in set(s["cond"]):
            print("  %s against %s: one is missing" % (a, b))
            continue
        ta = s[s["cond"] == a].groupby(KEY).agg(
            ins=("inserted_final", "mean"), tip=("tip_steps_over_threshold", "mean"),
            suc=("success", "mean")).reset_index()
        tb = s[s["cond"] == b].groupby(KEY).agg(
            ins=("inserted_final", "mean"), tip=("tip_steps_over_threshold", "mean"),
            suc=("success", "mean")).reset_index()
        j = ta.merge(tb, on=KEY, suffixes=("_a", "_b"))
        mi, ilo, ihi = boot_mean(j["ins_a"] - j["ins_b"])
        mt, tlo, thi = boot_mean(j["tip_a"] - j["tip_b"])
        ms, slo, shi = boot_mean(j["suc_a"] - j["suc_b"])
        print("\n  %s minus %s, paired over %d solvable anatomies" % (a, b, len(j)))
        print("    d insertion  %8.1f mm [%7.1f %7.1f]" % (mi, ilo, ihi))
        print("    d tip cost   %8.2f    [%7.2f %7.2f]" % (mt, tlo, thi))
        print("    d success    %8.3f    [%7.3f %7.3f]" % (ms, slo, shi))
    print()

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        section_a(df, pool)
        section_b(df, pool)
        section_c(df, pool)
        section_d(df, pool)
        section_e(df, pool)
        section_f(df, pool)