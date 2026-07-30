import re, glob, os, itertools
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
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    df["prog"] = df["frac"].clip(upper=1.0)
    df["prog_z"] = np.where(df["jam"] == 1, 0.0, df["prog"])
    return df

def seed_table(df):
    rows = []
    for (c, s), d in df.groupby(["cond", "seed"]):
        rows.append(dict(cond=c, seed=int(s), n=len(d),
                         prog=d["prog"].mean(), prog_z=d["prog_z"].mean(),
                         cost=d[COST].mean(), jam=d["jam"].mean(),
                         suc=d["success"].mean(), ins=d["inserted_final"].mean()))
    return pd.DataFrame(rows).sort_values(["cond", "seed"]).reset_index(drop=True)

def add_eff(t, progcol):
    r = t[t["cond"] == REF]
    p0, c0 = r[progcol].mean(), r["cost"].mean()
    dp = p0 - t[progcol]
    dc = c0 - t["cost"]
    t = t.copy()
    t["eff"] = np.where(np.abs(dp) > 1e-6, dc / dp, np.nan)
    t["dprog"] = dp
    t["dcost"] = dc
    return t

def s1(t, pool):
    print("=== 1. seed level table, pool %s ===" % pool)
    print(t.round(3).to_string(index=False))
    print("\n  spread within condition, the thing anatomy level intervals hide:")
    print("  %-24s %6s %10s %10s %10s %10s" %
          ("cond", "seeds", "prog_min", "prog_max", "cost_min", "cost_max"))
    for c, d in t.groupby("cond"):
        print("  %-24s %6d %10.3f %10.3f %10.2f %10.2f" %
              (c, len(d), d["prog_z"].min(), d["prog_z"].max(),
               d["cost"].min(), d["cost"].max()))
    print()

def s2(df, t, pool):
    print("=== 2. bootstrap over SEEDS against bootstrap over anatomies, pool %s ===" % pool)
    print("  the seed level interval is the honest one for a claim about the method.\n")
    print("  %-24s %6s %26s %26s" %
          ("cond", "seeds", "prog_z over seeds", "prog_z over anatomies"))
    for c in sorted(t["cond"].unique()):
        v = t[t["cond"] == c]["prog_z"].values
        if len(v) >= 2:
            idx = RNG.integers(0, len(v), size=(NBOOT, len(v)))
            s = v[idx].mean(axis=1)
            slo, shi = np.percentile(s, [2.5, 97.5])
        else:
            slo = shi = float("nan")
        g = df[df["cond"] == c].groupby(KEY)["prog_z"].mean().values
        idx = RNG.integers(0, len(g), size=(NBOOT, len(g)))
        a = g[idx].mean(axis=1)
        alo, ahi = np.percentile(a, [2.5, 97.5])
        print("  %-24s %6d  %6.3f [%6.3f %6.3f]  %6.3f [%6.3f %6.3f]" %
              (c, len(v), v.mean(), slo, shi, g.mean(), alo, ahi))
    print("\n  if the seed interval is several times wider, every earlier interval")
    print("  in this project understates the uncertainty of the method claim.\n")

def perm(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool_v = np.concatenate([a, b]); n = len(a); N = len(pool_v)
    cnt = tot = 0
    for idx in itertools.combinations(range(N), n):
        mask = np.zeros(N, bool); mask[list(idx)] = True
        d = pool_v[mask].mean() - pool_v[~mask].mean()
        tot += 1
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / tot, tot

def s3(t, pool):
    print("=== 3. exact permutation on seeds, pool %s ===" % pool)
    print("  every possible relabelling is enumerated, so the p value is exact.")
    print("  with small seed counts the smallest attainable p is printed too.\n")
    pairs = [("r4tip_w0.01", "sacpid_tip_w20_d30"),
             ("r4tip_w0.003", "sacpid_tip_w20_d20"),
             ("r4tip_w0.001", "sacpid_tip_w20_d20"),
             ("r4tip_w0.01", "sacpid_tip_w20_d20")]
    for a, b in pairs:
        ta, tb = t[t["cond"] == a], t[t["cond"] == b]
        if len(ta) == 0 or len(tb) == 0:
            continue
        print("  %s (%d seeds) against %s (%d seeds)" % (a, len(ta), b, len(tb)))
        for col in ["prog_z", "cost", "eff"]:
            va = ta[col].dropna().values; vb = tb[col].dropna().values
            if len(va) < 1 or len(vb) < 1:
                continue
            o, p, n = perm(va, vb)
            print("    %-8s obs diff %9.3f   p = %.3f  over %d relabellings"
                  % (col, o, p, n))
        print()

def s4(t, pool):
    print("=== 4. what the incoming d20 seeds would have to do, pool %s ===" % pool)
    a, b = "r4tip_w0.01", "sacpid_tip_w20_d20"
    ta, tb = t[t["cond"] == a], t[t["cond"] == b]
    if len(ta) == 0 or len(tb) == 0:
        print("  one arm missing\n"); return
    for col in ["eff", "prog_z", "cost"]:
        va = ta[col].dropna().values; vb = tb[col].dropna().values
        if len(va) == 0 or len(vb) == 0:
            continue
        tgt = va.mean()
        for n_new in [3]:
            need = (tgt * (len(vb) + n_new) - vb.sum()) / n_new
            print("  %-8s %s mean %8.3f, %s currently %8.3f over %d seeds"
                  % (col, a, tgt, b, vb.mean(), len(vb)))
            print("           %d new seeds would need to average %8.3f to draw level"
                  % (n_new, need))
            print("           current %s seed range %8.3f to %8.3f"
                  % (b, vb.min(), vb.max()))
    print("\n  if the required average lies outside the observed seed range, the")
    print("  comparison cannot be overturned by the seeds already running.\n")

if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = load(pool)
        t = add_eff(seed_table(df), "prog_z")
        s1(t, pool); s2(df, t, pool); s3(t, pool); s4(t, pool)