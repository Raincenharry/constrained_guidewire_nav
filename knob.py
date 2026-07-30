import re, glob, os
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
DROP = ["r1", "r4_w0.0003"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
PLEN = "path_length_at_reset"
COST = "shadow_tip"
BUDGET = {"sacpid_tip_w20_d20": 20.0, "sacpid_tip_w20_d30": 30.0,
          "sacpid_tip_w20_d150": 150.0}
WEIGHT = {"r4tip_w0.0003": 0.0003, "r4tip_w0.001": 0.001, "r4tip_w0.003": 0.003,
          "r4tip_w0.01": 0.01, "r4tip_w0.03": 0.03}

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

def pooled(df):
    return df.groupby("cond").agg(cost=(COST, "mean"), prog=("prog_z", "mean"),
                                  jam=("jam", "mean")).reset_index()

def s1(tr, te):
    print("=== 1. leave one out, interior points separated from extrapolation ===")
    print("  the earlier version mixed the two and the endpoints dominated.\n")
    for name, g0 in [("train", tr), ("test", te)]:
        g = g0[g0["cond"].isin(WEIGHT)].copy()
        g["k"] = g["cond"].map(WEIGHT)
        g = g[g["cost"] > 0].sort_values("k").reset_index(drop=True)
        print("  --- R4, pool %s ---" % name)
        print("  %-10s %10s %12s %10s %14s" %
              ("w", "realised", "predicted", "rel err", "kind"))
        inter, extra = [], []
        for i in range(len(g)):
            trn = g.drop(g.index[i]); tst = g.iloc[i]
            if len(trn) < 2:
                continue
            kind = "interior" if 0 < i < len(g) - 1 else "extrapolation"
            c = np.polyfit(np.log(trn["k"]), np.log(trn["cost"]), 1)
            pred = float(np.exp(np.polyval(c, np.log(tst["k"]))))
            e = abs(pred - tst["cost"]) / tst["cost"]
            (inter if kind == "interior" else extra).append(e)
            print("  %-10.4f %10.2f %12.2f %10.2f %14s" %
                  (tst["k"], tst["cost"], pred, e, kind))
        if inter:
            print("  mean interior error %.2f" % np.mean(inter))
        if extra:
            print("  mean extrapolation error %.2f" % np.mean(extra))
        g2 = g0[g0["cond"].isin(BUDGET)].copy()
        g2["k"] = g2["cond"].map(BUDGET)
        g2 = g2[g2["cost"] > 0].sort_values("k").reset_index(drop=True)
        print("\n  --- SACPID identity, pool %s ---" % name)
        print("  %-10s %10s %10s %10s" % ("d", "realised", "ratio", "rel err"))
        errs = []
        for _, r in g2.iterrows():
            e = abs(r["cost"] - r["k"]) / r["k"]
            errs.append(e)
            print("  %-10.0f %10.2f %10.2f %10.2f" % (r["k"], r["cost"], r["cost"] / r["k"], e))
        if errs:
            print("  mean identity error %.2f" % np.mean(errs))
        print()

def s2(tr, te):
    print("=== 2. the practical question: hit a target cost on unseen anatomy ===")
    print("  R4 route: sweep on train, pick the w whose train cost equals the target,")
    print("  deploy, observe the test cost. SACPID route: set d equal to the target,")
    print("  deploy, no calibration runs at all.\n")
    m = tr.merge(te, on="cond", suffixes=("_tr", "_te"))
    print("  --- R4, target taken as its own train cost ---")
    print("  %-16s %10s %10s %10s %10s" %
          ("w", "target", "achieved", "rel err", "over"))
    errs = []
    for c, w in sorted(WEIGHT.items(), key=lambda kv: kv[1]):
        r = m[m["cond"] == c]
        if len(r) == 0:
            continue
        t0 = float(r["cost_tr"].iloc[0]); a0 = float(r["cost_te"].iloc[0])
        if t0 <= 0:
            continue
        e = abs(a0 - t0) / t0
        errs.append(e)
        print("  %-16.4f %10.2f %10.2f %10.2f %10s" %
              (w, t0, a0, e, "yes" if a0 > t0 else "no"))
    if errs:
        print("  mean transfer error %.2f, over target in %d of %d"
              % (np.mean(errs), sum(1 for c, w in WEIGHT.items()
                 if len(m[m["cond"] == c]) and
                 float(m[m["cond"] == c]["cost_te"].iloc[0]) >
                 float(m[m["cond"] == c]["cost_tr"].iloc[0])), len(errs)))
    print("\n  --- SACPID, target is d, zero calibration runs ---")
    print("  %-16s %10s %10s %10s %10s" % ("d", "target", "achieved", "rel err", "over"))
    errs = []; over = 0
    for c, b in sorted(BUDGET.items(), key=lambda kv: kv[1]):
        r = m[m["cond"] == c]
        if len(r) == 0:
            continue
        a0 = float(r["cost_te"].iloc[0])
        e = abs(a0 - b) / b
        errs.append(e); over += 1 if a0 > b else 0
        print("  %-16.0f %10.2f %10.2f %10.2f %10s" %
              (b, b, a0, e, "yes" if a0 > b else "no"))
    if errs:
        print("  mean error %.2f, over budget in %d of %d" % (np.mean(errs), over, len(errs)))
    print("\n  runs required: R4 needs the whole sweep before the first deployment,")
    print("  SACPID needs none. count the sweep runs from the seed table.\n")

def s3(tr, te):
    print("=== 3. cross pool transfer of the knob ===")
    m = tr.merge(te, on="cond", suffixes=("_tr", "_te"))
    print("  %-24s %10s %10s %10s %10s" %
          ("cond", "cost_tr", "cost_te", "ratio", "family"))
    for _, r in m.sort_values("cost_tr").iterrows():
        fam = "sacpid" if r["cond"] in BUDGET else ("r4" if r["cond"] in WEIGHT else "ref")
        ratio = r["cost_te"] / r["cost_tr"] if r["cost_tr"] > 1e-9 else float("nan")
        print("  %-24s %10.2f %10.2f %10.2f %10s" %
              (r["cond"], r["cost_tr"], r["cost_te"], ratio, fam))
    for fam, keys in [("sacpid", BUDGET), ("r4", WEIGHT)]:
        v = [r["cost_te"] / r["cost_tr"] for _, r in m.iterrows()
             if r["cond"] in keys and r["cost_tr"] > 1e-9]
        if v:
            print("\n  %s transfer ratio: mean %.2f, sd %.2f, range %.2f to %.2f"
                  % (fam, np.mean(v), np.std(v), min(v), max(v)))
    print("\n  a tighter spread means the knob transfers more predictably.\n")

def s4(tr, te):
    print("=== 4. monotonicity of the knob ===")
    for name, g in [("train", tr), ("test", te)]:
        for fam, keys, lab in [("SACPID", BUDGET, "d"), ("R4", WEIGHT, "w")]:
            d = g[g["cond"].isin(keys)].copy()
            if len(d) < 2:
                continue
            d["k"] = d["cond"].map(keys)
            d = d.sort_values("k")
            mono = bool(np.all(np.diff(d["cost"].values) >= 0))
            print("  %-6s %-7s pool %-6s %s ascending -> cost %s, monotone: %s" %
                  (fam, "", name, lab,
                   np.round(d["cost"].values, 2), mono))
    print()

if __name__ == "__main__":
    tr, te = pooled(load("train")), pooled(load("test"))
    s1(tr, te); s2(tr, te); s3(tr, te); s4(tr, te)