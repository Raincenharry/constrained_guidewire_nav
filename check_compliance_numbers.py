# Verifies the hardcoded compliance numbers against evals/, and harvests R4 costs.
# Replicates the seed level cost of fig_frontier in figs_06_aug.py:
# per seed cost = mean of shadow_tip over that seed's evaluation episodes.
# Reads only. Writes nothing.

import os, re, glob
import numpy as np
import pandas as pd

EVAL_DIR = "evals"
COST = "shadow_tip"
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")

# what I hardcoded in fig_compliance.py, from note 6.1, test pool
HARD = {
    "sacpid_tip_w20_d20":  [13.1, 18.7, 9.1, 27.9, 33.1, 8.9, 9.0, 10.9, 18.0, 13.1, 20.3],
    "sacpid_tip_w20_d30":  [4.9, 3.4, 82.3, 11.1],
    "sacpid_tip_w20_d150": [61.1, 89.3, 55.5],
}

def per_seed_cost(pool):
    rows = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*_seed100.csv"))):
        m = PAT.match(os.path.basename(path))
        if m is None or m.group("pool") != pool:
            continue
        d = pd.read_csv(path)
        if COST not in d.columns:
            continue
        d = d[d[COST].notna()]
        if len(d) < 100:
            continue
        rows.append(dict(cond=m.group("cond"), seed=int(m.group("seed")),
                         n=len(d), cost=float(d[COST].mean())))
    return pd.DataFrame(rows)

print("=" * 64)
print("CHECK 1: hardcoded compliance numbers against evals/ (test pool)")
print("=" * 64)
t = per_seed_cost("test")
for cond, hard in HARD.items():
    g = t[t.cond == cond].sort_values("seed")
    actual = g["cost"].tolist()
    print("\n%s" % cond)
    print("  seeds present:", g["seed"].tolist())
    print("  actual per seed :", ["%.1f" % v for v in actual])
    print("  hardcoded (note):", ["%.1f" % v for v in hard])
    a_sorted = sorted(round(v, 1) for v in actual)
    h_sorted = sorted(round(v, 1) for v in hard)
    print("  multiset match (1 dp):", a_sorted == h_sorted)
    if len(actual):
        print("  seed mean actual %.2f   note pooled would be %.2f"
              % (np.mean(actual), np.mean(hard)))
        under = sum(1 for v in actual if v < float(cond.split("_d")[1]))
        print("  under budget: %d of %d" % (under, len(actual)))

print("\n" + "=" * 64)
print("CHECK 2: R4 per weight per seed cost, test AND train pool")
print("   (for the interpretability contrast figure, SACPID monotone in d")
print("    versus R4 not monotone in w)")
print("=" * 64)
for pool in ("test", "train"):
    p = per_seed_cost(pool)
    r4 = p[p.cond.str.startswith("r4tip")].copy()
    print("\n-- %s pool --" % pool)
    for cond, g in r4.groupby("cond"):
        g = g.sort_values("seed")
        print("  %-16s seeds %s  costs %s  mean %.2f"
              % (cond, g["seed"].tolist(),
                 ["%.2f" % v for v in g["cost"]], g["cost"].mean()))
