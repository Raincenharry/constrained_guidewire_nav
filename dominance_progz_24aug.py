import numpy as np
import pandas as pd
import frontier

COST = frontier.COST


def seed_table(df):
    rows = []
    for (c, s), d in df.groupby(["cond", "seed"]):
        rows.append(dict(cond=c, seed=s, n=len(d),
                         prog=d["prog"].mean(),
                         prog_z=d["prog_z"].mean(),
                         cost=d[COST].mean(),
                         jam=d["jam"].mean()))
    return pd.DataFrame(rows).sort_values(["cond", "seed"])


def dominance(t, col, pool):
    print("=== dominance on %s, pool %s ===" % (col, pool))
    print("  an R4 seed dominates a SACPID seed if it has")
    print("  strictly more %s AND strictly less cost.\n" % col)
    r4 = t[t["cond"].str.startswith("r4tip")]
    sp = t[t["cond"].str.startswith("sacpid")]
    print("  %-28s %8s %10s %s" % ("sacpid seed", col, "cost", "dominated by"))
    none_by_cond = {}
    for _, b in sp.iterrows():
        dom = r4[(r4[col] > b[col]) & (r4["cost"] < b["cost"])]
        names = ", ".join("%s s%d" % (r["cond"].replace("r4tip_w", "w"), r["seed"])
                          for _, r in dom.iterrows())
        short = b["cond"].replace("sacpid_tip_", "")
        if len(dom) == 0:
            none_by_cond[short] = none_by_cond.get(short, 0) + 1
        print("  %-28s %8.3f %10.2f %s" %
              ("%s s%d" % (short, b["seed"]), b[col], b["cost"],
               names if names else "NONE"))
    print("\n  reverse, SACPID seeds dominating R4 seeds:")
    n_rev = 0
    weights = set()
    for _, a in r4.iterrows():
        dom = sp[(sp[col] > a[col]) & (sp["cost"] < a["cost"])]
        if len(dom):
            n_rev += 1
            weights.add(a["cond"])
            print("  %s s%d dominated by %s" %
                  (a["cond"], a["seed"],
                   ", ".join("%s s%d" % (r["cond"], r["seed"])
                             for _, r in dom.iterrows())))
    if n_rev == 0:
        print("    none")
    print("\n  COUNTS on %s, pool %s" % (col, pool))
    for c, d in sp.groupby("cond"):
        short = c.replace("sacpid_tip_", "")
        print("    %-16s undominated %d of %d" %
              (short, none_by_cond.get(short, 0), len(d)))
    print("    r4 seeds dominated by a sacpid seed: %d of %d, across %d weights"
          % (n_rev, len(r4), len(weights)))
    print()


if __name__ == "__main__":
    for pool in ["train", "test"]:
        print("############ pool %s ############" % pool)
        df = frontier.load(pool)
        df = df.copy()
        df["prog_z"] = np.where(df["jam"] == 1, 0.0, df["prog"])
        t = seed_table(df)
        print("=== seed table, pool %s, check prog against the 6 aug read ===" % pool)
        print(t.round(3).to_string(index=False))
        print()
        dominance(t, "prog", pool)
        dominance(t, "prog_z", pool)
