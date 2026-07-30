"""Standalone reader for the 30 July session.
Discovers conditions from eval filenames. Does not import read_29_july.py.
Run from the project root, wherever evals/ lives.
"""
import glob, os, re
import pandas as pd

PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
BAND_LO, BAND_HI = 125.4, 199.4          # p10 to p90 overlap fixed on 29 July, not re tuned
REQUIRED = ["inserted_final", "success"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def inventory(verbose=True):
    rows, skipped = [], []
    for path in sorted(glob.glob("evals/*_seed100.csv")):
        m = PAT.match(os.path.basename(path))
        if m is None:
            skipped.append(path)
            continue
        rows.append(dict(cond=m.group("cond"), seed=int(m.group("seed")),
                         pool=m.group("pool"), path=path))
    inv = pd.DataFrame(rows)
    if verbose:
        print("=== excluded, non standard filename ===")
        for p in skipped:
            print("   ", p)
        print("\n=== conditions discovered, seeds per pool ===")
        t = (inv.groupby(["cond", "pool"]).seed
                .apply(lambda s: ",".join(str(x) for x in sorted(s)))
                .unstack(fill_value=""))
        print(t.to_string())
    return inv


def load(pool, conds=None):
    inv = inventory(verbose=False)
    inv = inv[inv.pool == pool]
    if conds is not None:
        inv = inv[inv.cond.isin(conds)]
    frames = []
    for r in inv.itertuples():
        d = pd.read_csv(r.path)
        d["cond"], d["seed"] = r.cond, r.seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def metrics(g):
    row = {"seeds": g.seed.nunique(), "n": len(g),
           "ins_mean": g.inserted_final.mean(),
           "ins_med": g.inserted_final.median(),
           "ins_q25": g.inserted_final.quantile(0.25),
           "ins_q75": g.inserted_final.quantile(0.75),
           "success": g.success.mean()}
    if "end_reason" in g.columns:
        row["stuck"] = g.end_reason.astype(str).str.lower().str.contains("stuck").mean()
    for col, name in (("steps_over_threshold", "over"),
                      ("tip_steps_over_threshold", "tip_over")):
        if col in g.columns:
            row[name] = g[col].mean()
    return pd.Series(row)


def table(d, by):
    if len(d) == 0:
        return pd.DataFrame()
    return pd.DataFrame({k: metrics(g) for k, g in d.groupby(by)}).T


def main():
    inv = inventory()
    test, train = load("test"), load("train")

    print("\n=== schema, first file ===")
    print(list(pd.read_csv(inv.path.iloc[0]).columns))
    missing = [c for c in REQUIRED if c not in test.columns]
    if missing:
        print("\nSTOP, required columns absent:", missing)
        return
    if "end_reason" in test.columns:
        print("\n=== end_reason vocabulary, test pool ===")
        print(test.end_reason.value_counts().to_string())

    for name, d in (("TEST", test), ("TRAIN", train)):
        print(f"\n=== condition table, {name} pool ===")
        print(table(d, "cond").round(3).to_string())

    print("\n=== per seed, TEST pool ===")
    print(table(test, ["cond", "seed"]).round(3).to_string())

    band = test[(test.inserted_final >= BAND_LO) & (test.inserted_final <= BAND_HI)]
    print(f"\n=== matched insertion band {BAND_LO} to {BAND_HI} mm, per condition ===")
    print(table(band, "cond").round(3).to_string())
    print("\n=== matched band, PER SEED, the section 17 rerun ===")
    print(table(band, ["cond", "seed"]).round(3).to_string())

    edges = list(range(0, 401, 50))
    test = test.assign(ins_bin=pd.cut(test.inserted_final, edges))
    for col in ("steps_over_threshold", "tip_steps_over_threshold"):
        if col not in test.columns:
            continue
        print(f"\n=== mean {col} by insertion bin, TEST pool ===")
        print(test.pivot_table(index="cond", columns="ins_bin", values=col,
                               aggfunc="mean", observed=False).round(2).to_string())
        print(f"--- episode counts per cell ---")
        print(test.pivot_table(index="cond", columns="ins_bin", values=col,
                               aggfunc="size", observed=False).fillna(0).astype(int).to_string())

    print("\n=== w0.03 against the pre registered criterion ===")
    w = table(test[test.cond.astype(str).str.contains("w0.03")], ["cond", "seed"])
    if len(w) == 0:
        print("no w0.03 test evals found")
        return
    print(w.round(3).to_string())
    if "stuck" in w.columns:
        collapse = bool((w.stuck > 0.05).all() and (w.ins_mean < 120).all())
        healthy = bool(((w.ins_mean > 180) & (w.stuck < 0.02)).any())
        print("verdict:", "COLLAPSE" if collapse else ("HEALTHY" if healthy else "AMBIGUOUS"))
    else:
        print("no end_reason column, stuck rate not computable from evals")


if __name__ == "__main__":
    main()