"""
tables_21aug.py

Generates the three report tables for report_draft_2026_08_19_v6.md:

  Table 1  section 3.1  conditions, seed counts, pooled results, both pools
  Table 2  section 3.2  per seed realised cost and budget compliance
  Table 3  section 3.3  identity prediction against fitted leave one out

Run from ~/project in the eve environment. Writes markdown to
reads/2026_08_21_tables.txt and prints the same to stdout.

Cost is shadow_tip. ep_cost is exactly zero on every eval row because
evaluate.py uses ZeroCost(). hinge_cost_shadow is identical to shadow_max
and is the device signal, not the tip signal. Neither is used here.

Bootstrap is 5000 resamples with default_rng(0), matching the four
analysis scripts.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RNG_SEED = 0
N_BOOT = 5000
OUT = "reads/2026_08_21_tables.txt"

# condition name, filename stem glob, expected seed count
CONDITIONS = [
    ("r1_baseline",    "r1_baseline_s*",            5),
    ("r4tip w=0.0003", "r4tip_w0.0003_s*",          2),
    ("r4tip w=0.001",  "r4tip_w0.001_s*",           4),
    ("r4tip w=0.003",  "r4tip_w0.003_s*",           3),
    ("r4tip w=0.01",   "r4tip_w0.01_s*",            3),
    ("r4tip w=0.03",   "r4tip_w0.03_s*",            2),
    ("d = 20",         "sacpid_tip_w20_d20_s*",     11),
    ("d = 30",         "sacpid_tip_w20_d30_s*",     4),
    ("d = 150",        "sacpid_tip_w20_d150_s*",    3),
    ("lowered gain",   "sacpid_tip_ki8_d20_s*",     2),
]

WEIGHTS = [0.0003, 0.001, 0.003, 0.01, 0.03]
WEIGHT_CONDS = ["r4tip w=0.0003", "r4tip w=0.001", "r4tip w=0.003",
                "r4tip w=0.01", "r4tip w=0.03"]
BUDGETS = {"d = 20": 20.0, "d = 30": 30.0, "d = 150": 150.0}

# stems that must never enter any condition, checked explicitly
EXCLUDE = ["_ep30", "run_epoch-50", "cmp_", "canary", "bothsignals", "probe"]


def seed_of(path):
    m = re.search(r"_s(\d+)_(train|test)_seed100\.csv$", path)
    if m is None:
        raise ValueError("cannot read seed from " + path)
    return int(m.group(1))


def files_for(stem, pool):
    pat = os.path.join("evals", stem + "_" + pool + "_seed100.csv")
    got = [f for f in glob.glob(pat) if not any(x in f for x in EXCLUDE)]
    got.sort(key=seed_of)
    return got


def load_episodes(path):
    d = pd.read_csv(path)
    if len(d) != 100:
        raise ValueError("expected 100 episodes, got %d in %s" % (len(d), path))
    frac = d["inserted_final"] / d["path_length_at_reset"]
    success = (d["end_reason"] == "target_reached").astype(int)
    jam = ((frac > 1.15) & (success == 0)).astype(int)
    prog = frac.clip(upper=1.0)
    out = pd.DataFrame({
        "arch_seed": d["arch_seed"],
        "frac": frac,
        "success": success,
        "jam": jam,
        "prog": prog,
        "prog_z": prog.where(jam == 0, 0.0),
        "cost": d["shadow_tip"],
        "buckle": (d["force_max"] > 500.0).astype(int),
        "inserted_final": d["inserted_final"],
    })
    out.attrs["mismatch"] = int((success != d["success"].astype(int)).sum())
    return out


def boot_ci(values, rng=None, statistic=np.mean):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    draws = statistic(values[idx], axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def fmt_ci(lo, hi):
    if np.isnan(lo):
        return "no interval"
    return "[%.3f, %.3f]" % (lo, hi)


def collect():
    """Returns {condition: {pool: {'seeds': [...], 'episodes': DataFrame}}}"""
    data = {}
    problems = []
    for name, stem, expected in CONDITIONS:
        data[name] = {}
        for pool in ("train", "test"):
            paths = files_for(stem, pool)
            if len(paths) != expected:
                problems.append("%s %s: expected %d seeds, found %d"
                                % (name, pool, expected, len(paths)))
            per_seed = []
            frames = []
            for p in paths:
                ep = load_episodes(p)
                if ep.attrs["mismatch"]:
                    problems.append("%s: %d success mismatches"
                                    % (p, ep.attrs["mismatch"]))
                per_seed.append((seed_of(p), ep))
                frames.append(ep)
            data[name][pool] = {
                "seeds": per_seed,
                "episodes": pd.concat(frames, ignore_index=True) if frames else None,
            }
    return data, problems


def table_one(data, rng, w):
    w("## Table 1. Conditions, seed counts and pooled results, both pools\n")
    w("Cost is accumulated hinged tip force in newton steps. Progress is a path")
    w("length fraction. prog excludes jam episodes; prog_z scores them zero.")
    w("Anatomy level intervals resample episodes; seed level intervals resample")
    w("seed means. Conditions with fewer than three seeds print no interval.\n")
    head = ("| Condition | Seeds | Pool | prog_nojam | prog_z | prog_z anatomy 95% "
            "| prog_z seed 95% | Cost (N steps) | Jam rate |")
    w(head)
    w("|" + "|".join([" --- "] * 9) + "|")
    for name, _, _ in CONDITIONS:
        for pool in ("train", "test"):
            blk = data[name][pool]
            if blk["episodes"] is None:
                continue
            ep = blk["episodes"]
            seed_means_z = np.array([e["prog_z"].mean() for _, e in blk["seeds"]])
            nojam = ep.loc[ep["jam"] == 0, "prog"].mean()
            pz = ep["prog_z"].mean()
            anat = ep.groupby("arch_seed")["prog_z"].mean().values
            if len(anat) != 100:
                w("  WARNING %s %s: %d anatomies, expected 100"
                  % (name, pool, len(anat)))
            a_lo, a_hi = boot_ci(anat)
            s_lo, s_hi = boot_ci(seed_means_z)
            w("| %s | %d | %s | %.3f | %.3f | %s | %s | %.2f | %.3f |" % (
                name, len(blk["seeds"]), pool, nojam, pz,
                fmt_ci(a_lo, a_hi), fmt_ci(s_lo, s_hi),
                ep["cost"].mean(), ep["jam"].mean()))
    w("")


def table_two(data, w):
    w("## Table 2. Per seed realised cost and budget compliance\n")
    w("One row per seed. Under budget is realised cost below d on that pool.\n")
    w("| Budget d | Seed | Train cost | Under d | Test cost | Under d |")
    w("|" + "|".join([" --- "] * 6) + "|")
    totals = {"train": [0, 0], "test": [0, 0]}
    for name, d in BUDGETS.items():
        tr = {s: e["cost"].mean() for s, e in data[name]["train"]["seeds"]}
        te = {s: e["cost"].mean() for s, e in data[name]["test"]["seeds"]}
        for s in sorted(tr):
            a, b = tr[s], te[s]
            totals["train"][1] += 1
            totals["test"][1] += 1
            totals["train"][0] += int(a < d)
            totals["test"][0] += int(b < d)
            w("| %s | s%d | %.2f | %s | %.2f | %s |" % (
                name, s, a, "yes" if a < d else "no",
                b, "yes" if b < d else "no"))
        w("| %s | pooled | %.2f | %d of %d | %.2f | %d of %d |" % (
            name,
            np.mean(list(tr.values())), sum(v < d for v in tr.values()), len(tr),
            np.mean(list(te.values())), sum(v < d for v in te.values()), len(te)))
    w("")
    w("All constrained seeds: %d of %d under budget on train, %d of %d on test."
      % (totals["train"][0], totals["train"][1],
         totals["test"][0], totals["test"][1]))
    w("")


def loo_fit(weights, costs):
    """Leave one out log log fit. Returns list of relative errors."""
    lw = np.log(np.asarray(weights, dtype=float))
    lc = np.log(np.asarray(costs, dtype=float))
    errs = []
    for i in range(len(lw)):
        keep = [j for j in range(len(lw)) if j != i]
        slope, intercept = np.polyfit(lw[keep], lc[keep], 1)
        pred = np.exp(slope * lw[i] + intercept)
        errs.append(abs(pred - costs[i]) / costs[i])
    return errs


def table_three(data, w):
    w("## Table 3. Identity prediction against fitted leave one out\n")
    w("The budget predicts realised cost as cost = d, at zero calibration cost.")
    w("The penalty has no equivalent, so its prediction is a leave one out fit of")
    w("log cost on log weight, which costs the full five weight sweep. Interior")
    w("weights interpolate; the two endpoints extrapolate and are reported apart.\n")
    w("| Method | Pool | Prediction | Calibration runs | Mean relative error |")
    w("|" + "|".join([" --- "] * 5) + "|")
    rows = {}
    for pool in ("train", "test"):
        ident = []
        for name, d in BUDGETS.items():
            c = data[name][pool]["episodes"]["cost"].mean()
            ident.append(abs(c - d) / d)
        costs = [data[n][pool]["episodes"]["cost"].mean() for n in WEIGHT_CONDS]
        errs = loo_fit(WEIGHTS, costs)
        rows[pool] = {
            "identity": float(np.mean(ident)),
            "identity_each": ident,
            "interior": float(np.mean(errs[1:4])),
            "all": float(np.mean(errs)),
            "outer": float(np.mean([errs[0], errs[4]])),
            "costs": costs,
        }
    for pool in ("train", "test"):
        r = rows[pool]
        w("| Budget | %s | identity, cost = d | 0 | %.2f |" % (pool, r["identity"]))
        w("| Penalty, interior | %s | fitted, interpolated | 5 | %.2f |"
          % (pool, r["interior"]))
        w("| Penalty, endpoints | %s | fitted, extrapolated | 5 | %.2f |"
          % (pool, r["outer"]))
        w("| Penalty, all five | %s | fitted, pooled | 5 | %.2f |"
          % (pool, r["all"]))
    w("")
    for pool in ("train", "test"):
        r = rows[pool]
        w("Per budget identity error, %s: d20 %.2f, d30 %.2f, d150 %.2f."
          % (pool, r["identity_each"][0], r["identity_each"][1],
             r["identity_each"][2]))
    for pool in ("train", "test"):
        c = rows[pool]["costs"]
        w("Penalty cost by ascending weight, %s: %s."
          % (pool, ", ".join("%.2f" % v for v in c)))
        asc = all(c[i] > c[i + 1] for i in range(len(c) - 1))
        w("  monotone decreasing: %s" % ("yes" if asc else "no"))
    w("")
    return rows


def verification(data, rows, w):
    w("## Verification against report_draft_2026_08_19_v6.md\n")
    w("Every number the draft states in sections 3.1 to 3.3, recomputed.\n")

    d20t = data["d = 20"]["test"]
    per_seed = [e["cost"].mean() for _, e in d20t["seeds"]]
    w("d20 test per seed cost: " + ", ".join("%.1f" % v for v in per_seed))
    w("  draft: 13.1, 18.7, 9.1, 27.9, 33.1, 8.9, 9.0, 10.9, 18.0, 13.1, 20.3")
    over = [v for v in per_seed if v > 20.0]
    w("  seeds over budget: %d, overshoot %s"
      % (len(over), ", ".join("%.0f%%" % (100 * (v / 20 - 1)) for v in over)))
    w("  draft: 3 seeds over, by 39 percent, 66 percent and 2 percent")

    for name, pool, draft in [("d = 20", "train", 10.69), ("d = 20", "test", 16.55),
                              ("d = 30", "train", 12.68), ("d = 30", "test", 25.44),
                              ("d = 150", "train", 71.42), ("d = 150", "test", 68.64)]:
        got = data[name][pool]["episodes"]["cost"].mean()
        flag = "" if abs(got - draft) < 0.02 else "   <<< DIFFERS"
        w("pooled cost %s %s: %.2f, draft %.2f%s" % (name, pool, got, draft, flag))

    tr_max = max(e["cost"].mean() for _, e in data["d = 20"]["train"]["seeds"])
    w("d20 train max per seed cost: %.2f, draft 18.5" % tr_max)

    ep = data["d = 20"]["test"]["episodes"]
    w("d20 test prog_z: %.3f, draft 0.343" % ep["prog_z"].mean())

    w("identity error test %.2f, draft 0.29" % rows["test"]["identity"])
    w("identity error train %.2f, draft 0.52" % rows["train"]["identity"])
    w("penalty interior test %.2f, draft 0.52" % rows["test"]["interior"])
    w("penalty interior train %.2f, draft 0.70" % rows["train"]["interior"])
    w("penalty all five test %.2f, draft 6.14" % rows["test"]["all"])

    seed_errs, under = [], 0
    for name, d in BUDGETS.items():
        for _, e in data[name]["test"]["seeds"]:
            c = e["cost"].mean()
            seed_errs.append(abs(c - d) / d)
            under += int(c < d)
    w("seed level identity error test %.2f over %d seeds, draft 0.54 over 18"
      % (float(np.mean(seed_errs)), len(seed_errs)))
    w("  under budget %d of %d, draft 14 of 18" % (under, len(seed_errs)))

    w("")
    w("Operator framing, section 3.3. Note the two methods are held to different")
    w("targets: the budget to d, the penalty to its own training pool cost,")
    w("because the penalty has no prescribed target. Check the sentence reads fairly.")
    exceed, errs = 0, []
    for name in WEIGHT_CONDS:
        tr = data[name]["train"]["episodes"]["cost"].mean()
        te = data[name]["test"]["episodes"]["cost"].mean()
        exceed += int(te > tr)
        errs.append(abs(te - tr) / tr)
    w("  penalty exceeds own train cost on %d of 5, mean error %.2f, draft 5 of 5 at 0.59"
      % (exceed, float(np.mean(errs))))
    over_b = sum(1 for n, d in BUDGETS.items()
                 if data[n]["test"]["episodes"]["cost"].mean() > d)
    w("  budget over d on %d of 3, at identity error %.2f, draft 0 of 3 at 0.29"
      % (over_b, rows["test"]["identity"]))
    w("")


def main():
    rng = np.random.default_rng(RNG_SEED)
    data, problems = collect()

    lines = []

    def w(s):
        lines.append(s)

    w("# Report tables, generated %s" % pd.Timestamp.today().date())
    w("")
    w("Source: evals/ episode CSVs, cost column shadow_tip, eleven seed basis.")
    w("Script tables_21aug.py. Bootstrap 5000 resamples, default_rng(0).")
    w("Excluded by name: %s." % ", ".join(EXCLUDE))
    w("r1_s* excluded as duplicate of r1_baseline_s*; r4_w0.0003_s* excluded as")
    w("the device signal condition, distinct from r4tip_w0.0003_s*.")
    w("")

    if problems:
        w("## SEED COUNT AND INTEGRITY PROBLEMS")
        w("")
        for p in problems:
            w("  " + p)
        w("")

    table_one(data, rng, w)
    table_two(data, w)
    rows = table_three(data, w)
    verification(data, rows, w)

    text = "\n".join(lines)
    os.makedirs("reads", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print("\nwritten to " + OUT)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
