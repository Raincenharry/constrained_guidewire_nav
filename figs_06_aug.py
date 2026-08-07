"""
figs_31_july.py

Three poster figures plus the matched insertion jam test.
Reads evals/ with the jam.py loader. Writes figs_31_july/ and prints
every number so it can be quoted without reopening the PNGs.

Run from the project root with the eve env active.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_DIR = "evals"
OUT = "figs_06_aug"
DROP = ["r1", "r4_w0.0003"]
EXCLUDE_FIG = ["sacpid_tip_ki8_d20"]
PAT = re.compile(r"^(?P<cond>.+)_s(?P<seed>\d+)_(?P<pool>train|test)_seed100\.csv$")
KEY = ["arch_type", "arch_seed"]
PLEN = "path_length_at_reset"
COST = "shadow_tip"
DEV = "force_max"
REF = "r1_baseline"
REFSET = ["r1_baseline", "sacpid_tip_w20_d150"]
MATCH_MM = 125.4
BUCKLE_N = 500.0
THRESHOLDS = [100, 200, 300, 400, 500, 750, 1000]
NBOOT = 5000
RNG = np.random.default_rng(0)
CONNECT = False  # hairlines from seed to mean: tried and rejected, too noisy
                 # with eleven d20 seeds. faint points are deliberately not
                 # individually identified, say so in the caption.

os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT = {
    "r1_baseline": "R1",
    "r4tip_w0.0003": "w0.0003",
    "r4tip_w0.001": "w0.001",
    "r4tip_w0.003": "w0.003",
    "r4tip_w0.01": "w0.01",
    "r4tip_w0.03": "w0.03",
    "sacpid_tip_w20_d20": "d20",
    "sacpid_tip_w20_d30": "d30",
    "sacpid_tip_w20_d150": "d150",
    "sacpid_tip_ki8_d20": "ki8 d20",
}


def family(c):
    if c == REF:
        return "unconstrained"
    if c.startswith("r4tip"):
        return "fixed penalty"
    return "budget"


STYLE = {
    "unconstrained": dict(color="0.15", marker="s"),
    "fixed penalty": dict(color="#1f77b4", marker="o"),
    "budget": dict(color="#d62728", marker="^"),
}

# linestyle and marker per condition, so the threshold sweep is traceable
# within a colour family
LINE = {
    "r1_baseline":         ("-",  "o"),
    "r4tip_w0.0003":       ("-",  "o"),
    "r4tip_w0.001":        ("--", "s"),
    "r4tip_w0.003":        ("-.", "^"),
    "r4tip_w0.01":         (":",  "D"),
    "r4tip_w0.03":         ((0, (3, 1, 1, 1)), "v"),
    "sacpid_tip_w20_d150": ("-",  "o"),
    "sacpid_tip_w20_d20":  ("--", "s"),
    "sacpid_tip_w20_d30":  ("-.", "^"),
    "sacpid_tip_ki8_d20":  (":",  "D"),
}

# candidate label offsets in points, tried in order. the first candidate whose
# text box misses every condition mean marker, the legend, and every already
# placed label wins. replaces the hand tuned offsets, which were valid for one
# pool and one set of seed counts only.
LBL_CAND = [(9, 6), (9, -14), (-9, 6), (-9, -14),
            (9, 18), (9, -26), (-9, 18), (-9, -26),
            (0, 20), (0, -30)]

# manual override, only needed if a rerun prints the fallback warning.
# key is the condition name, value is an offset in points.
LBL_FORCE = {}

MARK_HALF = 8.0   # half size in points of the s=130 condition mean marker
LBL_PAD = 3.0     # clearance in points around every placed label


def _overlap(a, b):
    """a and b are (x0, y0, x1, y1) boxes in display points."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def place_labels(fig, ax, items, fontsize=8, avoid=None):
    """items is a list of (cond, x, y, text) in data coordinates.

    Places each label at the first candidate offset that collides with
    nothing already on the axis. Works in display coordinates, so it is
    independent of the axis range and needs no retuning when a pool changes
    or a seed count grows. Labels are placed from the top of the plot
    downwards so the result is deterministic."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = sorted(items, key=lambda it: -it[2])
    pts = [ax.transData.transform((x, y)) for _, x, y, _ in items]
    taken = [(px - MARK_HALF, py - MARK_HALF, px + MARK_HALF, py + MARK_HALF)
             for px, py in pts]
    for x, y in (avoid or []):
        px, py = ax.transData.transform((x, y))
        taken.append((px - 4.0, py - 4.0, px + 4.0, py + 4.0))
    leg = ax.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(renderer=rend)
        taken.append((bb.x0 - LBL_PAD, bb.y0 - LBL_PAD,
                      bb.x1 + LBL_PAD, bb.y1 + LBL_PAD))
    for (cond, x, y, text) in items:
        cands = [LBL_FORCE[cond]] if cond in LBL_FORCE else LBL_CAND
        chosen, box = cands[0], None
        for dx, dy in cands:
            t = ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize)
            bb = t.get_window_extent(renderer=rend)
            trial = (bb.x0 - LBL_PAD, bb.y0 - LBL_PAD,
                     bb.x1 + LBL_PAD, bb.y1 + LBL_PAD)
            t.remove()
            if not any(_overlap(trial, o) for o in taken):
                chosen, box = (dx, dy), trial
                break
        t = ax.annotate(text, (x, y), textcoords="offset points",
                        xytext=chosen, fontsize=fontsize)
        if box is None:
            bb = t.get_window_extent(renderer=rend)
            box = (bb.x0 - LBL_PAD, bb.y0 - LBL_PAD,
                   bb.x1 + LBL_PAD, bb.y1 + LBL_PAD)
            print("  LABEL FALLBACK for %s, no candidate was free" % cond)
        taken.append(box)


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
    df = df[df[COST].notna()].copy()
    df["clamped"] = (df["clamp_steps"] > 0).astype(float)
    df["frac"] = df["inserted_final"] / df[PLEN]
    df["jam"] = ((df["frac"] > 1.15) & (df["success"] == 0)).astype(float)
    df["prog"] = df["frac"].clip(upper=1.0)
    df["prog_z"] = np.where(df["jam"] == 1, 0.0, df["prog"])
    df["buckle"] = (df[DEV] > BUCKLE_N).astype(float)
    return df


def boot(x, min_n=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < min_n:
        return (float(x.mean()) if n else np.nan), np.nan, np.nan
    idx = RNG.integers(0, n, size=(NBOOT, n))
    s = x[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired_diff(d, cond, metric, thresh_mm, mode="any"):
    """mode 'any':  an anatomy enters if at least one episode of that arm
                    cleared thresh_mm. Permissive, and for a low insertion
                    arm it selects that arm's furthest episodes.
       mode 'mean': an anatomy enters only if the arm's mean insertion over
                    its episodes cleared thresh_mm. Stricter, and this is the
                    reading that produces the low retention counts recorded
                    in the 30 July notes."""
    a = d[d["cond"] == REF]
    b = d[d["cond"] == cond]
    if mode == "any":
        ga = a[a["inserted_final"] > thresh_mm].groupby(KEY)[metric].mean().rename("ref")
        gb = b[b["inserted_final"] > thresh_mm].groupby(KEY)[metric].mean().rename("arm")
    else:
        ma = a.groupby(KEY)["inserted_final"].mean()
        mb = b.groupby(KEY)["inserted_final"].mean()
        keep = ma.index[ma > thresh_mm].intersection(mb.index[mb > thresh_mm])
        ga = a.groupby(KEY)[metric].mean().reindex(keep).rename("ref")
        gb = b.groupby(KEY)[metric].mean().reindex(keep).rename("arm")
    j = pd.concat([ga, gb], axis=1).dropna()
    return (j["arm"] - j["ref"]).values


def fig_buckle(df, pool):
    d = df[(df["clamp_steps"] == 0) & (~df["cond"].isin(EXCLUDE_FIG))].copy()
    order = (d.groupby("cond")["inserted_final"].mean()
             .sort_values(ascending=False).index.tolist())

    print("\n=== FIG A panel 1, unclamped buckle rate, pool %s ===" % pool)
    print("  %-10s %7s %11s %22s" % ("cond", "n_unc", "insertion", "buckle rate"))
    rows = []
    for c in order:
        s = d[d["cond"] == c]
        m, lo, hi = boot(s["buckle"].values)
        rows.append((c, len(s), s["inserted_final"].mean(), m, lo, hi))
        print("  %-10s %7d %9.1f mm  %6.3f [%.3f %.3f]" %
              (SHORT.get(c, c), len(s), s["inserted_final"].mean(), m, lo, hi))

    pairs = []
    for mode in ("any", "mean"):
        print("\n=== FIG A panel 2, matched above %.1f mm, mode %s, against R1 ==="
              % (MATCH_MM, mode))
        print("  %-10s %7s %24s" % ("cond", "npair", "d buckle rate"))
        rows2 = []
        for c in order:
            if c == REF:
                continue
            dd = paired_diff(d, c, "buckle", MATCH_MM, mode)
            m, lo, hi = boot(dd)
            rows2.append((c, len(dd), m, lo, hi))
            print("  %-10s %7d %8.3f [%.3f %.3f]" %
                  (SHORT.get(c, c), len(dd), m, lo, hi))
        if mode == "mean":
            pairs = rows2

    print("\n=== FIG A panel 3, threshold sweep, unclamped ===")
    print("  %-10s %s" % ("cond", "".join("%8d" % t for t in THRESHOLDS)))
    sweep = {}
    for c in order:
        s = d[d["cond"] == c]
        r = [float((s[DEV] > t).mean()) for t in THRESHOLDS]
        sweep[c] = r
        print("  %-10s %s" % (SHORT.get(c, c), "".join("%8.3f" % v for v in r)))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    y = np.arange(len(rows))
    for i, (c, n, ins, m, lo, hi) in enumerate(rows):
        st = STYLE[family(c)]
        ax[0].errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=st["marker"],
                       color=st["color"], capsize=3, ms=6)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(["%s\nn=%d, %.0f mm" % (SHORT.get(c, c), n, ins)
                           for c, n, ins, _, _, _ in rows], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("buckle rate, unclamped episodes")
    ax[0].set_title("a. rate by condition\n(ordered by insertion)", fontsize=10)
    ax[0].axvline(0, lw=0.8, color="0.8")

    MIN_PAIR = 0
    pairs_plot = [p for p in pairs if p[1] >= MIN_PAIR and np.isfinite(p[2])]
    y2 = np.arange(len(pairs_plot))
    for i, (c, n, m, lo, hi) in enumerate(pairs_plot):
        st = STYLE[family(c)]
        ax[1].errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=st["marker"],
                       color=st["color"], capsize=3, ms=6)
    ax[1].set_yticks(y2)
    ax[1].set_yticklabels(["%s\nnpair=%d" % (SHORT.get(c, c), n)
                           for c, n, _, _, _ in pairs_plot], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].axvline(0, lw=1.0, color="0.3", ls="--")
    ax[1].set_xlabel("difference in buckle rate against R1")
    ax[1].set_title("b. matched insertion, paired, strict\n(arm mean above %.0f mm)"
                    % MATCH_MM, fontsize=10)

    for c in order:
        st = STYLE[family(c)]
        ls, mk = LINE.get(c, ("-", "."))
        ax[2].plot(THRESHOLDS, sweep[c], marker=mk, ls=ls, color=st["color"],
                   lw=1.3, ms=4.5, label=SHORT.get(c, c))
    ax[2].axvline(BUCKLE_N, lw=0.8, color="0.7", ls=":")
    ax[2].set_xlabel("threshold on device peak force, N")
    ax[2].set_ylabel("buckle rate")
    ax[2].set_title("c. the threshold is a knob,\nthe ordering is stable", fontsize=10)
    ax[2].legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Buckling, %s pool, unclamped episodes only" % pool, fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "figA_buckling_%s.png" % pool)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % p)


def fig_frontier(df, pool):
    d = df[~df["cond"].isin(EXCLUDE_FIG)]
    ps = (d.groupby(["cond", "seed"])
            .agg(prog=("prog_z", "mean"), cost=(COST, "mean"))
            .reset_index())

    print("\n=== FIG B, efficiency frontier, seed level, pool %s ===" % pool)
    print("  %-10s %6s %22s %22s" % ("cond", "seeds", "progress", "cost"))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    seen = set()
    labels = []
    faint = []
    for c, g in ps.groupby("cond"):
        st = STYLE[family(c)]
        lab = family(c) if family(c) not in seen else None
        seen.add(family(c))
        ax.scatter(g["prog"], g["cost"], s=26, color=st["color"],
                   alpha=0.35, marker=st["marker"], zorder=2, linewidths=0)
        mp, mc = g["prog"].mean(), g["cost"].mean()
        if CONNECT:
            for _, r in g.iterrows():
                ax.plot([r["prog"], mp], [r["cost"], mc], color=st["color"],
                        lw=0.5, alpha=0.16, zorder=1)
        ax.scatter([mp], [mc], s=130, color=st["color"], marker=st["marker"],
                   edgecolors="white", linewidths=1.0, zorder=4, label=lab)
        labels.append((c, mp, mc, "%s (%d)" % (SHORT.get(c, c), len(g))))
        for _, r in g.iterrows():
            faint.append((r["prog"], r["cost"]))
        print("  %-10s %6d  %.3f [%.3f %.3f]  %7.2f [%6.2f %6.2f]" % (
            SHORT.get(c, c), len(g), mp, g["prog"].min(), g["prog"].max(),
            mc, g["cost"].min(), g["cost"].max()))

    ax.set_xlabel("progress, jam episodes scored zero, higher is better")
    ax.set_ylabel("integrated tip cost, newton steps, lower is better")
    ax.set_title("Efficiency frontier, %s pool\none faint point per seed, "
                 "solid marker is the condition mean, seed count in brackets"
                 % pool, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.margins(x=0.10, y=0.10)
    fig.tight_layout()
    place_labels(fig, ax, labels, avoid=faint)
    p = os.path.join(OUT, "figB_frontier_%s.png" % pool)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % p)


def fig_path(df, pool):
    ref = df[df["cond"] == REF]
    reach = ref.groupby(KEY)["inserted_final"].median().rename("reach")
    jamr = ref.groupby(KEY)["jam"].mean().rename("jamr")
    plen = df.groupby(KEY)[PLEN].first().rename("path")
    solved = (df[df["cond"].isin(REFSET)].groupby(KEY)["success"]
              .max().rename("solved"))
    t = pd.concat([plen, reach, solved, jamr], axis=1).dropna()

    ok = t[t["solved"] == 1]
    over = t[(t["solved"] == 0) & (t["reach"] >= t["path"])]
    short = t[(t["solved"] == 0) & (t["reach"] < t["path"])]

    print("\n=== FIG C, path length against reach, pool %s ===" % pool)
    print("  solvable set defined from %s" % ", ".join(REFSET))
    print("  %-28s %5s %10s %12s %10s" %
          ("group", "n", "path mean", "reach median", "jam rate"))
    for lab, g in [("solved", ok),
                   ("never solved, fell short", short),
                   ("never solved, reach exceeded", over)]:
        if len(g) == 0:
            print("  %-28s %5d" % (lab, 0))
            continue
        print("  %-28s %5d %10.1f %12.1f %10.3f" %
              (lab, len(g), g["path"].mean(), g["reach"].median(),
               g["jamr"].mean()))

    best, bacc = np.nan, 0.0
    for thr in np.arange(150, 340, 1.0):
        acc = ((t["path"] < thr) == (t["solved"] == 1)).mean()
        if acc > bacc:
            bacc, best = acc, thr
    maj = max((t["solved"] == 1).mean(), (t["solved"] == 0).mean())
    print("  threshold classifier  %.0f mm at %.1f pct against %.1f pct majority"
          % (best, 100 * bacc, 100 * maj))

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    lim = [t["path"].min() * 0.9, t["path"].max() * 1.05]
    ax.plot(lim, lim, color="0.6", lw=1.0, ls="--", label="reach equals demand")
    ax.scatter(short["path"], short["reach"], s=26, facecolors="none",
               edgecolors="#d62728", lw=1.0, label="never solved, fell short")
    ax.scatter(over["path"], over["reach"], s=34, facecolors="none",
               edgecolors="#ff7f0e", lw=1.2, marker="D",
               label="never solved, reach exceeded demand")
    ax.scatter(ok["path"], ok["reach"], s=26, color="#1f77b4",
               label="solved by R1 or d150")
    ax.axvline(best, color="0.3", lw=1.0, ls=":")
    ax.annotate("%.0f mm" % best, (best, lim[0]), textcoords="offset points",
                xytext=(4, 6), fontsize=8, color="0.3")
    ax.set_xlabel("path length to target at reset, mm")
    ax.set_ylabel("median insertion reached by R1, mm")
    ax.set_title("Solvability is a property of the anatomy, %s pool\n"
                 "reach does not track demand" % pool, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "figC_pathlength_%s.png" % pool)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % p)


def jam_matched(df, pool):
    """The deciding test. Is the jam advantage real or is it just low insertion."""
    d = df[~df["cond"].isin(EXCLUDE_FIG)].copy()
    order = (d.groupby("cond")["inserted_final"].mean()
             .sort_values(ascending=False).index.tolist())
    print("\n=== JAM DECISION, pool %s ===" % pool)
    print("  unpaired, all episodes")
    print("  %-10s %9s %9s" % ("cond", "jam rate", "progress"))
    for c in order:
        s = d[d["cond"] == c]
        print("  %-10s %9.3f %9.3f" % (SHORT.get(c, c), s["jam"].mean(),
                                       s["prog_z"].mean()))
    for thr, mode in ((0.0, "any"), (MATCH_MM, "any"), (MATCH_MM, "mean")):
        print("\n  paired against R1, both arms above %.1f mm, mode %s" % (thr, mode))
        print("  %-10s %7s %24s" % ("cond", "npair", "d jam rate"))
        for c in order:
            if c == REF:
                continue
            dd = paired_diff(d, c, "jam", thr, mode)
            m, lo, hi = boot(dd)
            print("  %-10s %7d %8.3f [%.3f %.3f]" %
                  (SHORT.get(c, c), len(dd), m, lo, hi))


if __name__ == "__main__":
    for pool in ("test", "train"):
        print("\n" + "#" * 12 + " pool %s " % pool + "#" * 12)
        df = load(pool)
        fig_buckle(df, pool)
        fig_frontier(df, pool)
        fig_path(df, pool)
        jam_matched(df, pool)