"""
Figures from data already on disk. Reads ~/project/data, writes ~/project/figures.

Only covers figures whose data exists. The three condition comparison figures
wait on the SACPID sweep and are not here yet.

Run from the project root with the eve env active.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.expanduser("~/project/data")
OUT = os.path.expanduser("~/project/figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def training_env(d):
    """OmniSafe builds two envs in one process and both write to the same CSV.
    Keep the training one."""
    return d[d.env_instance == d.env_instance.value_counts().idxmax()].reset_index(drop=True)


MIN_STEPS = 250000   # exclude runs still in progress


def fig_lambda():
    """PLACEHOLDER. 20 July smoke run, 10 epochs at 500 steps per epoch, cost_limit 25.
    Lambda is per epoch and trustworthy. Cost here is OmniSafe's training rollout
    mean, which is windowed, so this shows the controller engaging and NOT
    convergence to d. Replace with a sweep run once one is past 20 epochs, taking
    ep_cost from episodes_seed*.csv rather than from progress.csv."""
    hits = glob.glob(os.path.join(DATA, "omnisafe_runs", "**", "progress.csv"), recursive=True)
    if not hits:
        print("skip fig_lambda: no smoke run progress.csv")
        return
    g = pd.read_csv(sorted(hits)[-1])
    ep = range(len(g))

    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    ax1.plot(ep, g["Metrics/EpCost"], color="#c0392b", marker="o", ms=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("episode cost [N steps]", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_ylim(0, g["Metrics/EpCost"].max() * 1.1)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ep, g["Metrics/LagrangeMultiplier"], color="#2c3e50", marker="s", ms=3)
    ax2.set_ylabel("Lagrange multiplier", color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    ax2.set_ylim(-0.02, g["Metrics/LagrangeMultiplier"].max() * 1.15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_lambda_PLACEHOLDER.png"))
    plt.close(fig)
    print("f1_lambda_PLACEHOLDER.png  (smoke run, not converged, replace after sweep)")


def fig_probe():
    """Success plateaus while force cost keeps falling. The premise of the project."""
    PLATEAU_AT = 113.0

    f = os.path.join(DATA, "runs_r1_probe600k", "episodes_seed7.csv")
    if not os.path.exists(f):
        print("skip fig_probe: no probe CSV")
        return
    d = training_env(pd.read_csv(f))
    d["cum"] = d.steps.cumsum()

    x, succ, shadow = [], [], []
    for i in range(0, len(d), 200):
        b = d.iloc[i:i + 200]
        if len(b) < 100:
            continue
        x.append(b.cum.iloc[-1] / 1000.0)
        succ.append(b.success.mean() * 100)
        shadow.append(b.hinge_cost_shadow.median())

    plateau = [s for xi, s in zip(x, succ) if xi >= PLATEAU_AT]
    pmean = sum(plateau) / len(plateau)

    fig, ax1 = plt.subplots(figsize=(5.8, 3.6))

    # Marker for where the plateau begins, so it is not left to the eye.
    ax1.axvline(PLATEAU_AT, color="#95a5a6", lw=1, alpha=0.8, zorder=1)
    ax1.text(PLATEAU_AT - 8, 1.0, "113k", ha="right", fontsize=8, color="#7f8c8d")

    # Plateau line drawn only over the plateau, not back across the rise.
    ax1.plot([PLATEAU_AT, x[-1]], [pmean, pmean],
             color="#2c3e50", ls="--", lw=1, alpha=0.7, zorder=2)
    ax1.text(PLATEAU_AT + 8, pmean + 1.3, "plateau mean %.1f%%" % pmean,
             ha="left", fontsize=9, color="#2c3e50")

    ax1.plot(x, succ, color="#2c3e50", marker="o", ms=4, zorder=3)
    ax1.set_xlabel("environment steps [thousands]")
    ax1.set_ylabel("success rate [%]", color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    # Fixed limits chosen so the two series never cross. A crossing on a dual
    # axis carries no meaning and invites a false reading.
    ax1.set_ylim(0, 38)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, shadow, color="#c0392b", marker="s", ms=4, zorder=3)
    ax2.set_ylabel("median episode force cost [N steps]", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(20, 262)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f2_probe.png"))
    plt.close(fig)
    print("f2_probe.png")


def collect_weight_sweep():
    rows = []
    pats = [(os.path.join(DATA, "runs_r4_w*", "episodes_seed*.csv"), None),
            (os.path.join(DATA, "runs_r1_baseline_s*", "episodes_seed*.csv"), 0.0)]
    for pat, forced in pats:
        for f in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(f))
            w = forced
            if w is None:
                w = float(tag.split("_w")[1].split("_s")[0])
            d = training_env(pd.read_csv(f))
            steps = int(d.steps.sum())
            if steps < MIN_STEPS:
                print("  excluded %s, only %d steps, still training" % (tag, steps))
                continue
            b = d.iloc[50:].tail(200)
            rows.append(dict(weight=w, tag=tag, env_steps=steps,
                             success=b.success.mean() * 100,
                             shadow=b.hinge_cost_shadow.median(),
                             inserted=b.inserted_final.mean()))
    return pd.DataFrame(rows)


def fig_weight_frontier():
    """Weight on the x axis, so this figure carries the nonlinearity.
    R1 is drawn as five individual points at a nominal x position rather than
    as a shaded band. A band on a linear panel and a band on a log panel have
    wildly different visual weight even when they describe the same five runs."""
    t = collect_weight_sweep()
    if t.empty:
        print("skip fig_weight_frontier: no completed runs")
        return

    r1 = t[t.weight == 0.0]
    r4 = t[t.weight > 0]
    weights = sorted(r4.weight.unique())
    r1_x = weights[0] / 3.0          # nominal position, weight is actually zero
    sep_x = weights[0] / 1.75

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})

    for ax, col in ((ax0, "success"), (ax1, "shadow")):
        ax.axvline(sep_x, color="#bdc3c7", lw=1, ls=":", zorder=1)
        ax.scatter([r1_x] * len(r1), r1[col], color="#2c3e50", s=34, zorder=3)
        ax.scatter(r4.weight, r4[col], color="#c0392b", s=34, zorder=3)
        m = r4.groupby("weight")[col].mean()
        ax.plot(m.index, m.values, color="#c0392b", lw=1, alpha=0.45, zorder=2)

    ax0.set_ylabel("success rate [%]")
    ax0.set_ylim(-1.5, max(r1.success.max(), r4.success.max()) * 1.35)

    ax1.set_yscale("log")
    ax1.set_ylabel("median episode force cost [N steps]")
    ax1.set_xlabel("force penalty weight")
    ax1.set_ylim(r4.shadow.min() / 3.0, r1.shadow.max() * 3.0)

    ax1.set_xscale("log")
    ax1.set_xlim(r1_x / 1.7, weights[-1] * 2.2)
    # Ticks forced at every position. Matplotlib draws decade ticks only, which
    # left 0.0003 unlabelled.
    ax1.set_xticks([r1_x] + weights)
    ax1.set_xticklabels(["R1"] + ["%g" % w for w in weights])
    ax1.minorticks_off()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f3_weight_frontier.png"))
    plt.close(fig)
    print("f3_weight_frontier.png")
    print(t.sort_values(["weight", "tag"]).to_string(
        index=False, float_format=lambda x: "%.4f" % x))

def fig_weight_scatter():
    """Cost against success, one point per run. Better is up and to the left.
    This is the only form where a run's cost and its success stay attached to
    each other, which matters because w0.0003 seed 2 is worst on both axes: it
    did not trade success for safety, it simply lost."""
    t = collect_weight_sweep()
    if t.empty:
        print("skip fig_weight_scatter: no completed runs")
        return

    style = {
        0.0000: ("#2c3e50", "o", "R1, no penalty"),
        0.0003: ("#c0392b", "o", "w = 0.0003"),
        0.0010: ("#e67e22", "s", "w = 0.001"),
        0.0030: ("#f0b27a", "^", "w = 0.003"),
        0.0100: ("#95a5a6", "D", "w = 0.01"),
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.axhspan(-1.2, 0.9, color="#bdc3c7", alpha=0.22, zorder=1)
    ax.text(t.shadow.max() * 2.0, 0.2, "never reaches a target",
            ha="right", fontsize=8.5, color="#7f8c8d", zorder=2)

    for w, g in t.groupby("weight"):
        key = round(w, 4)
        col, mark, lab = style.get(key, ("#7f8c8d", "o", "w = %g" % w))
        ax.scatter(g.shadow, g.success, color=col, marker=mark, s=44,
                   zorder=3, label=lab, edgecolor="white", linewidth=0.6)
        # Seed labels on the working weight only. Three points, one config.
        if key == 0.0003:
            for _, r in g.iterrows():
                seed = r.tag.split("_s")[-1]
                ax.annotate("s%s" % seed, (r.shadow, r.success),
                            textcoords="offset points", xytext=(7, 4),
                            fontsize=8.5, color=col)

    ax.set_xscale("log")
    ax.set_xlim(t.shadow.min() / 2.5, t.shadow.max() * 2.4)
    ax.set_ylim(-1.2, t.success.max() * 1.35)
    ax.set_xlabel("median episode force cost [N steps]   (lower is safer)")
    ax.set_ylabel("success rate [%]")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f3b_cost_vs_success.png"))
    plt.close(fig)
    print("f3b_cost_vs_success.png")

def fig_constrained(run_tag=None):
    """THE result figure. Lambda from progress.csv, which is per epoch and honest.
    Episode cost from our own CSV binned into epochs, NOT from OmniSafe's EpCost,
    which is windowed. Needs a run past roughly 25 epochs to show settling."""
    cands = sorted(glob.glob(os.path.join(DATA, "runs_sacpid_*")))
    if run_tag:
        cands = [c for c in cands if run_tag in c]
    for rd in cands:
        eps = glob.glob(os.path.join(rd, "episodes_seed*.csv"))
        prog = glob.glob(os.path.join(rd, "**", "progress.csv"), recursive=True)
        cfgs = glob.glob(os.path.join(rd, "**", "config.json"), recursive=True)
        if not (eps and prog and cfgs):
            continue

        import json
        d_val = json.load(open(cfgs[0]))["lagrange_cfgs"]["cost_limit"]
        g = pd.read_csv(sorted(prog)[-1])
        if len(g) < 20:
            print("  %s only %d epochs, too early" % (os.path.basename(rd), len(g)))
            continue

        d = training_env(pd.read_csv(eps[0]))
        d["cum"] = d.steps.cumsum()
        d["epoch"] = (d.cum // 6000).astype(int)
        per = d.groupby("epoch").hinge_cost_shadow.median()

        fig, ax1 = plt.subplots(figsize=(5.8, 3.6))
        ax1.plot(per.index, per.values, color="#c0392b", lw=1.6)
        ax1.axhline(d_val, color="#c0392b", ls=":", lw=1.4)
        ax1.text(per.index.max() * 0.97, d_val * 1.25, "d = %g" % d_val,
                 ha="right", fontsize=9, color="#c0392b")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("median episode cost [N steps]", color="#c0392b")
        ax1.tick_params(axis="y", labelcolor="#c0392b")
        ax1.set_ylim(0, max(per.max(), d_val) * 1.2)

        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(range(len(g)), g["Metrics/LagrangeMultiplier"], color="#2c3e50", lw=1.6)
        ax2.set_ylabel("Lagrange multiplier", color="#2c3e50")
        ax2.tick_params(axis="y", labelcolor="#2c3e50")

        name = os.path.basename(rd).replace("runs_", "")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "f5_%s.png" % name))
        plt.close(fig)
        print("f5_%s.png" % name)

def fig_arch():
    """Success against path length, with the two pool means drawn. The pool
    comparison is the point of the split and it was not on the plot at all."""
    files = sorted(glob.glob(os.path.join(DATA, "evals", "*.csv")))
    if not files:
        print("skip fig_arch: no eval CSVs")
        return
    d = pd.concat([pd.read_csv(f) for f in files])
    d["arch_str"] = d.arch_type.astype(str)

    EXCLUDE = "ArchType.II"
    tr = d[d.train_pool == 1]
    te = d[d.train_pool == 0]
    tr_mean = tr[tr.arch_str != EXCLUDE].success.mean() * 100
    te_mean = te.success.mean() * 100

    g = d.groupby("arch_type").agg(n=("success", "size"),
                                   success=("success", "mean"),
                                   path=("path_length_at_reset", "mean"),
                                   shadow=("hinge_cost_shadow", "median"))
    pool_of = d.groupby("arch_type").train_pool.first()
    g = g.sort_values("path")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.axhline(tr_mean, color="#2c3e50", ls="--", lw=1, alpha=0.75, zorder=1)
    ax.axhline(te_mean, color="#c0392b", ls=":", lw=1.4, alpha=0.85, zorder=1)
    ax.text(g.path.max() + 5, tr_mean + 4.0,
            "training pool excl. II  %.1f%%" % tr_mean,
            ha="right", fontsize=8.5, color="#2c3e50")
    ax.text(g.path.max() + 5, tr_mean + 2.3,
            "held out pool  %.1f%%" % te_mean,
            ha="right", fontsize=8.5, color="#c0392b")

    for name, r in g.iterrows():
        label = str(name).replace("ArchType.", "")
        held = pool_of[name] == 0
        ax.scatter(r.path, r.success * 100, s=54, zorder=3,
                   color="white" if held else "#2c3e50",
                   edgecolor="#2c3e50", linewidth=1.4)
        ax.annotate(label, (r.path, r.success * 100),
                    textcoords="offset points", xytext=(7, 5), fontsize=10)

    # Without this the filled zero at long path reads as "long, therefore hard",
    # which is the reading the data does not support.
    ax.annotate("trains at 9.5%,\nfails 125 of 125\ndeterministic",
                xy=(g.path.max(), 0.0), xytext=(-8, 22),
                textcoords="offset points", ha="right", fontsize=8.5,
                color="#7f8c8d")

    ax.scatter([], [], s=54, color="#2c3e50", edgecolor="#2c3e50", label="training pool")
    ax.scatter([], [], s=54, color="white", edgecolor="#2c3e50", label="held out")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_xlabel("mean path length [mm]")
    ax.set_ylabel("success rate [%]")
    ax.set_xlim(g.path.min() - 4, g.path.max() + 6)
    ax.set_ylim(-2.5, max(g.success) * 100 * 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f4_arch_pathlength.png"))
    plt.close(fig)
    print("f4_arch_pathlength.png")
    print(g.to_string(float_format=lambda x: "%.3f" % x))
    print("  training pool excl II %.3f   held out pool %.3f"
          % (tr_mean / 100, te_mean / 100))

def fig_arch_transfer():
    """Stochastic training rate against deterministic evaluation rate for the
    four training anatomies. Three sit near the diagonal. Type II sits on the
    floor, which is the finding: the anatomy is not hard, the mean action
    collapses a capability that only survives under exploration noise."""
    files = sorted(glob.glob(os.path.join(DATA, "evals", "*.csv")))
    runs = sorted(glob.glob(os.path.join(DATA, "runs_r1_baseline_s*",
                                         "episodes_seed*.csv")))
    if not files or not runs:
        print("skip fig_arch_transfer: missing eval or baseline data")
        return

    ev = pd.concat([pd.read_csv(f) for f in files])
    ev = ev[ev.train_pool == 1]
    ev_rate = ev.groupby("arch_type").success.mean() * 100

    late = []
    for f in runs:
        d = training_env(pd.read_csv(f))
        late.append(d.iloc[50:].tail(200))
    tr_rate = pd.concat(late).groupby("arch_type").success.mean() * 100

    common = [a for a in tr_rate.index if a in ev_rate.index]
    if not common:
        print("skip fig_arch_transfer: no shared arch types")
        return

    hi = max(tr_rate[common].max(), ev_rate[common].max()) * 1.25

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, hi], [0, hi], color="#bdc3c7", ls="--", lw=1, zorder=1)
    ax.text(hi * 0.62, hi * 0.68, "equal transfer", fontsize=8.5,
            color="#7f8c8d", rotation=45, rotation_mode="anchor")

    for a in common:
        label = str(a).replace("ArchType.", "")
        collapsed = ev_rate[a] < 1.0
        ax.scatter(tr_rate[a], ev_rate[a], s=58, zorder=3,
                   color="#c0392b" if collapsed else "#2c3e50",
                   edgecolor="white", linewidth=0.6)
        ax.annotate(label, (tr_rate[a], ev_rate[a]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlim(0, hi)
    ax.set_ylim(-1.0, hi)
    ax.set_xlabel("training success, stochastic [%]")
    ax.set_ylabel("evaluation success, deterministic [%]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f4b_arch_transfer.png"))
    plt.close(fig)
    print("f4b_arch_transfer.png")
    print(pd.DataFrame({"train": tr_rate[common], "eval": ev_rate[common]}).to_string(
        float_format=lambda x: "%.1f" % x))

if __name__ == "__main__":
    fig_lambda()
    fig_probe()
    fig_weight_frontier()
    fig_weight_scatter()
    fig_arch()
    fig_arch_transfer()
    fig_constrained()
