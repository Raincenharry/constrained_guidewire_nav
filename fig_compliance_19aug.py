# FIG G, poster headline, two panel, test pool, eleven seed basis.
# Left  : SACPID. Realised cost against the budget d you set. Points below the
#         reference line are under budget. Means climb with d (monotone).
# Right : R4. Realised cost against the penalty weight w. The curve is a
#         calibration curve that costs the five weight sweep to draw, and w is not
#         in cost units, so there is no free reference line the way d is.
# All numbers hardcoded from the frozen evals, verified by check_compliance_numbers.py.

import numpy as np
import matplotlib.pyplot as plt

OUT = "figs_19_aug/figG_compliance.png"

SACPID = {
    20:  [13.1, 18.7, 9.1, 27.9, 33.1, 8.9, 9.0, 10.9, 18.0, 13.1, 20.3],
    30:  [4.9, 3.4, 82.3, 11.1],
    150: [61.1, 89.3, 55.5],
}
R4 = {
    0.0003: [103.97, 34.96],
    0.001:  [27.25, 46.11, 53.84, 34.60],
    0.003:  [23.87, 46.82, 26.16],
    0.01:   [16.77, 3.46, 0.98],
    0.03:   [0.40, 0.02],
}

RED, BLUE, GREY = "#d62728", "#1f77b4", "0.35"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 17, "axes.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

rng = np.random.default_rng(0)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True,
                               gridspec_kw={"width_ratios": [1.0, 1.15]})

budgets = sorted(SACPID)
xpos = {d: i for i, d in enumerate(budgets)}
half = 0.34
for d in budgets:
    vals = np.asarray(SACPID[d], float)
    x = xpos[d]
    axL.hlines(d, x - half, x + half, color=GREY, lw=2.4, ls="--", zorder=2,
               label="budget d, set before training" if d == budgets[0] else None)
    jit = (rng.random(vals.size) - 0.5) * 0.42
    axL.scatter(np.full(vals.size, x) + jit, vals, s=95, color=RED,
                edgecolor="white", lw=1.2, alpha=0.9, zorder=3)
    axL.scatter([x], [vals.mean()], s=340, marker="D", color=RED,
                edgecolor="black", lw=1.6, zorder=4,
                label="seed mean" if d == budgets[0] else None)

axL.set_xticks(list(xpos.values()))
axL.set_xticklabels(["%d\n(%d seeds)" % (d, len(SACPID[d])) for d in budgets])
axL.set_xlim(-0.6, len(budgets) - 0.4)
axL.set_xlabel("safety budget d (newton steps)")
axL.set_ylabel("realised episode cost (newton steps)")
axL.set_title("Constrained (SACPID)", fontsize=18, pad=10)
axL.legend(loc="upper left", frameon=False, fontsize=13)
axL.text(0.5, 0.03, "points below the line are under budget",
         transform=axL.transAxes, ha="center", va="bottom", fontsize=13, color=GREY)

weights = sorted(R4)
for w in weights:
    vals = np.asarray(R4[w], float)
    jit = np.exp((rng.random(vals.size) - 0.5) * 0.18)
    axR.scatter(np.full(vals.size, w) * jit, vals, s=95, color=BLUE,
                edgecolor="white", lw=1.2, alpha=0.9, zorder=3)
means = [np.mean(R4[w]) for w in weights]
axR.plot(weights, means, color=BLUE, lw=1.6, alpha=0.45, zorder=2)
axR.scatter(weights, means, s=340, marker="D", color=BLUE,
            edgecolor="black", lw=1.6, zorder=4, label="seed mean")
axR.annotate("calibration curve,\ncosts the 5 weight sweep",
             xy=(0.001, 40.45), xytext=(0.0004, 4),
             fontsize=12.5, color=BLUE,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))

axR.set_xscale("log")
axR.set_xlabel("penalty weight w")
axR.set_title("Fixed penalty (R4)", fontsize=18, pad=10)
axR.legend(loc="upper right", frameon=False, fontsize=13)
axR.text(0.5, 0.03, "w carries no cost units, no target to prescribe",
         transform=axR.transAxes, ha="center", va="bottom", fontsize=13, color=GREY)

axL.set_yscale("log")
axL.set_ylim(0.015, 260)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
