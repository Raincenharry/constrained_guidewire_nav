"""
Poster figure: disjoint aortic arch morphology families.

Train pool  -> ArchType I, II, IV, V   (seen during training)
Test pool   -> ArchType VI, VII        (never seen during training)

Geometry is generated headless from stEVE AorticArch, exactly as
build_steve_env in steve_cmdp.py constructs it. Each arch is drawn as a
frontal silhouette on the (x, z) plane, with the centerlines offset by the
per point vessel radii. A few extra seeds are drawn faintly behind each
representative to show the within family variability.

Run with the eve env already active:
    python figure_arch_families.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eve
from eve.intervention.vesseltree.aorticarch import ArchType

AorticArch = eve.intervention.vesseltree.AorticArch

# ============================================================
# Configuration
# ============================================================
TRAIN_TYPES = [ArchType.I, ArchType.II, ArchType.IV, ArchType.V]
TEST_TYPES = [ArchType.VI, ArchType.VII]

# Match the training anatomy construction in steve_cmdp.py exactly.
SCALING_XYZD = [1.0, 1.0, 1.0, 0.75]

# Seeds start at 1 because AorticArch treats seed 0 as falsy.
REP_SEED = 1                 # the seed drawn as the solid silhouette
VARIANT_SEEDS = [2, 3, 4, 5] # extra seeds drawn faintly to show the family

# Frontal projection: keep x (lateral) and z (long axis), drop y (depth).
PROJ = (0, 2)

# Set to True if the arches render upside down on the first pass.
FLIP_VERTICAL = False

# Colours. Train is one hue, test another, so the two sets separate at a glance.
TRAIN_FILL = "#2c7a7b"
TEST_FILL = "#dd6b20"
TRAIN_BG = "#e6f2f2"
TEST_BG = "#fdf0e4"
VARIANT_ALPHA = 0.10

MARGIN_MM = 12.0     # padding around the anatomy in the shared axes
SCALE_BAR_MM = 50.0  # reference bar length

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_14_aug")
OUT_STEM = "arch_families"
DPI = 300

FONT_BASE = 17
FONT_PANEL = 21
FONT_GROUP = 26
FONT_SUB = 18

# ============================================================
# Geometry helpers
# ============================================================
def make_arch(arch_type, seed):
    """Build one AorticArch and populate its branches."""
    vt = AorticArch(arch_type=arch_type, seed=seed, scaling_xyzd=SCALING_XYZD)
    vt.reset(0)
    return vt


def branch_silhouette(coords, radii, proj):
    """
    Turn one branch centerline into a closed 2D polygon by offsetting each
    centerline point along the local normal by that point's radius.
    """
    p = np.asarray(coords)[:, proj].astype(float)   # (M, 2)
    r = np.asarray(radii, dtype=float)              # (M,)
    d = np.gradient(p, axis=0)                       # local tangent per point
    norm = np.maximum(np.linalg.norm(d, axis=1), 1e-8)
    t = d / norm[:, None]
    perp = np.column_stack([-t[:, 1], t[:, 0]])      # rotate tangent by 90 deg
    left = p + perp * r[:, None]
    right = p - perp * r[:, None]
    return np.vstack([left, right[::-1]])            # forward one side, back other


def draw_arch(ax, vt, fill_color):
    """Draw one arch: faint variant centerlines are added separately."""
    for br in vt.branches:
        poly = branch_silhouette(br.coordinates, br.radii, PROJ)
        ax.fill(poly[:, 0], poly[:, 1], color=fill_color, lw=0, zorder=3)


def draw_variants(ax, arch_type, hue):
    """Overlay a few other seeds of the same type as faint filled halos."""
    for sd in VARIANT_SEEDS:
        vt = make_arch(arch_type, sd)
        for br in vt.branches:
            poly = branch_silhouette(br.coordinates, br.radii, PROJ)
            ax.fill(poly[:, 0], poly[:, 1], color=hue,
                    alpha=VARIANT_ALPHA, lw=0, zorder=1)


# ============================================================
# Build the representatives and the shared axis limits
# ============================================================
def collect(types):
    reps = {}
    for at in types:
        reps[at] = make_arch(at, REP_SEED)
    return reps


def shared_limits(all_reps):
    xs, ys = [], []
    for vt in all_reps.values():
        cl = np.asarray(vt.centerline_coordinates)[:, PROJ]
        rmax = float(np.asarray([r for br in vt.branches for r in br.radii]).max())
        xs.append(cl[:, 0].min() - rmax)
        xs.append(cl[:, 0].max() + rmax)
        ys.append(cl[:, 1].min() - rmax)
        ys.append(cl[:, 1].max() + rmax)
    x0, x1 = min(xs) - MARGIN_MM, max(xs) + MARGIN_MM
    y0, y1 = min(ys) - MARGIN_MM, max(ys) + MARGIN_MM
    return x0, x1, y0, y1


def style_axis(ax, x0, x1, y0, y1):
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    if FLIP_VERTICAL:
        ax.set_ylim(y1, y0)
    else:
        ax.set_ylim(y0, y1)
    ax.axis("off")


def add_scale_bar(ax, x0, y0):
    x_start = x0 + MARGIN_MM
    y_line = y0 + MARGIN_MM
    ax.plot([x_start, x_start + SCALE_BAR_MM], [y_line, y_line],
            color="#333333", lw=4, solid_capstyle="butt", zorder=5)
    ax.text(x_start + SCALE_BAR_MM / 2.0, y_line + 4,
            "%d mm" % int(SCALE_BAR_MM), ha="center", va="bottom",
            fontsize=FONT_SUB, color="#333333")


def type_label(arch_type):
    return "Type " + str(arch_type).replace("ArchType.", "")


# ============================================================
# Figure
# ============================================================
def build_figure():
    plt.rcParams.update({
        "font.size": FONT_BASE,
        "font.family": "DejaVu Sans",
    })

    train_reps = collect(TRAIN_TYPES)
    test_reps = collect(TEST_TYPES)
    all_reps = {**train_reps, **test_reps}
    x0, x1, y0, y1 = shared_limits(all_reps)

    fig = plt.figure(figsize=(21, 8))
    sub_train, sub_test = fig.subfigures(1, 2, width_ratios=[4, 2], wspace=0.03)
    sub_train.set_facecolor(TRAIN_BG)
    sub_test.set_facecolor(TEST_BG)

    # Train group
    sub_train.suptitle("Training pool", fontsize=FONT_GROUP, fontweight="bold",
                       y=0.98)
    sub_train.text(0.5, 0.905, "arch types seen during training",
                   ha="center", va="top", fontsize=FONT_SUB, color="#2a5b5b")
    ax_train = sub_train.subplots(1, len(TRAIN_TYPES))
    for i, at in enumerate(TRAIN_TYPES):
        ax = ax_train[i]
        draw_variants(ax, at, TRAIN_FILL)
        draw_arch(ax, train_reps[at], TRAIN_FILL)
        style_axis(ax, x0, x1, y0, y1)
        ax.set_title(type_label(at), fontsize=FONT_PANEL, fontweight="bold",
                     pad=6)
        if i == 0:
            add_scale_bar(ax, x0, y0)

    # Test group
    sub_test.suptitle("Test pool", fontsize=FONT_GROUP,
                      fontweight="bold", y=0.98)
    sub_test.text(0.5, 0.905, "arch types never seen during training",
                  ha="center", va="top", fontsize=FONT_SUB, color="#9c4221")
    ax_test = sub_test.subplots(1, len(TEST_TYPES))
    for i, at in enumerate(TEST_TYPES):
        ax = ax_test[i]
        draw_variants(ax, at, TEST_FILL)
        draw_arch(ax, test_reps[at], TEST_FILL)
        style_axis(ax, x0, x1, y0, y1)
        ax.set_title(type_label(at), fontsize=FONT_PANEL, fontweight="bold",
                     pad=6)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, OUT_STEM + ".png")
    pdf = os.path.join(OUT_DIR, OUT_STEM + ".pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    build_figure()
