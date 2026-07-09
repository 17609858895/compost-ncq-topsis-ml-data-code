"""
Figure 2. Pooled management-parameter landscape across the 7-target compost database.
Panel A: lower-triangle Spearman correlation (ellipse-coded) among the 10 continuous
         management/substrate parameters shared by all 7 datasets, pooled over the
         unique parameter combinations in the combined database.
Panel B: split half-violin plots showing how the distribution of each continuous
         parameter differs between enclosed and open composting systems.

Style adapted from: 期刊配图下三角椭圆相关热图 (ref #14) and
顶刊配图：下三角相关矩阵 + 特征分组与半边小提琴图 (ref #17).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import (TARGETS, COMMON_FEATURES, TARGET_ORDER, TARGET_LABELS,
                    FEATURE_LABELS, TARGET_COLORS, RESULTS_DIR, CMAP_DIVERGE)
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

CONTINUOUS = [
    'Period (d)', 'Turning times', 'Compost volume (m3)', 'Application Rate (%DW)',
    'V2_Ventilation Interval (min)', 'V3_Ventilation Duration (min)', 'V4_Ventilation Day',
    'V5_Ventilation rate (L/min/kg iniDW)', 'Initial Moisture Content (%)', 'Initial C/N (%)'
]

# ---- Pool the unique parameter combinations across the 7 datasets ----
frames = []
for name in TARGET_ORDER:
    fpath, ycol, code = TARGETS[name]
    df = pd.read_csv(fpath)
    frames.append(df[COMMON_FEATURES])
pooled = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
print("Pooled unique parameter combinations:", pooled.shape)

# ============ Panel A: lower-triangle ellipse correlation heatmap ============
corr = pooled[CONTINUOUS].corr(method="spearman")
labels = [FEATURE_LABELS.get(c, c) for c in CONTINUOUS]
short_labels = [
    "Period", "Turning", "Volume", "Bulking", "Aer. interval",
    "Aer. duration", "Aer. onset", "Aer. rate", "Moisture", "C/N ratio",
]
n = len(CONTINUOUS)

fig = plt.figure(figsize=(13.8, 13.2), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.03, 1], wspace=0.26, hspace=0.38)
axA = fig.add_subplot(gs[0, 0])

cmap = plt.get_cmap(CMAP_DIVERGE)
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

for i in range(n):
    for j in range(n):
        if i > j:  # lower triangle: ellipse
            r = corr.iloc[i, j]
            color = cmap(norm(r))
            width = 0.9
            height = 0.9 * (1 - abs(r) * 0.7)
            angle = 45 if r > 0 else -45
            e = Ellipse((j, i), width=width, height=height, angle=angle,
                        facecolor=color, edgecolor="black", lw=0.4, alpha=0.92)
            axA.add_patch(e)
        elif i < j:  # upper triangle: text
            r = corr.iloc[i, j]
            axA.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=7.5,
                     color="black" if abs(r) < 0.6 else "white")
        else:
            axA.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#f2f2f2", edgecolor="none"))

axA.set_xlim(-0.5, n - 0.5)
axA.set_ylim(n - 0.5, -0.5)
axA.set_xticks(range(n))
axA.set_yticks(range(n))
axA.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8.8)
axA.set_yticklabels(short_labels, fontsize=9.0)
axA.set_aspect("equal")
for spine in axA.spines.values():
    spine.set_visible(False)
axA.tick_params(length=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
caxA = axA.inset_axes([0.10, 1.055, 0.80, 0.045])
cbar = fig.colorbar(sm, cax=caxA, orientation="horizontal")
cbar.set_label("Spearman's $\\rho$", fontsize=10)
cbar.ax.xaxis.set_label_position("top")
cbar.ax.tick_params(labelsize=9.5, length=3.6, width=1.0)
axA.set_title("A", fontsize=14, fontweight="bold", loc="left")

# ============ Panel B: split half-violins by enclosed vs. open system ============
axB = fig.add_subplot(gs[0, 1])

# rank-normalize each continuous feature to [0,1] for fair joint display (robust to outliers)
z = pooled[CONTINUOUS].copy()
for c in CONTINUOUS:
    z[c] = z[c].rank(pct=True)
z["enclosed"] = pooled["M1_is Enclosed"].astype(int)

from scipy.stats import gaussian_kde

positions = np.arange(n)
width = 0.42
colors = {0: "#78BFC2", 1: "#C97979"}
group_labels = {0: "Open system", 1: "Enclosed system"}

for pos, feat in zip(positions, CONTINUOUS):
    for grp, side in zip([1, 0], [1, -1]):  # enclosed on right (+), open on left (-)
        vals = z.loc[z["enclosed"] == grp, feat].dropna().values
        if len(vals) < 3 or np.std(vals) == 0:
            continue
        kde = gaussian_kde(vals)
        ymin, ymax = -0.05, 1.05
        ys = np.linspace(ymin, ymax, 200)
        dens = kde(ys)
        dens = dens / dens.max() * width
        axB.fill_betweenx(ys, pos, pos + side * dens, color=colors[grp], alpha=0.75,
                          edgecolor="black", lw=0.5,
                          label=group_labels[grp] if pos == 0 else None)

axB.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.6)
axB.set_xticks(positions)
axB.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8.8)
axB.set_ylabel("Within-feature percentile rank", fontsize=10)
axB.set_xlim(-0.7, n - 0.3)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.08),
           fontsize=10.6, ncol=2, handlelength=1.2, columnspacing=1.1)
axB.set_title("B", fontsize=14, fontweight="bold", loc="left")

# ============ Panel C: dataset size per target ============
axC = fig.add_subplot(gs[1, 0])
dsum = pd.read_csv(os.path.join(RESULTS_DIR, "dataset_summary.csv")).set_index("Target").reindex(TARGET_ORDER)
bars = axC.bar(TARGET_ORDER, dsum["n_samples"], color=[TARGET_COLORS[t] for t in TARGET_ORDER],
                edgecolor="black", linewidth=0.6)
for b, n_feat in zip(bars, dsum["n_features"]):
    axC.text(b.get_x() + b.get_width()/2, b.get_height() + 6, f"{int(b.get_height())}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
axC.set_xticks(range(len(TARGET_ORDER)))
axC.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=9.5)
axC.set_ylabel("Number of samples", fontsize=10)
axC.set_ylim(0, dsum["n_samples"].max() * 1.18)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.set_title("C", fontsize=14, fontweight="bold", loc="left")

# ============ Panel D: composting period range per target ============
axD = fig.add_subplot(gs[1, 1])
period_data = []
for name in TARGET_ORDER:
    fpath, ycol, code = TARGETS[name]
    df = pd.read_csv(fpath)
    period_data.append(df["Period (d)"].dropna().values)
bp = axD.boxplot(period_data, positions=range(len(TARGET_ORDER)), widths=0.55, patch_artist=True,
                  showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, name in zip(bp["boxes"], TARGET_ORDER):
    patch.set_facecolor(TARGET_COLORS[name])
    patch.set_alpha(0.75)
    patch.set_edgecolor("black")
for med in bp["medians"]:
    med.set_color("black")
axD.set_xticks(range(len(TARGET_ORDER)))
axD.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=9.5)
axD.set_ylabel("Composting period (d)", fontsize=10)
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.set_title("D", fontsize=14, fontweight="bold", loc="left")

fig.suptitle("Pooled management-parameter landscape across the 7-target compost database "
              f"(n = {len(pooled)} unique parameter combinations)", fontsize=11, y=1.005)

polish_figure(fig, boxed_axes=(axA,), text_scale=1.08)
save_combined_figure(fig, OUT_DIR, "Fig1_dataset_structure")


def draw_panel_a(panel_fig):
    ax = panel_fig.add_subplot(111)
    for i in range(n):
        for j in range(n):
            if i > j:
                r = corr.iloc[i, j]
                e = Ellipse((j, i), width=0.9, height=0.9 * (1 - abs(r) * 0.7),
                            angle=45 if r > 0 else -45,
                            facecolor=cmap(norm(r)), edgecolor="black", lw=0.4, alpha=0.92)
                ax.add_patch(e)
            elif i < j:
                r = corr.iloc[i, j]
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=8.5,
                        color="black" if abs(r) < 0.6 else "white")
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#f2f2f2", edgecolor="none"))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    sm_local = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_local.set_array([])
    cax = ax.inset_axes([0.10, 1.055, 0.80, 0.045])
    cb = panel_fig.colorbar(sm_local, cax=cax, orientation="horizontal")
    cb.set_label("Spearman's $\\rho$", fontsize=10.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=9.8, length=3.6, width=1.0)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    for pos, feat in zip(positions, CONTINUOUS):
        for grp, side in zip([1, 0], [1, -1]):
            vals = z.loc[z["enclosed"] == grp, feat].dropna().values
            if len(vals) < 3 or np.std(vals) == 0:
                continue
            kde = gaussian_kde(vals)
            ys = np.linspace(-0.05, 1.05, 200)
            dens = kde(ys)
            dens = dens / dens.max() * width
            ax.fill_betweenx(ys, pos, pos + side * dens, color=colors[grp], alpha=0.75,
                             edgecolor="black", lw=0.5,
                             label=group_labels[grp] if pos == 0 else None)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Within-feature percentile rank", fontsize=11)
    ax.set_xlim(-0.7, n - 0.3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.56, 1.08),
              fontsize=10.8, ncol=2, handlelength=1.2, columnspacing=1.1)
    return ax


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111)
    bars = ax.bar(TARGET_ORDER, dsum["n_samples"], color=[TARGET_COLORS[t] for t in TARGET_ORDER],
                  edgecolor="black", linewidth=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10.5)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_ylim(0, dsum["n_samples"].max() * 1.18)
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    bp = ax.boxplot(period_data, positions=range(len(TARGET_ORDER)), widths=0.55,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, name in zip(bp["boxes"], TARGET_ORDER):
        patch.set_facecolor(TARGET_COLORS[name])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10.5)
    ax.set_ylabel("Composting period (d)", fontsize=11)
    return ax


save_independent_subfigures(OUT_DIR, "Fig1_dataset_structure", [
    ("a", (6.4, 5.8), draw_panel_a, None),
    ("b", (6.6, 5.8), draw_panel_b, None),
    ("c", (5.8, 5.0), draw_panel_c, None),
    ("d", (5.8, 5.0), draw_panel_d, None),
])
print("Saved Fig2.")
