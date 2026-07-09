"""Fig. 9. TOPSIS robustness, applicability domain and objective contrast.

This script draws the consolidated figure from the already-exported Fig. 9a/9b
analysis tables so it does not depend on pickled model objects.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import CMAP_DIVERGE, CMAP_SEQ, TARGET_LABELS, TARGET_ORDER
from plot_style import configure_publication_style, polish_figure, save_figure_with_subfigures


OUT_DIR = os.path.dirname(__file__)
STEM = "Fig8_decision_robustness"
SOURCE_A = os.path.join(os.path.dirname(__file__), "..", "Fig9a_weight_sensitivity_applicability")
SOURCE_B = os.path.join(os.path.dirname(__file__), "..", "Fig9b_single_objective_comparison")
SOURCE_C = os.path.join(os.path.dirname(__file__), "..", "Fig6_multiobjective_topsis")
configure_publication_style()


def simplex_xy(weights):
    weights = np.asarray(weights, dtype=float)
    c = weights[:, 1]
    q = weights[:, 2]
    return c + 0.5 * q, np.sqrt(3) * 0.5 * q


def rank_class(percentile):
    if percentile <= 1:
        return "Top 1%"
    if percentile <= 5:
        return "1-5%"
    if percentile <= 10:
        return "5-10%"
    return ">10%"


def annotate_heatmap(ax, data, fmt="{:.2f}", threshold=0.55):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=10.0, color=color)


def format_heatmap_axis(ax, rows, cols, *, xtick_size=11.2, ytick_size=11.6):
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=34, ha="right", fontsize=xtick_size)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=ytick_size)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


weight_eval = pd.read_csv(os.path.join(SOURCE_A, "Table_Fig9_weight_sensitivity_scenarios.csv"))
if "Rank class" not in weight_eval:
    weight_eval["Rank class"] = weight_eval["Balanced_recipe_rank_percentile"].map(rank_class)
grid_eval = weight_eval[weight_eval["Set"] == "Full simplex"].copy()
moderate_eval = weight_eval[weight_eval["Set"] == "Moderate perturbation"].copy()
ad_summary = pd.read_csv(os.path.join(SOURCE_A, "Table_Fig9_applicability_domain_summary.csv"))
lever_gap = pd.read_csv(os.path.join(SOURCE_A, "Table_Fig9_nearest_neighbor_lever_gap.csv"))
score_table = pd.read_csv(os.path.join(SOURCE_B, "Table_Fig9_objective_strategy_scores.csv"), index_col=0)
change_table = pd.read_csv(os.path.join(SOURCE_B, "Table_Fig9_objective_strategy_favourable_changes.csv"), index_col=0)
conformal_table = pd.read_csv(os.path.join(SOURCE_C, "Table_conformal_uncertainty_baseline_vs_topsis.csv"))

strategy_labels = {
    "Typical practice": "Typical",
    "N-only optimum": "N-only",
    "C/GHG-only optimum": "C/GHG-only",
    "GI-only optimum": "GI-only",
    "TOPSIS balanced optimum": "TOPSIS",
}
strategy_short = [strategy_labels.get(idx, idx) for idx in score_table.index]
objective_labels = ["N", "C/GHG", "GI", "TOPSIS"]
target_labels = [TARGET_LABELS.get(t, t) for t in TARGET_ORDER]

fig = plt.figure(figsize=(17.6, 23.8), dpi=300)
gs = fig.add_gridspec(
    4,
    2,
    width_ratios=[1.02, 1.18],
    height_ratios=[1.05, 1.02, 1.08, 1.04],
    wspace=0.42,
    hspace=0.30,
)

ax_a = fig.add_subplot(gs[0, 0])
x, y = simplex_xy(grid_eval[["N_weight", "C_weight", "Q_weight"]].values)
sc = ax_a.scatter(
    x,
    y,
    c=grid_eval["Balanced_recipe_rank_percentile"],
    s=17,
    cmap=CMAP_SEQ,
    edgecolors="none",
    alpha=0.94,
)
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
ax_a.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
base = np.array([[1 / 3, 1 / 3, 1 / 3]])
base_x, base_y = simplex_xy(base)
ax_a.scatter(base_x, base_y, marker="*", s=240, color="#E2B36E", edgecolors="black", linewidths=1.0, zorder=5)
ax_a.text(-0.04, -0.035, "N", ha="right", va="top", fontsize=14.5, fontweight="bold")
ax_a.text(1.04, -0.035, "C/GHG", ha="left", va="top", fontsize=14.5, fontweight="bold")
ax_a.text(0.5, np.sqrt(3) / 2 + 0.035, "GI", ha="center", va="bottom", fontsize=14.5, fontweight="bold")
ax_a.set_aspect("equal")
ax_a.set_xlim(-0.08, 1.08)
ax_a.set_ylim(-0.07, np.sqrt(3) / 2 + 0.10)
ax_a.set_axis_off()
cax_a = ax_a.inset_axes([0.18, -0.10, 0.66, 0.045])
cbar_a = fig.colorbar(sc, cax=cax_a, orientation="horizontal")
cbar_a.set_label("Rank percentile of balanced recipe", fontsize=12.8)
cbar_a.ax.tick_params(labelsize=12.0)

ax_b = fig.add_subplot(gs[0, 1])
class_order = ["Top 1%", "1-5%", "5-10%", ">10%"]
colors = ["#83AE8F", "#B8D3B1", "#F6D58B", "#C97979"]
bar_labels = ["Moderate perturbation", "Full simplex"]
left = np.zeros(len(bar_labels))
for cls, color in zip(class_order, colors):
    vals = []
    for label in bar_labels:
        sub = weight_eval[weight_eval["Set"] == label]
        vals.append((sub["Rank class"] == cls).mean() * 100)
    ax_b.barh(bar_labels, vals, left=left, color=color, edgecolor="white", height=0.56, label=cls)
    for i, val in enumerate(vals):
        if val >= 6:
            ax_b.text(left[i] + val / 2, i, f"{val:.0f}%", ha="center", va="center", fontsize=12.0)
    left += np.asarray(vals)
ax_b.set_xlim(0, 100)
ax_b.set_xlabel("Weight scenarios (%)")
ax_b.set_ylabel("")
ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False, fontsize=12.2)

ax_c = fig.add_subplot(gs[1, 0])
metric_labels = {
    "Typical-practice nearest-neighbour distance": "Typical",
    "TOPSIS optimum nearest-neighbour distance": "TOPSIS",
    "Median top-5%-TOPSIS nearest-neighbour distance": "Top 5% median",
    "Median sampled-recipe nearest-neighbour distance": "All median",
    "95th percentile sampled-recipe nearest-neighbour distance": "All 95th",
}
ad_plot = ad_summary[ad_summary["Metric"].isin(metric_labels)].copy()
ad_plot["Label"] = ad_plot["Metric"].map(metric_labels)
order = ["Typical", "TOPSIS", "Top 5% median", "All median", "All 95th"]
ad_plot = ad_plot.set_index("Label").loc[order].reset_index()
ad_colors = ["#4E5358", "#C97979", "#83AE8F", "#B7B7B7", "#B7B7B7"]
ax_c.barh(ad_plot["Label"], ad_plot["Value"], color=ad_colors, edgecolor="white", height=0.58)
for y_pos, val in enumerate(ad_plot["Value"]):
    ax_c.text(val + 0.025, y_pos, f"{val:.2f}", va="center", ha="left", fontsize=12.0)
ax_c.set_xlabel("Nearest-observed distance")
ax_c.set_xlim(0, max(ad_plot["Value"]) * 1.18)
ax_c.invert_yaxis()

ax_d = fig.add_subplot(gs[1, 1])
gap_plot = lever_gap.head(8).iloc[::-1].copy()
ax_d.barh(
    gap_plot["Lever label"],
    gap_plot["Absolute normalized gap"],
    color="#777FBC",
    edgecolor="white",
    height=0.62,
)
for y_pos, val in enumerate(gap_plot["Absolute normalized gap"]):
    ax_d.text(val + 0.01, y_pos, f"{val:.2f}", va="center", ha="left", fontsize=12.0)
ax_d.axvline(0.25, color="#C97979", linestyle="--", lw=1.4)
ax_d.set_xlabel("Absolute normalized gap")
ax_d.set_xlim(0, max(0.35, gap_plot["Absolute normalized gap"].max() * 1.22))

ax_e = fig.add_subplot(gs[2, 0])
obj_vals = score_table.values.astype(float)
im_e = ax_e.imshow(obj_vals, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
annotate_heatmap(ax_e, obj_vals)
format_heatmap_axis(ax_e, strategy_short, objective_labels, xtick_size=11.4, ytick_size=11.6)
cbar_e = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02)
cbar_e.set_label("Score", fontsize=12.2)
cbar_e.ax.tick_params(labelsize=11.0)

ax_f = fig.add_subplot(gs[2, 1])
change_vals = change_table[TARGET_ORDER].values.astype(float)
max_abs = max(1.0, float(np.nanmax(np.abs(change_vals))))
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
im_f = ax_f.imshow(change_vals, cmap=CMAP_DIVERGE, norm=norm, aspect="auto")
for i in range(change_vals.shape[0]):
    for j in range(change_vals.shape[1]):
        ax_f.text(j, i, f"{change_vals[i, j]:+.1f}", ha="center", va="center", fontsize=9.6, color="black")
format_heatmap_axis(ax_f, strategy_short, target_labels, xtick_size=10.6, ytick_size=11.2)
cbar_f = fig.colorbar(im_f, ax=ax_f, fraction=0.034, pad=0.018)
cbar_f.set_label("Favourable change vs typical", fontsize=11.8)
cbar_f.ax.tick_params(labelsize=10.8)

endpoints = conformal_table["Endpoint"].tolist()
conf_labels = [TARGET_LABELS.get(t, t) for t in endpoints]
typical = conformal_table["Typical practice point"].astype(float).to_numpy()
topsis = conformal_table["TOPSIS point"].astype(float).to_numpy()
half_width = conformal_table["Split-conformal half-width (95%)"].astype(float).to_numpy()
coverage = conformal_table["Empirical held-out coverage"].astype(float).to_numpy()
favourable = conformal_table["Point-estimate direction favourable"].astype(str).str.lower().eq("yes").to_numpy()

ax_g = fig.add_subplot(gs[3, 0])
y = np.arange(len(endpoints))
ax_g.errorbar(
    typical,
    y + 0.13,
    xerr=half_width,
    fmt="o",
    color="#777FBC",
    ecolor="#99BADF",
    elinewidth=1.6,
    capsize=3,
    label="Typical",
)
ax_g.errorbar(
    topsis,
    y - 0.13,
    xerr=half_width,
    fmt="o",
    color="#C97979",
    ecolor="#DF9E9B",
    elinewidth=1.6,
    capsize=3,
    label="TOPSIS",
)
ax_g.set_yticks(y)
ax_g.set_yticklabels(conf_labels)
ax_g.invert_yaxis()
ax_g.set_xlabel("Predicted endpoint value with 95% residual PI")
ax_g.legend(loc="upper right", frameon=False, fontsize=12.2)

ax_h = fig.add_subplot(gs[3, 1])
coverage_colors = ["#83AE8F" if ok else "#C97979" for ok in favourable]
ax_h.barh(conf_labels, coverage * 100, color=coverage_colors, edgecolor="white", height=0.58)
ax_h.axvline(95, color="black", linestyle="--", lw=1.2)
for ypos, val in enumerate(coverage * 100):
    if val < 95:
        ax_h.text(val - 0.45, ypos, f"{val:.1f}%", va="center", ha="right", fontsize=11.8)
    else:
        ax_h.text(val + 0.5, ypos, f"{val:.1f}%", va="center", ha="left", fontsize=11.8)
ax_h.set_xlabel("Held-out empirical coverage (%)")
ax_h.set_xlim(88, 101)
ax_h.invert_yaxis()

polish_figure(fig, text_scale=1.08)
for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]:
    ax.xaxis.label.set_fontsize(15.0)
    ax.yaxis.label.set_fontsize(15.0)
    ax.tick_params(labelsize=12.4)
for text in ax_e.texts:
    text.set_fontsize(10.2)
for text in ax_f.texts:
    text.set_fontsize(9.6)

save_figure_with_subfigures(
    fig,
    OUT_DIR,
    STEM,
    [[ax_a, cbar_a.ax], ax_b, ax_c, ax_d, [ax_e, cbar_e.ax], [ax_f, cbar_f.ax], ax_g, ax_h],
)
plt.close(fig)

moderate_top1 = (moderate_eval["Balanced_recipe_rank_percentile"] <= 1).mean() * 100
full_top5 = (grid_eval["Balanced_recipe_rank_percentile"] <= 5).mean() * 100
topsis_dist = float(ad_summary.loc[ad_summary["Metric"] == "TOPSIS optimum nearest-neighbour distance", "Value"].iloc[0])
print("Saved", STEM)
print(f"Moderate perturbation top-1% share: {moderate_top1:.1f}%")
print(f"Full-simplex top-5% share: {full_top5:.1f}%")
print(f"TOPSIS nearest-observed distance: {topsis_dist:.3f}")
