"""
Figure 4. Cross-target SHAP importance of the 20 shared management/substrate
parameters, plus a worked SHAP example for TN loss.
Panel A: grouped heatmap of normalized SHAP importance (%) across all 7 targets.
Panel B: SHAP beeswarm for TN loss (full feature set).
Panel C: category-level importance averaged across the 7 targets.

Style adapted from: 期刊配图：模型解释SHAP蜂群图+重要性柱状图和类别贡献饼图组合展示 (#16),
期刊配图：用变量热图解读模型预测 (#67/#208).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import (TARGET_ORDER, TARGET_LABELS, COMMON_FEATURES, FEATURE_LABELS,
                    FEATURE_GROUPS, FEATURE_GROUP_ORDER, FEATURE_GROUP_COLORS,
                    RESULTS_DIR, TARGETS, CMAP_SEQ, TARGET_COLORS)
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

INK = "#343A40"
SOFT_INK = "#49515A"
LINE_INK = "#6E7781"
SHAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "NatureNew_SHAP", ["#6F8FB8", "#F6F4EE", "#C97979"], N=256
)


def soften_figure_text(fig):
    """Keep Fig. 3 readable without the overly heavy all-bold text layer."""
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.05)
            spine.set_color(LINE_INK)
        ax.tick_params(colors=SOFT_INK, width=1.05, length=5.2)
        for item in [ax.xaxis.label, ax.yaxis.label]:
            item.set_color(INK)
            item.set_fontweight("semibold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(INK)
            label.set_fontweight("semibold")
        for text in ax.texts:
            if text.get_color() not in ("white", "#ffffff", "#FFFFFF"):
                text.set_color(INK)
            text.set_fontweight("normal")
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_color(INK)
                item.set_fontweight("normal")
            if legend.get_title() is not None:
                legend.get_title().set_color(INK)
                legend.get_title().set_fontweight("normal")
    for legend in fig.legends:
        for item in legend.get_texts():
            item.set_color(INK)
            item.set_fontweight("normal")
        if legend.get_title() is not None:
            legend.get_title().set_color(INK)
            legend.get_title().set_fontweight("normal")


def save_softened_independent_subfigures(out_dir, stem, panel_builders):
    sub_dir = os.path.join(out_dir, "subfigures")
    os.makedirs(sub_dir, exist_ok=True)
    for label, figsize, draw_func, boxed_axes in panel_builders:
        panel_fig = plt.figure(figsize=figsize, dpi=300)
        result = draw_func(panel_fig)
        if isinstance(result, dict):
            axes = result.get("axes", ())
            boxed_axes = result.get("boxed_axes", boxed_axes)
        if boxed_axes is None:
            boxed_axes = ()
        elif boxed_axes == "returned":
            boxed_axes = axes if isinstance(axes, (list, tuple)) else (axes,)
        polish_figure(panel_fig, boxed_axes=boxed_axes, text_scale=1.04)
        soften_figure_text(panel_fig)
        for ext, kwargs in (("png", {"bbox_inches": "tight", "dpi": 600}),
                            ("pdf", {"bbox_inches": "tight"})):
            path = os.path.join(sub_dir, f"{stem}_{label}.{ext}")
            try:
                panel_fig.savefig(path, **kwargs)
            except PermissionError:
                root, suffix = os.path.splitext(path)
                panel_fig.savefig(f"{root}_updated{suffix}", **kwargs)
        plt.close(panel_fig)
    with open(os.path.join(sub_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {stem} subfigures\n\n")
        f.write("Subfigures were independently rendered on standalone canvases, not cropped from the combined figure.\n")

SHORT_FEATURE_LABELS = {
    "Composting period (d)": "Period",
    "Additive 1 type": "Additive 1",
    "Additive 2 type": "Additive 2",
    "Additive 3 type": "Additive 3",
    "Additive 4 type": "Additive 4",
    "Initial C/N ratio": "C/N ratio",
    "Aeration rate (L/min/kg)": "Aer. rate",
    "Primary feedstock type": "Feedstock 1",
    "Bulking agent rate (%DW)": "Bulking rate",
    "Initial moisture (%)": "Moisture",
    "Composting method": "Method",
    "Initial TN (%)": "Initial TN",
    "Pile volume (m3)": "Pile volume",
    "Aeration onset day": "Aer. onset",
}

shapimp = pd.read_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"))
shapimp = shapimp[shapimp["Feature"].isin(COMMON_FEATURES)]
pivot = shapimp.pivot(index="Feature", columns="Target", values="Importance_pct").reindex(COMMON_FEATURES)[TARGET_ORDER]
pivot["__group"] = [FEATURE_GROUPS[f] for f in pivot.index]
pivot["__total"] = pivot[TARGET_ORDER].mean(axis=1)
pivot = pivot.sort_values(["__group", "__total"], ascending=[True, False])
groups = pivot.pop("__group")
totals = pivot.pop("__total")

fig = plt.figure(figsize=(17.4, 17.0), dpi=300)
outer = fig.add_gridspec(3, 1, height_ratios=[1.66, 1.03, 1.08], hspace=0.40)
gsA = outer[0].subgridspec(1, 2, width_ratios=[0.82, 0.18], wspace=0.08)
gsBC = outer[1].subgridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.34)

# ============ Panel A: grouped heatmap ============
axA = fig.add_subplot(gsA[0, 0])
axA_legend = fig.add_subplot(gsA[0, 1])
data = pivot[TARGET_ORDER].values
im = axA.imshow(data, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=np.nanpercentile(data, 97))

labels = [FEATURE_LABELS.get(f, f) for f in pivot.index]
axA.set_yticks(range(len(labels)))
axA.set_yticklabels(labels, fontsize=13.6)
axA.set_xticks(range(len(TARGET_ORDER)))
axA.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=13.6)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        axA.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10.4,
                 color="white" if v > np.nanpercentile(data, 70) else INK)

# group color bar on the left
unique_groups = [g for g in FEATURE_GROUP_ORDER if g in set(groups)]
group_colors = {g: FEATURE_GROUP_COLORS[g] for g in unique_groups}
for i, (feat, g) in enumerate(zip(pivot.index, groups)):
    axA.add_patch(plt.Rectangle((-1.7, i - 0.5), 0.5, 1, color=group_colors[g], clip_on=False))
# group legend
handles = [plt.Rectangle((0,0),1,1, color=group_colors[g]) for g in unique_groups]
axA_legend.axis("off")
axA_legend.legend(handles, unique_groups, loc="center left", bbox_to_anchor=(0.0, 0.50),
                  ncol=1, frameon=False, fontsize=13.4,
                  title="Feature category", title_fontsize=14.3,
                  handlelength=1.35, handleheight=0.85, labelspacing=0.82)

axA.set_xlim(-2.0, len(TARGET_ORDER) - 0.5)
dividerA = make_axes_locatable(axA)
caxA = dividerA.append_axes("top", size="5.2%", pad=0.34)
cbar = fig.colorbar(im, cax=caxA, orientation="horizontal")
cbar.set_label("SHAP importance (% of total)", fontsize=14.0)
cbar.ax.xaxis.set_label_position("top")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.tick_params(labelsize=12.5, length=3.8, width=1.05)

# ============ Panel B: SHAP beeswarm for TN loss ============
axB = fig.add_subplot(gsBC[0, 0])
d = np.load(os.path.join(RESULTS_DIR, "shap_TN.npz"), allow_pickle=True)
sv_full = d["shap_values_full"]
X_full = d["X_full"]
feat_full = [SHORT_FEATURE_LABELS.get(FEATURE_LABELS.get(f, f), FEATURE_LABELS.get(f, f)) for f in d["feat_names_full"]]
Xdf = pd.DataFrame(X_full, columns=feat_full)
plt.sca(axB)
axes_before_shap = set(fig.axes)
shap.summary_plot(sv_full, Xdf, show=False, plot_size=None, max_display=10,
                  color_bar=True, cmap=SHAP_CMAP)
shap_cbar_axes = [ax for ax in fig.axes if ax not in axes_before_shap]
axB.tick_params(labelsize=13.4)
axB.xaxis.label.set_fontsize(14.2)

# ============ Panel C: category-level importance (avg across targets) ============
axC = fig.add_subplot(gsBC[0, 1])
cat_imp = pivot[TARGET_ORDER].mean(axis=1).groupby(groups).sum()
cat_imp = cat_imp.sort_values(ascending=False)
colors = [group_colors[g] for g in cat_imp.index]
ypos_c = np.arange(len(cat_imp))[::-1]
bars = axC.barh(ypos_c, cat_imp.values, color=colors, edgecolor="white",
                linewidth=1.0, height=0.56, alpha=0.92)
axC.set_yticks(ypos_c)
axC.set_yticklabels(cat_imp.index, fontsize=13.5)
axC.tick_params(axis="y", length=0, pad=4)
axC.spines["left"].set_visible(False)
axC.set_xlabel("Mean SHAP share (%)", fontsize=14.2)
axC.set_xlim(0, cat_imp.max() * 1.22)
axC.grid(axis="x", color="#e6e6e6", linewidth=0.7)
axC.set_axisbelow(True)
for bar, value in zip(bars, cat_imp.values):
    axC.text(value + cat_imp.max() * 0.025, bar.get_y() + bar.get_height() / 2,
             f"{value:.1f}", va="center", ha="left", fontsize=12.5, color=INK)

# ============ Panel D: top universal levers, ranked by mean SHAP importance ============
axD = fig.add_subplot(outer[2])
top_n = 10
order_idx = totals.sort_values(ascending=False).index[:top_n]
top_pivot = pivot.loc[order_idx, TARGET_ORDER]
top_totals = totals.loc[order_idx]
top_groups = groups.loc[order_idx]
ypos = np.arange(top_n)[::-1]

bar_colors = [group_colors[g] for g in top_groups]
axD.barh(ypos, top_totals.values, color=bar_colors, edgecolor=LINE_INK, linewidth=0.45, height=0.55,
         alpha=0.55, label=None)
for y, feat in zip(ypos, order_idx):
    for target in TARGET_ORDER:
        axD.scatter(top_pivot.loc[feat, target], y, color=TARGET_COLORS[target], s=38,
                     edgecolors="white", linewidths=0.5, zorder=3,
                     label=TARGET_LABELS[target] if y == ypos[0] else None)
axD.set_yticks(ypos)
axD.set_yticklabels([FEATURE_LABELS.get(f, f) for f in order_idx], fontsize=14.0)
axD.set_xlabel("SHAP importance (% of total) -- bars: mean across endpoints; dots: per-endpoint value",
               fontsize=14.2, labelpad=8)
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.tick_params(axis="x", labelsize=13.4)
axD.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=7, frameon=False,
           fontsize=14.2, title="Endpoint", title_fontsize=15.0)

# Shift the complete SHAP panel left to balance its short y labels and open space before panel C.
b_pos = axB.get_position()
b_shift = 0.050
axB.set_position([b_pos.x0 - b_shift, b_pos.y0, b_pos.width, b_pos.height])
for cbar_ax in shap_cbar_axes:
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0 - b_shift, cbar_pos.y0,
                          cbar_pos.width, cbar_pos.height])

polish_figure(fig, boxed_axes=(axA,), text_scale=1.16)
soften_figure_text(fig)
save_combined_figure(fig, OUT_DIR, "Fig3_shap_importance")


def draw_panel_a(panel_fig):
    grid = panel_fig.add_gridspec(1, 2, width_ratios=[0.80, 0.20], wspace=0.08)
    ax = panel_fig.add_subplot(grid[0, 0])
    legend_ax = panel_fig.add_subplot(grid[0, 1])
    im_local = ax.imshow(data, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=np.nanpercentile(data, 97))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > np.nanpercentile(data, 70) else INK)
    for i, (_, g) in enumerate(zip(pivot.index, groups)):
        ax.add_patch(plt.Rectangle((-1.7, i - 0.5), 0.5, 1, color=group_colors[g], clip_on=False))
    handles_local = [plt.Rectangle((0, 0), 1, 1, color=group_colors[g]) for g in unique_groups]
    legend_ax.axis("off")
    legend_ax.legend(handles_local, unique_groups, loc="center left",
                     ncol=1, frameon=False, fontsize=9.2,
                     title="Feature category", title_fontsize=9.8,
                     handlelength=1.2, labelspacing=0.68)
    ax.set_xlim(-2.0, len(TARGET_ORDER) - 0.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5.2%", pad=0.28)
    cb = panel_fig.colorbar(im_local, cax=cax, orientation="horizontal")
    cb.set_label("SHAP importance (% of total)", fontsize=10.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=9.5, length=3.4, width=1.0)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    plt.sca(ax)
    axes_before = set(panel_fig.axes)
    shap.summary_plot(sv_full, Xdf, show=False, plot_size=None, max_display=10,
                      color_bar=True, cmap=SHAP_CMAP)
    cbar_axes = [a for a in panel_fig.axes if a not in axes_before]
    ax.tick_params(labelsize=9.5)
    return [ax] + cbar_axes


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111)
    bars = ax.barh(ypos_c, cat_imp.values, color=colors, edgecolor="white",
                   linewidth=1.0, height=0.56, alpha=0.92)
    ax.set_yticks(ypos_c)
    ax.set_yticklabels(cat_imp.index, fontsize=10.0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean SHAP share (%)", fontsize=11)
    ax.set_xlim(0, cat_imp.max() * 1.22)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, cat_imp.values):
        ax.text(value + cat_imp.max() * 0.025, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center", ha="left", fontsize=10.5, color=INK)
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.barh(ypos, top_totals.values, color=bar_colors, edgecolor=LINE_INK, linewidth=0.45,
            height=0.55, alpha=0.55)
    for y, feat in zip(ypos, order_idx):
        for target in TARGET_ORDER:
            ax.scatter(top_pivot.loc[feat, target], y, color=TARGET_COLORS[target], s=38,
                       edgecolors="white", linewidths=0.5, zorder=3,
                       label=TARGET_LABELS[target] if y == ypos[0] else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in order_idx], fontsize=12.2)
    ax.set_xlabel("SHAP importance (% of total) -- bars: mean across endpoints; dots: per-endpoint value",
                  fontsize=12.2, labelpad=14)
    ax.tick_params(axis="x", labelsize=11.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=7, frameon=False,
              fontsize=12.0, title="Endpoint", title_fontsize=12.6)
    return ax


save_softened_independent_subfigures(OUT_DIR, "Fig3_shap_importance", [
    ("a", (7.2, 8.3), draw_panel_a, None),
    ("b", (6.2, 6.0), draw_panel_b, None),
    ("c", (5.8, 4.8), draw_panel_c, None),
    ("d", (10.8, 6.0), draw_panel_d, None),
])
print("Saved Fig4.")

pivot[TARGET_ORDER].round(2).to_csv(os.path.join(OUT_DIR, "Table_shap_importance_pct.csv"))
