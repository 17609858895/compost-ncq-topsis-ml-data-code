"""
Figure 3. Cross-target comparison of 8 machine-learning algorithms.
Panel A: radar chart of test-set R2 (8 algorithm axes, one polygon per target).
Panel B: heatmap of test-set R2 with the best algorithm per target marked.

Style adapted from: Bioresource Technology radar chart (#288), 期刊文章配图：
基于雷达图的多机器学习模型表现评估对比 (#185).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import TARGET_ORDER, TARGET_LABELS, TARGET_COLORS, RESULTS_DIR, CMAP_SEQ
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

perf = pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv"))
MODELS = ["RandomForest","ExtraTrees","GradientBoosting","HistGradientBoosting","Ridge","ElasticNet","MLP","SVR"]
MODEL_LABELS = {"RandomForest":"RF","ExtraTrees":"ET","GradientBoosting":"GBR","HistGradientBoosting":"HGB",
                "Ridge":"Ridge","ElasticNet":"ENet","MLP":"MLP","SVR":"SVR"}

pivot = perf.pivot(index="Model", columns="Target", values="R2_test").reindex(MODELS)[TARGET_ORDER]
pivot_plot = pivot.clip(lower=0)  # radius cannot be negative

fig = plt.figure(figsize=(14.4, 11.8), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.16], height_ratios=[1, 1], wspace=0.28, hspace=0.40)

# ============ Panel A: radar chart ============
axA = fig.add_subplot(gs[0, 0], polar=True)
n = len(MODELS)
angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
angles += angles[:1]

for target in TARGET_ORDER:
    vals = pivot_plot[target].tolist()
    vals += vals[:1]
    axA.plot(angles, vals, color=TARGET_COLORS[target], linewidth=1.8, label=TARGET_LABELS[target], alpha=0.9)
    axA.fill(angles, vals, color=TARGET_COLORS[target], alpha=0.06)

axA.set_xticks(angles[:-1])
axA.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11, fontweight="bold")
axA.set_ylim(0, 1)
axA.set_yticks([0.2,0.4,0.6,0.8,1.0])
axA.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="grey")
axA.set_rlabel_position(20)
axA.spines['polar'].set_color('grey')
axA.grid(color="grey", alpha=0.3)
axA.set_title("A   Test-set $R^2$ by algorithm", fontsize=13, fontweight="bold", loc="left", pad=20)
axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9,
           title="Target", title_fontsize=10, ncol=4)

# ============ Panel B: heatmap with best-model markers ============
axB = fig.add_subplot(gs[0, 1])
data = pivot.values
im = axB.imshow(data, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")

for i in range(len(MODELS)):
    for j in range(len(TARGET_ORDER)):
        val = data[i, j]
        is_best = (val == pivot[TARGET_ORDER[j]].max())
        txt_color = "white" if val > 0.55 else "black"
        weight = "bold" if is_best else "normal"
        marker = " *" if is_best else ""
        axB.text(j, i, f"{val:.2f}{marker}", ha="center", va="center", fontsize=9,
                 color=txt_color, fontweight=weight)

axB.set_xticks(range(len(TARGET_ORDER)))
axB.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=9.5)
axB.set_yticks(range(len(MODELS)))
axB.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
caxB = axB.inset_axes([0.16, 1.055, 0.68, 0.045])
cbar = fig.colorbar(im, cax=caxB, orientation="horizontal")
cbar.set_label("Test-set $R^2$", fontsize=10)
cbar.ax.xaxis.set_label_position("top")
cbar.ax.tick_params(labelsize=9.5, length=3.6, width=1.0)
axB.set_title("B   Algorithm $\\times$ target performance matrix ($\\star$ = best model)",
               fontsize=13, fontweight="bold", loc="left", pad=12)

# ============ Panel C: train vs. test R2 (overfitting check) ============
axCC = fig.add_subplot(gs[1, 0])
for target in TARGET_ORDER:
    sub = perf[perf["Target"] == target]
    axCC.scatter(sub["R2_train"], sub["R2_test"], color=TARGET_COLORS[target], s=46,
                  alpha=0.85, edgecolors="black", linewidths=0.4, label=TARGET_LABELS[target])
lims = [-0.05, 1.02]
axCC.plot(lims, lims, color="grey", lw=1, ls="--", alpha=0.7, label="1:1 line")
axCC.set_xlim(lims); axCC.set_ylim(lims)
axCC.set_xlabel("Training-set $R^2$", fontsize=10.5)
axCC.set_ylabel("Test-set $R^2$", fontsize=10.5)
axCC.spines["top"].set_visible(False)
axCC.spines["right"].set_visible(False)
axCC.set_title("C   Generalization across all model $\\times$ endpoint combinations\n"
                "(points near the diagonal indicate limited overfitting)",
                fontsize=12, fontweight="bold", loc="left")
axCC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9, ncol=4)

# ============ Panel D: best-model test R2 vs. 3-fold CV R2 per endpoint ============
axDD = fig.add_subplot(gs[1, 1])
best_rows = perf.loc[perf.groupby("Target")["R2_test"].idxmax()].set_index("Target").reindex(TARGET_ORDER)
x = np.arange(len(TARGET_ORDER))
w = 0.38
axDD.bar(x - w/2, best_rows["CV_R2"], width=w, color="#99BADF", edgecolor="black",
         linewidth=0.6, label="3-fold CV $R^2$ (training data)")
axDD.bar(x + w/2, best_rows["R2_test"], width=w, color="#DF9E9B",
         edgecolor="black", linewidth=0.6, label="Held-out test $R^2$")
for i, (cv, te, m) in enumerate(zip(best_rows["CV_R2"], best_rows["R2_test"], best_rows["Model"])):
    axDD.text(i, max(cv, te) + 0.03, MODEL_LABELS[m], ha="center", fontsize=8, fontweight="bold")
axDD.set_xticks(x)
axDD.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=9.5)
axDD.set_ylabel("$R^2$", fontsize=10.5)
axDD.set_ylim(0, 1.05)
axDD.axhline(0, color="black", lw=0.6)
axDD.spines["top"].set_visible(False)
axDD.spines["right"].set_visible(False)
axDD.legend(loc="upper right", frameon=False, fontsize=9)
axDD.set_title("D   Best model per endpoint: cross-validation vs. held-out performance",
                fontsize=12, fontweight="bold", loc="left")

polish_figure(fig, boxed_axes=(axB,))
save_combined_figure(fig, OUT_DIR, "Fig2_model_performance")


def draw_panel_a(panel_fig):
    ax = panel_fig.add_subplot(111, polar=True)
    for target in TARGET_ORDER:
        vals = pivot_plot[target].tolist() + pivot_plot[target].tolist()[:1]
        ax.plot(angles, vals, color=TARGET_COLORS[target], linewidth=1.8,
                label=TARGET_LABELS[target], alpha=0.9)
        ax.fill(angles, vals, color=TARGET_COLORS[target], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.set_rlabel_position(20)
    ax.spines["polar"].set_color("grey")
    ax.grid(color="grey", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False,
              fontsize=9.5, title="Target", title_fontsize=10, ncol=4)
    return ax


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    im_local = ax.imshow(data, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(TARGET_ORDER)):
            val = data[i, j]
            is_best = (val == pivot[TARGET_ORDER[j]].max())
            ax.text(j, i, f"{val:.2f}{' *' if is_best else ''}", ha="center", va="center",
                    fontsize=9.5, color="white" if val > 0.55 else "black",
                    fontweight="bold" if is_best else "normal")
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10.5)
    cax = ax.inset_axes([0.16, 1.055, 0.68, 0.045])
    cb = panel_fig.colorbar(im_local, cax=cax, orientation="horizontal")
    cb.set_label("Test-set $R^2$", fontsize=10.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=9.8, length=3.6, width=1.0)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111)
    for target in TARGET_ORDER:
        sub = perf[perf["Target"] == target]
        ax.scatter(sub["R2_train"], sub["R2_test"], color=TARGET_COLORS[target], s=46,
                   alpha=0.85, edgecolors="black", linewidths=0.4, label=TARGET_LABELS[target])
    ax.plot(lims, lims, color="grey", lw=1, ls="--", alpha=0.7, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Training-set $R^2$", fontsize=11)
    ax.set_ylabel("Test-set $R^2$", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9.5, ncol=4)
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.bar(x - w / 2, best_rows["CV_R2"], width=w, color="#99BADF", edgecolor="black",
           linewidth=0.6, label="3-fold CV $R^2$ (training data)")
    ax.bar(x + w / 2, best_rows["R2_test"], width=w, color="#DF9E9B",
           edgecolor="black", linewidth=0.6, label="Held-out test $R^2$")
    for i, (cv, te, m) in enumerate(zip(best_rows["CV_R2"], best_rows["R2_test"], best_rows["Model"])):
        ax.text(i, max(cv, te) + 0.03, MODEL_LABELS[m], ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    return ax


save_independent_subfigures(OUT_DIR, "Fig2_model_performance", [
    ("a", (6.0, 5.6), draw_panel_a, None),
    ("b", (6.6, 5.3), draw_panel_b, None),
    ("c", (5.8, 5.1), draw_panel_c, None),
    ("d", (6.2, 5.1), draw_panel_d, None),
])
print("Saved Fig3.")

# Also export the performance table for the manuscript
pivot.round(3).to_csv(os.path.join(OUT_DIR, "Table_model_performance_R2.csv"))
