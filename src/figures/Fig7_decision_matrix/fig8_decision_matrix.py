"""
Figure 8. Cross-endpoint decision matrix: how each of the 20 shared
management/substrate parameters relates to all 7 nutrient/carbon/quality
endpoints, and which levers behave as "universal wins" vs. "trade-off"
parameters that require the multi-objective balancing in Fig. 7.

Panel A: feature-category level association strength across endpoints
         (mean absolute Spearman correlation within each category).
Panel B: desirability-adjusted version of the same matrix (sign-flipped so
         that positive = "this management change moves the endpoint toward
         its desirable side"); rows that are uniformly one colour are
         "universal levers".
Panel C: universal-consensus ranking -- mean desirability-adjusted effect
         across the 7 endpoints (significant correlations only).
Panel D: cross-endpoint conflict ranking -- spread (max-min) of the
         desirability-adjusted effect across endpoints, i.e. how strongly a
         lever creates trade-offs that require multi-objective balancing.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import (TARGET_ORDER, TARGET_DESIRABLE_DIRECTION, COMMON_FEATURES,
                     FEATURE_LABELS, FEATURE_GROUPS, FEATURE_GROUP_ORDER,
                     FEATURE_GROUP_COLORS, RESULTS_DIR, CMAP_DIVERGE,
                     CMAP_SEQ, TARGET_LABELS)
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

TOP_LEGEND_FONT_SIZE = 16.0
TOP_LEGEND_TITLE_SIZE = 17.0
STANDALONE_TOP_LEGEND_FONT_SIZE = 11.8
STANDALONE_TOP_LEGEND_TITLE_SIZE = 12.6

corr = pd.read_csv(os.path.join(RESULTS_DIR, "feature_target_corr.csv"))
mat = corr.pivot(index="Feature", columns="Target", values="rho").reindex(COMMON_FEATURES)[TARGET_ORDER]
matp = corr.pivot(index="Feature", columns="Target", values="p").reindex(COMMON_FEATURES)[TARGET_ORDER]

mat_adj = mat.copy()
for t in TARGET_ORDER:
    mat_adj[t] = mat_adj[t] * TARGET_DESIRABLE_DIRECTION[t]

sig = matp < 0.05
mat_adj_sig = mat_adj.where(sig)

# Ordering: group by FEATURE_GROUPS, then by mean |rho| within group (descending)
groups = pd.Series({f: FEATURE_GROUPS[f] for f in COMMON_FEATURES})
mean_abs = mat.abs().mean(axis=1)
order_df = pd.DataFrame({"group": groups, "mean_abs": mean_abs})
order_df = order_df.sort_values(["group", "mean_abs"], ascending=[True, False])
ORDER = order_df.index.tolist()
unique_groups = [g for g in FEATURE_GROUP_ORDER if g in set(groups.loc[ORDER])]
group_colors = {g: FEATURE_GROUP_COLORS[g] for g in unique_groups}

mat = mat.loc[ORDER]
matp = matp.loc[ORDER]
mat_adj = mat_adj.loc[ORDER]
mat_adj_sig = mat_adj_sig.loc[ORDER]
n = len(ORDER)
labels = [FEATURE_LABELS.get(f, f) for f in ORDER]
short_labels = [
    "Additive 1", "Additive 4", "Additive 3", "Additive 2",
    "Aer. type", "Aer. onset", "Aer. rate", "Aer. interval", "Aer. duration",
    "Feedstock 1", "Feedstock 2", "Feedstock 3",
    "C/N ratio", "Moisture", "Turning", "Period",
    "Bulking rate", "Enclosed", "Method", "Pile volume",
]
rank_labels = {
    "Application Rate (%DW)": "Bulking rate",
    "Additive 1 type": "Additive 1",
    "Additive 2 type": "Additive 2",
    "Additive 3 type": "Additive 3",
    "Additive 4 type": "Additive 4",
    "V1_Ventilation Type": "Aeration type",
    "V4_Ventilation Day": "Aeration onset",
    "V5_Ventilation rate (L/min/kg iniDW)": "Aeration rate",
    "Aeration interval (min)": "Aeration interval",
    "Aeration duration (min)": "Aeration duration",
    "Initial C/N (%)": "Initial C/N",
    "M1_is Enclosed": "Enclosed system",
    "Period (d)": "Composting period",
    "Turning times": "Turning frequency",
    "Initial Moisture Content (%)": "Initial moisture",
    "Primary feedstock type": "Primary feedstock",
    "Secondary feedstock type": "Secondary feedstock",
    "Tertiary feedstock type": "Tertiary feedstock",
}

fig = plt.figure(figsize=(17.6, 17.2), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.18], width_ratios=[1, 1],
                      wspace=0.56, hspace=0.36)
fig.subplots_adjust(top=0.962)


def grouped_heatmap(ax, data, pvals, title, cbar_label, annotate_stars=True, show_ylabels=True):
    im = ax.imshow(data.values, cmap=CMAP_DIVERGE, vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(len(TARGET_ORDER)):
            r = data.iloc[i, j]
            if pd.isna(r):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee", edgecolor="none"))
                continue
            if annotate_stars:
                p = pvals.iloc[i, j]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                txt = f"{r:.2f}{star}"
            else:
                txt = f"{r:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                    color="white" if abs(r) > 0.55 else "black")
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10.5)
    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels(labels, fontsize=9.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    # group colour sidebar
    for i, f in enumerate(ORDER):
        ax.add_patch(plt.Rectangle((-1.9, i - 0.5), 0.55, 1, color=group_colors[groups[f]], clip_on=False))
    ax.set_xlim(-2.2, len(TARGET_ORDER) - 0.5)
    cax = ax.inset_axes([0.14, 1.045, 0.72, 0.04])
    cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=12.2)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=11.5, length=3.8, width=1.1)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    return im, cbar


def category_strength_heatmap(ax):
    group_counts = groups.loc[ORDER].value_counts().reindex(unique_groups)
    group_strength = mat.abs().groupby(groups.loc[ORDER]).mean().reindex(unique_groups)
    vmax = max(0.45, float(np.nanmax(group_strength.values)) * 1.05)
    im = ax.imshow(group_strength.values, cmap=CMAP_SEQ, vmin=0, vmax=vmax, aspect="auto")
    for i in range(group_strength.shape[0]):
        for j in range(group_strength.shape[1]):
            v = group_strength.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.0,
                    color="white" if v > vmax * 0.58 else "black")
    ax.set_xticks(range(len(TARGET_ORDER)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=10.5)
    ax.set_yticks(range(len(unique_groups)))
    ax.set_yticklabels([f"{g}  (n={int(group_counts[g])})" for g in unique_groups], fontsize=11.0)
    for i, g in enumerate(unique_groups):
        ax.add_patch(plt.Rectangle((-0.90, i - 0.5), 0.35, 1, color=group_colors[g], clip_on=False))
    ax.set_xlim(-1.05, len(TARGET_ORDER) - 0.5)
    cax = ax.inset_axes([0.16, 1.045, 0.68, 0.04])
    cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Mean absolute Spearman's $\\rho$", fontsize=12.2)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=11.5, length=3.8, width=1.1)
    ax.set_title("A   Feature-category association strength", fontsize=13, fontweight="bold", loc="left", pad=10)
    return im, cbar, group_strength


# ============ Panel A: feature-category association strength ============
axA = fig.add_subplot(gs[0, 0])
_, cbarA, group_strength = category_strength_heatmap(axA)
handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors[g]) for g in unique_groups]
fig.legend(handles, unique_groups, loc="upper center", bbox_to_anchor=(0.5, 0.516),
           ncol=6, frameon=False, fontsize=TOP_LEGEND_FONT_SIZE)

# ============ Panel B: desirability-adjusted correlation ============
axB = fig.add_subplot(gs[0, 1])
_, cbarB = grouped_heatmap(axB, mat_adj, matp, "B   Desirability-adjusted effect (+ = toward desirable outcome)",
                           "Desirability-adjusted $\\rho$",
                           show_ylabels=False)
axB.set_yticklabels(short_labels, fontsize=9.5)
axB.tick_params(axis="y", length=7.2, labelleft=True)

# ============ Panel C: universal-consensus ranking ============
axC = fig.add_subplot(gs[1, 0])
n_sig = sig.loc[ORDER].sum(axis=1)
consensus = mat_adj_sig.mean(axis=1)
cons_df = pd.DataFrame({"consensus": consensus, "n_sig": n_sig, "group": groups.loc[ORDER]}).dropna()
cons_df = cons_df.sort_values("consensus", ascending=True)
ypos = np.arange(len(cons_df))
bar_colors = ["#83AE8F" if v > 0 else "#C97979" for v in cons_df["consensus"]]
cons_ypos = ypos.copy()
cons_bar_colors = list(bar_colors)
axC.barh(ypos, cons_df["consensus"], color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
for y, (feat, row) in zip(ypos, cons_df.iterrows()):
    axC.text(row["consensus"] + (0.012 if row["consensus"] >= 0 else -0.012), y,
             f"n={int(row['n_sig'])}/7", va="center",
             ha="left" if row["consensus"] >= 0 else "right", fontsize=8.5, color="grey")
axC.set_yticks(ypos)
axC.set_yticklabels([rank_labels.get(f, FEATURE_LABELS.get(f, f)) for f in cons_df.index], fontsize=10.0)
axC.axvline(0, color="black", lw=0.8)
axC.set_xlabel("Mean desirability-adjusted effect", fontsize=12.2)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.set_xlim(-0.55, 0.55)
axC.set_title("C   Universal levers: consensus direction across endpoints",
               fontsize=13, fontweight="bold", loc="left")

# ============ Panel D: cross-endpoint conflict ranking ============
axD = fig.add_subplot(gs[1, 1])
conflict = (mat_adj_sig.max(axis=1) - mat_adj_sig.min(axis=1))
conf_df = pd.DataFrame({"conflict": conflict, "n_sig": n_sig, "group": groups.loc[ORDER]}).dropna()
conf_df = conf_df[conf_df["n_sig"] >= 2].sort_values("conflict", ascending=True)
ypos = np.arange(len(conf_df))
bar_colors = [group_colors[g] for g in conf_df["group"]]
axD.barh(ypos, conf_df["conflict"], color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
axD.set_yticks(ypos)
axD.set_yticklabels([rank_labels.get(f, FEATURE_LABELS.get(f, f)) for f in conf_df.index], fontsize=10.0)
axD.set_xlabel("Effect range across significant endpoints", fontsize=12.2)
axD.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.set_title("D   Trade-off levers: cross-endpoint conflict (effect range)",
               fontsize=13, fontweight="bold", loc="left")

polish_figure(fig, boxed_axes=(axA, axB))
save_combined_figure(fig, OUT_DIR, "Fig7_decision_matrix")


def draw_panel_a(panel_fig):
    ax = panel_fig.add_subplot(111)
    _, cb, _ = category_strength_heatmap(ax)
    handles_local = [plt.Rectangle((0, 0), 1, 1, color=group_colors[g]) for g in unique_groups]
    panel_fig.legend(handles_local, unique_groups, loc="upper center", bbox_to_anchor=(0.55, 1.22),
                     ncol=3, frameon=False, fontsize=STANDALONE_TOP_LEGEND_FONT_SIZE,
                     title="Feature category", title_fontsize=STANDALONE_TOP_LEGEND_TITLE_SIZE)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    _, cb = grouped_heatmap(ax, mat_adj, matp, "B   Desirability-adjusted effect (+ = toward desirable outcome)",
                            "Desirability-adjusted $\\rho$", show_ylabels=False)
    ax.set_yticklabels(short_labels, fontsize=8.8)
    ax.tick_params(axis="y", length=7.2, labelleft=True)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.barh(cons_ypos, cons_df["consensus"], color=cons_bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for y, (feat, row) in zip(cons_ypos, cons_df.iterrows()):
        ax.text(row["consensus"] + (0.012 if row["consensus"] >= 0 else -0.012), y,
                f"n={int(row['n_sig'])}/7", va="center",
                ha="left" if row["consensus"] >= 0 else "right", fontsize=8, color="grey")
    ax.set_yticks(cons_ypos)
    ax.set_yticklabels([rank_labels.get(f, FEATURE_LABELS.get(f, f)) for f in cons_df.index], fontsize=9.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean desirability-adjusted effect", fontsize=11.5)
    ax.set_xlim(-0.55, 0.55)
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.barh(np.arange(len(conf_df)), conf_df["conflict"], color=[group_colors[g] for g in conf_df["group"]],
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(np.arange(len(conf_df)))
    ax.set_yticklabels([rank_labels.get(f, FEATURE_LABELS.get(f, f)) for f in conf_df.index], fontsize=9.5)
    ax.set_xlabel("Effect range across significant endpoints", fontsize=11.5)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    return ax


save_independent_subfigures(OUT_DIR, "Fig7_decision_matrix", [
    ("a", (7.4, 5.8), draw_panel_a, None),
    ("b", (7.2, 7.0), draw_panel_b, None),
    ("c", (6.6, 6.4), draw_panel_c, None),
    ("d", (6.8, 6.4), draw_panel_d, None),
])
print("Saved Fig8.")

mat.round(3).to_csv(os.path.join(OUT_DIR, "Table_feature_endpoint_rho.csv"))
matp.round(4).to_csv(os.path.join(OUT_DIR, "Table_feature_endpoint_pvalue.csv"))
mat_adj.round(3).to_csv(os.path.join(OUT_DIR, "Table_feature_endpoint_desirability_adjusted.csv"))
group_strength.round(3).to_csv(os.path.join(OUT_DIR, "Table_feature_category_endpoint_strength.csv"))
cons_df.round(3).to_csv(os.path.join(OUT_DIR, "Table_universal_consensus_ranking.csv"))
conf_df.round(3).to_csv(os.path.join(OUT_DIR, "Table_crossendpoint_conflict_ranking.csv"))
