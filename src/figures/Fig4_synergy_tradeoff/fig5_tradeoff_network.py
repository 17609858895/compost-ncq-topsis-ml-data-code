"""
Figure 4. Synergy-versus-trade-off network among the 7 nutrient/carbon/quality
endpoints, based on the similarity of their management-response profiles
(Spearman correlation between each endpoint and the 20 shared management
parameters). Two endpoints with similar response profiles (positive
correlation) are 'synergistic' -- the same management adjustment moves both
in a favourable direction together once the desirability sign is accounted
for; endpoints with opposing profiles are 'trade-offs'.

Style adapted from: 期刊配图：Spearman相关系数网络图可视化展示变量关联强度 (#10) and
基于 Mantel 检验的网络图与相关性热力图结合的顶刊可视化图表实现 (#167).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import (TARGET_ORDER, TARGET_LABELS, TARGET_COLORS, TARGET_DESIRABLE_DIRECTION,
                    COMMON_FEATURES, RESULTS_DIR, FEATURE_GROUPS, FEATURE_GROUP_ORDER,
                    FEATURE_GROUP_COLORS, FEATURE_LABELS, CMAP_DIVERGE)
from plot_style import configure_publication_style, polish_figure, save_combined_figure

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

# Fig. 4 is imported into Word at near page width. Keep the canvas close to
# manuscript width and tune text after polishing so the panels stay readable
# without the heavy, crowded look of the first draft.
SYNERGY_COLOR = "#7EAD88"
TRADEOFF_COLOR = "#C97979"
AXIS_COLOR = "#262626"
GUIDE_COLOR = "#B8B8B8"
HEATMAP_CELL_FONTSIZE = 7.6
HEATMAP_TICK_FONTSIZE = 8.4
HEATMAP_CB_LABEL_FONTSIZE = 8.7
HEATMAP_CB_TICK_FONTSIZE = 8.0
PROFILE_LABEL_FONTSIZE = 10.6
PROFILE_TICK_FONTSIZE = 9.2
NETWORK_LABEL_FONTSIZE = 8.3
NETWORK_NODE_SIZE = 760
NETWORK_LEGEND_FONTSIZE = 8.7
FEATURE_LEGEND_FONTSIZE = 10.0
FEATURE_LEGEND_TITLE_FONTSIZE = 10.5
STANDALONE_PROFILE_LABEL_FONTSIZE = 11.4
STANDALONE_PROFILE_TICK_FONTSIZE = 10.0
STANDALONE_FEATURE_LEGEND_FONTSIZE = 9.6
STANDALONE_FEATURE_LEGEND_TITLE_FONTSIZE = 10.0
NETWORK_NODE_LABELS = {
    "NH3-N loss": r"NH$_3$-N",
    "N2O-N loss": r"N$_2$O-N",
    "CH4-C loss": r"CH$_4$-C",
    "CO2-C loss": r"CO$_2$-C",
    "TC loss": "TC",
    "TN loss": "TN",
    "Final GI": "GI",
}


def safe_save(fig_obj, path, **kwargs):
    try:
        fig_obj.savefig(path, **kwargs)
    except PermissionError:
        root, ext = os.path.splitext(path)
        alt = f"{root}_locked-copy{ext}"
        fig_obj.savefig(alt, **kwargs)
        print(f"Saved alternate file because the original is locked: {alt}")


def tune_heatmap_text(ax, colorbar_ax=None):
    for text in ax.texts:
        text.set_fontsize(HEATMAP_CELL_FONTSIZE)
        text.set_fontweight("normal")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(HEATMAP_TICK_FONTSIZE)
        label.set_fontweight("semibold")
    ax.tick_params(axis="both", which="major", labelsize=HEATMAP_TICK_FONTSIZE,
                   length=3.8, width=0.95, colors=AXIS_COLOR)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color(AXIS_COLOR)
    if colorbar_ax is not None:
        colorbar_ax.xaxis.set_ticks_position("top")
        colorbar_ax.xaxis.set_label_position("top")
        colorbar_ax.xaxis.label.set_fontsize(HEATMAP_CB_LABEL_FONTSIZE)
        colorbar_ax.xaxis.label.set_fontweight("semibold")
        colorbar_ax.tick_params(labelsize=HEATMAP_CB_TICK_FONTSIZE, length=2.8, width=0.85,
                                colors=AXIS_COLOR, labeltop=True, labelbottom=False, pad=1.5)
        colorbar_ax.xaxis.labelpad = 11
        for label in colorbar_ax.get_xticklabels() + colorbar_ax.get_yticklabels():
            label.set_fontsize(HEATMAP_CB_TICK_FONTSIZE)
            label.set_fontweight("normal")
        for spine in colorbar_ax.spines.values():
            spine.set_linewidth(0.9)


def tune_profile_axis(ax, label_size=PROFILE_LABEL_FONTSIZE, tick_size=PROFILE_TICK_FONTSIZE):
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("semibold")
    ax.yaxis.label.set_fontweight("semibold")
    ax.tick_params(axis="both", which="major", labelsize=tick_size,
                   length=4.4, width=1.05, colors=AXIS_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_size)
        label.set_fontweight("semibold")
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.15)
        ax.spines[side].set_color(AXIS_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tune_axes_legend(ax, font_size, title_size):
    legend = ax.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(font_size)
        text.set_fontweight("normal")
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(title_size)
        legend.get_title().set_fontweight("semibold")


def tune_network_legend(ax):
    tune_axes_legend(ax, NETWORK_LEGEND_FONTSIZE, NETWORK_LEGEND_FONTSIZE)


def tune_network_text(ax):
    for text in ax.texts:
        text.set_fontsize(NETWORK_LABEL_FONTSIZE)
        text.set_fontweight("semibold")


def tune_profile_legend(ax, font_size=FEATURE_LEGEND_FONTSIZE,
                        title_size=FEATURE_LEGEND_TITLE_FONTSIZE):
    tune_axes_legend(ax, font_size, title_size)


corr = pd.read_csv(os.path.join(RESULTS_DIR, "feature_target_corr.csv"))
# Build a 20 (features) x 7 (targets) matrix of "desirability-adjusted" rho:
# multiply rho by the desirability direction so that a positive value always
# means "this management change moves the endpoint toward the desirable side".
mat = corr.pivot(index="Feature", columns="Target", values="rho").reindex(COMMON_FEATURES)[TARGET_ORDER]
mat_adj = mat.copy()
for t in TARGET_ORDER:
    mat_adj[t] = mat_adj[t] * TARGET_DESIRABLE_DIRECTION[t]

# Pairwise similarity of desirability-adjusted response profiles
n = len(TARGET_ORDER)
sim = pd.DataFrame(np.eye(n), index=TARGET_ORDER, columns=TARGET_ORDER)
pval = pd.DataFrame(np.zeros((n, n)), index=TARGET_ORDER, columns=TARGET_ORDER)
for i, t1 in enumerate(TARGET_ORDER):
    for j, t2 in enumerate(TARGET_ORDER):
        if i == j:
            continue
        sub = mat_adj[[t1, t2]].dropna()
        r, p = spearmanr(sub[t1], sub[t2])
        sim.loc[t1, t2] = r
        pval.loc[t1, t2] = p

fig = plt.figure(figsize=(7.45, 6.85), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[0.98, 1.02],
                      wspace=0.34, hspace=0.48)

# ============ Panel A: network ============
axA = fig.add_subplot(gs[0, 0])
G = nx.Graph()
for t in TARGET_ORDER:
    G.add_node(t)
edges = []
for i, t1 in enumerate(TARGET_ORDER):
    for j, t2 in enumerate(TARGET_ORDER):
        if j <= i:
            continue
        r = sim.loc[t1, t2]
        p = pval.loc[t1, t2]
        if abs(r) >= 0.3:
            G.add_edge(t1, t2, weight=r, p=p)
            edges.append((t1, t2, r, p))

pos = {
    "CH4-C loss": (-0.08, 0.82),
    "N2O-N loss": (0.72, 0.62),
    "NH3-N loss": (1.04, 0.00),
    "Final GI": (0.66, -0.66),
    "TN loss": (-0.08, -0.86),
    "TC loss": (-0.86, -0.38),
    "CO2-C loss": (-0.90, 0.34),
}


def draw_network_labels(ax):
    light_text_nodes = {"NH3-N loss", "CH4-C loss", "TN loss"}
    for target, (x, y) in pos.items():
        ax.text(x, y, NETWORK_NODE_LABELS[target], ha="center", va="center",
                fontsize=NETWORK_LABEL_FONTSIZE, fontweight="semibold",
                color="white" if target in light_text_nodes else "black",
                zorder=6)


node_colors = [TARGET_COLORS[t] for t in G.nodes()]
for t1, t2, r, p in edges:
    color = SYNERGY_COLOR if r > 0 else TRADEOFF_COLOR
    style = "solid" if p < 0.05 else "dashed"
    edge_artists = nx.draw_networkx_edges(G, pos, edgelist=[(t1, t2)], width=0.45 + 2.45 * abs(r),
                                          edge_color=color, style=style, alpha=0.62, ax=axA)
    if edge_artists is not None:
        edge_artists.set_clip_on(False)

node_artists = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=NETWORK_NODE_SIZE, ax=axA,
                                      edgecolors=AXIS_COLOR, linewidths=0.85)
node_artists.set_clip_on(False)
draw_network_labels(axA)
axA.set_xlim(-1.16, 1.27)
axA.set_ylim(-1.16, 1.12)
axA.set_aspect("equal")
axA.axis("off")
axA.set_title("")
legend_elems = [
    Line2D([0], [0], color=SYNERGY_COLOR, lw=2.4, label="Synergy ($p$<0.05)"),
    Line2D([0], [0], color=SYNERGY_COLOR, lw=2.4, ls="dashed", label="Synergy (n.s.)"),
    Line2D([0], [0], color=TRADEOFF_COLOR, lw=2.4, label="Trade-off ($p$<0.05)"),
    Line2D([0], [0], color=TRADEOFF_COLOR, lw=2.4, ls="dashed", label="Trade-off (n.s.)"),
]
axA.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.50, -0.235),
           fontsize=NETWORK_LEGEND_FONTSIZE, frameon=False, ncol=2, columnspacing=0.9,
           handlelength=1.65, handletextpad=0.45, labelspacing=0.30, borderaxespad=0.0)

# ============ Panel B: similarity heatmap with significance stars ============
axB = fig.add_subplot(gs[0, 1])
im = axB.imshow(sim.values, cmap=CMAP_DIVERGE, vmin=-1, vmax=1)
for i in range(n):
    for j in range(n):
        r = sim.iloc[i, j]
        if i == j:
            star = ""
        else:
            p = pval.iloc[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        axB.text(j, i, f"{r:.2f}{star}", ha="center", va="center", fontsize=HEATMAP_CELL_FONTSIZE,
                 color="white" if abs(r) > 0.55 else "black")
axB.set_xticks(range(n)); axB.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=HEATMAP_TICK_FONTSIZE)
axB.set_yticks(range(n)); axB.set_yticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=HEATMAP_TICK_FONTSIZE)
caxB = axB.inset_axes([0.17, 1.070, 0.66, 0.040])
cbar = fig.colorbar(im, cax=caxB, orientation="horizontal")
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_label("Response-profile correlation", fontsize=HEATMAP_CB_LABEL_FONTSIZE, labelpad=11)
cbar.ax.xaxis.set_label_position("top")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.tick_params(labelsize=HEATMAP_CB_TICK_FONTSIZE, length=3.0, width=0.9,
                    labeltop=True, labelbottom=False, pad=1.5)
axB.set_title("")

# ============ Panels C-D: worked examples of the underlying response profiles ============
unique_groups = [g for g in FEATURE_GROUP_ORDER if g in set(FEATURE_GROUPS.values())]
group_colors = {g: FEATURE_GROUP_COLORS[g] for g in unique_groups}

def profile_panel(ax, t1, t2, label):
    sub = mat_adj[[t1, t2]].dropna()
    r, p = sim.loc[t1, t2], pval.loc[t1, t2]
    for feat, row in sub.iterrows():
        ax.scatter(row[t1], row[t2], color=group_colors.get(FEATURE_GROUPS.get(feat, "Other"), "#999999"),
                   s=38, edgecolors=AXIS_COLOR, linewidths=0.45, alpha=0.94, zorder=3)
    # OLS trend line
    b1, b0 = np.polyfit(sub[t1], sub[t2], 1)
    xs = np.linspace(sub[t1].min(), sub[t1].max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=SYNERGY_COLOR if r > 0 else TRADEOFF_COLOR,
            lw=1.6, ls="--", alpha=0.82)
    ax.axhline(0, color=GUIDE_COLOR, lw=0.75, ls=":")
    ax.axvline(0, color=GUIDE_COLOR, lw=0.75, ls=":")
    ax.set_xlabel(f"Adjusted response profile\n({TARGET_LABELS[t1]})", fontsize=PROFILE_LABEL_FONTSIZE)
    ax.set_ylabel(f"Adjusted response profile\n({TARGET_LABELS[t2]})", fontsize=PROFILE_LABEL_FONTSIZE)
    ax.margins(x=0.08, y=0.10)
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axC = fig.add_subplot(gs[1, 0])
profile_panel(axC, "TC loss", "TN loss", "C")

axD = fig.add_subplot(gs[1, 1])
profile_panel(axD, "CH4-C loss", "Final GI", "D")

handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=group_colors[g],
                  markeredgecolor=AXIS_COLOR, markeredgewidth=0.75,
                  markersize=5.4, label=g) for g in unique_groups]

polish_figure(fig, boxed_axes=(axB,), text_scale=1.02)
tune_heatmap_text(axB, cbar.ax)
tune_profile_axis(axC)
tune_profile_axis(axD)
fig.subplots_adjust(left=0.095, right=0.985, top=0.925, bottom=0.185)
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
           ncol=6, frameon=False, fontsize=FEATURE_LEGEND_FONTSIZE,
           title="Feature group", title_fontsize=FEATURE_LEGEND_TITLE_FONTSIZE,
           columnspacing=0.62, handletextpad=0.30, labelspacing=0.28, borderaxespad=0.0)
tune_network_legend(axA)
tune_network_text(axA)
save_combined_figure(fig, OUT_DIR, "Fig4_synergy_tradeoff")


def draw_panel_a(panel_fig):
    ax = panel_fig.add_subplot(111)
    for t1, t2, r, p in edges:
        edge_artists = nx.draw_networkx_edges(G, pos, edgelist=[(t1, t2)], width=0.45 + 2.45 * abs(r),
                                              edge_color=SYNERGY_COLOR if r > 0 else TRADEOFF_COLOR,
                                              style="solid" if p < 0.05 else "dashed", alpha=0.62, ax=ax)
        if edge_artists is not None:
            edge_artists.set_clip_on(False)
    node_artists = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=NETWORK_NODE_SIZE, ax=ax,
                                          edgecolors=AXIS_COLOR, linewidths=0.85)
    node_artists.set_clip_on(False)
    draw_network_labels(ax)
    ax.set_xlim(-1.16, 1.27)
    ax.set_ylim(-1.16, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=NETWORK_LEGEND_FONTSIZE, frameon=False, ncol=2,
              columnspacing=0.9, handlelength=1.65, handletextpad=0.45,
              labelspacing=0.30, borderaxespad=0.0)
    return ax


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    im_local = ax.imshow(sim.values, cmap=CMAP_DIVERGE, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            r = sim.iloc[i, j]
            if i == j:
                star = ""
            else:
                p = pval.iloc[i, j]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(j, i, f"{r:.2f}{star}", ha="center", va="center", fontsize=HEATMAP_CELL_FONTSIZE,
                    color="white" if abs(r) > 0.55 else "black")
    ax.set_xticks(range(n))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], rotation=35, ha="right", fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_yticks(range(n))
    ax.set_yticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=HEATMAP_TICK_FONTSIZE)
    cax = ax.inset_axes([0.17, 1.080, 0.66, 0.040])
    cb = panel_fig.colorbar(im_local, cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_label("Response-profile correlation", fontsize=HEATMAP_CB_LABEL_FONTSIZE, labelpad=11)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=HEATMAP_CB_TICK_FONTSIZE, length=3.0, width=0.9,
                      labeltop=True, labelbottom=False, pad=1.5)
    return {"axes": [ax, cb.ax], "boxed_axes": (ax,)}


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111)
    profile_panel(ax, "TC loss", "TN loss", "C")
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    profile_panel(ax, "CH4-C loss", "Final GI", "D")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=3,
              frameon=False, fontsize=STANDALONE_FEATURE_LEGEND_FONTSIZE,
              title="Feature group", title_fontsize=STANDALONE_FEATURE_LEGEND_TITLE_FONTSIZE,
              columnspacing=1.1, handletextpad=0.48, labelspacing=0.42,
              borderaxespad=0.0)
    return ax


def save_independent_subfigures_tuned(out_dir, stem, panel_builders):
    sub_dir = os.path.join(out_dir, "subfigures")
    os.makedirs(sub_dir, exist_ok=True)
    for label, figsize, draw_func, boxed_axes, post_tune in panel_builders:
        panel_fig = plt.figure(figsize=figsize, dpi=300)
        result = draw_func(panel_fig)
        if isinstance(result, dict):
            axes = result.get("axes", ())
            boxed_axes = result.get("boxed_axes", boxed_axes)
        else:
            axes = result
        if boxed_axes is None:
            boxed_axes = ()
        elif boxed_axes == "returned":
            boxed_axes = axes if isinstance(axes, (list, tuple)) else (axes,)
        polish_figure(panel_fig, boxed_axes=boxed_axes, text_scale=1.02)
        if post_tune is not None:
            post_tune(panel_fig, result)
        safe_save(panel_fig, os.path.join(sub_dir, f"{stem}_{label}.png"), bbox_inches="tight", dpi=600)
        safe_save(panel_fig, os.path.join(sub_dir, f"{stem}_{label}.pdf"), bbox_inches="tight")
        plt.close(panel_fig)
    with open(os.path.join(sub_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {stem} subfigures\n\n")
        f.write("Subfigures were independently rendered on standalone canvases, not cropped from the combined figure.\n")


def tune_panel_b(_fig_obj, result):
    tune_heatmap_text(result["axes"][0], result["axes"][1])


def tune_panel_d(_fig_obj, result):
    tune_profile_axis(result, STANDALONE_PROFILE_LABEL_FONTSIZE, STANDALONE_PROFILE_TICK_FONTSIZE)
    tune_profile_legend(result, STANDALONE_FEATURE_LEGEND_FONTSIZE,
                        STANDALONE_FEATURE_LEGEND_TITLE_FONTSIZE)


save_independent_subfigures_tuned(OUT_DIR, "Fig4_synergy_tradeoff", [
    ("a", (7.4, 6.4), draw_panel_a, None, lambda _fig_obj, result: tune_network_legend(result)),
    ("b", (6.3, 5.8), draw_panel_b, None, tune_panel_b),
    ("c", (6.1, 5.3), draw_panel_c, None,
     lambda _fig_obj, result: tune_profile_axis(result, STANDALONE_PROFILE_LABEL_FONTSIZE,
                                                STANDALONE_PROFILE_TICK_FONTSIZE)),
    ("d", (6.1, 5.8), draw_panel_d, None, tune_panel_d),
])
print("Saved Fig4.")

sim.round(3).to_csv(os.path.join(OUT_DIR, "Table_response_profile_correlation.csv"))
pval.round(4).to_csv(os.path.join(OUT_DIR, "Table_response_profile_pvalue.csv"))
