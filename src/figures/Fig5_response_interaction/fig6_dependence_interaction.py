"""
Figure 6. Response curves of four shared management/substrate "levers" across all
seven endpoints (desirability-adjusted, standardized SHAP contribution vs. the
within-dataset percentile of the lever), plus two worked SHAP dependence /
interaction examples.

Panels A-D: for each lever (initial C/N ratio, turning frequency, composting
period, initial moisture content), a LOWESS-smoothed curve of the
desirability-adjusted, standardized SHAP value is shown for each of the seven
endpoints. A positive value means that increasing the lever moves that endpoint
toward its desirable direction. The colour strip above each panel summarises,
at each percentile, the fraction of the seven endpoints for which the response
is favourable (green = broad "win-win", red = broad "lose-lose").

Panels E-F: classic SHAP dependence plots with interaction colouring for two
illustrative cases (TN loss vs. turning times, coloured by initial C/N; Final GI
vs. initial C/N, coloured by turning times), each with a LOWESS trend line.

Style adapted from: 期刊配图：SHAP依赖图+交互效应可视化 (#391/#333), ES&T 风格
依赖图布局 (#203/#4/#5).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import TARGET_ORDER, TARGET_LABELS, TARGET_COLORS, TARGET_DESIRABLE_DIRECTION, FEATURE_LABELS, RESULTS_DIR, TARGETS, CMAP_CONT
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
configure_publication_style()

LEVERS = ['Initial C/N (%)', 'Turning times', 'Period (d)', 'Initial Moisture Content (%)']

CONT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "Fig5_dependence_nature_new",
    ["#F6D58B", "#D8E7CA", "#99CDCE", "#99BADF", "#777FBC"],
    N=256,
)
FAVOURABLE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "Fig5_favourable_nature_new", ["#C97979", "#F6D58B", "#83AE8F"], N=256
)

# ---- load per-target SHAP data (common-feature subset) ----
data = {}
for name in TARGET_ORDER:
    _, _, code = TARGETS[name]
    d = np.load(os.path.join(RESULTS_DIR, f"shap_{code}.npz"), allow_pickle=True)
    X = pd.DataFrame(d["X"], columns=d["feature_names"])
    SV = pd.DataFrame(d["shap_values"], columns=d["feature_names"])
    data[name] = (X, SV)

XG = np.linspace(0, 100, 101)

fig = plt.figure(figsize=(16.6, 10.4), dpi=300)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.08], hspace=0.60, wspace=0.42)

# ============ Panels A-D: universal lever response curves ============
lever_panel_axes = []
for idx, feat in enumerate(LEVERS):
    ax = fig.add_subplot(gs[0, idx])
    curves = {}
    for name in TARGET_ORDER:
        X, SV = data[name]
        if feat not in X.columns:
            continue
        x = X[feat].values.astype(float)
        if np.std(x) == 0 or pd.Series(x).nunique() < 5:
            continue
        xr = pd.Series(x).rank(pct=True).values * 100  # within-dataset percentile
        y = SV[feat].values.astype(float)
        y = y / (np.std(y) + 1e-9)                       # standardize
        y = y * TARGET_DESIRABLE_DIRECTION[name]          # desirability-adjusted
        sm = lowess(y, xr, frac=0.55, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color=TARGET_COLORS[name], lw=2, label=TARGET_LABELS[name], alpha=0.9)
        curves[name] = np.interp(XG, sm[:, 0], sm[:, 1])

    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{FEATURE_LABELS.get(feat, feat)}\n(within-dataset percentile)", fontsize=9.5)
    if idx == 0:
        ax.set_ylabel("Desirability-adjusted,\nstandardized SHAP value", fontsize=9.5)
    ax.set_title(f"{chr(65+idx)}   {FEATURE_LABELS.get(feat, feat)}", fontsize=12, fontweight="bold", loc="left")

    # agreement strip above each panel: fraction of endpoints with favourable response
    cmat = np.vstack(list(curves.values()))
    frac_pos = (cmat > 0).mean(axis=0)
    strip = ax.inset_axes([0, 1.04, 1, 0.09])
    strip.imshow(frac_pos[np.newaxis, :], aspect="auto", cmap=FAVOURABLE_CMAP, vmin=0, vmax=1,
                  extent=[0, 100, 0, 1])
    strip.set_xticks([]); strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    if idx == 0:
        strip.set_ylabel("Favourable\nfraction", fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
    lever_panel_axes.append([ax, strip])

handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.490),
           ncol=7, frameon=False, fontsize=11.2, title="Endpoint", title_fontsize=11.8)

# ============ Panels E-F: SHAP dependence + interaction examples ============
def draw_horizontal_gradient_cbar(cax, norm, cmap, label):
    values = np.linspace(norm.vmin, norm.vmax, 129)
    for left, right in zip(values[:-1], values[1:]):
        mid = 0.5 * (left + right)
        cax.add_patch(
            mpl.patches.Rectangle(
                (left, 0),
                right - left,
                1,
                facecolor=cmap(norm(mid)),
                edgecolor="none",
                linewidth=0,
            )
        )
    cax.set_xlim(norm.vmin, norm.vmax)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(label, fontsize=8.0, labelpad=2)
    cax.tick_params(axis="x", labelsize=7.8, length=2.8, width=0.9, direction="out")
    for spine in cax.spines.values():
        spine.set_linewidth(0.9)
    cax._colorbar = True
    return cax


def dependence_panel(ax, target, xfeat, cfeat, label, *, cbar_outside=False):
    X, SV = data[target]
    x = X[xfeat].values.astype(float)
    y = SV[xfeat].values.astype(float)
    c = X[cfeat].values.astype(float)
    # LOWESS computed on the full data (robust trend), but axes clipped to the
    # 1st-98th percentile range so a handful of extreme outliers don't compress
    # the bulk of the data into a corner of the panel.
    order = np.argsort(x)
    sm = lowess(y[order], x[order], frac=0.4, return_sorted=True)
    x_lo, x_hi = np.percentile(x, [1, 98])
    c_lo, c_hi = np.percentile(c, [2, 98])
    norm = mpl.colors.Normalize(vmin=c_lo, vmax=c_hi)
    sc = ax.scatter(x, y, c=c, cmap=CONT_CMAP, norm=norm, s=18, alpha=0.82, edgecolors="none")
    ax.plot(sm[:, 0], sm[:, 1], color="#B86F72", lw=2.2, label="LOWESS trend")
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.7)
    pad = 0.04 * (x_hi - x_lo)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_xlabel(FEATURE_LABELS.get(xfeat, xfeat), fontsize=10)
    ax.set_ylabel(f"SHAP value\n({FEATURE_LABELS.get(xfeat, xfeat)})", fontsize=10, labelpad=8)
    if cbar_outside:
        cax = ax.inset_axes([0.57, 1.055, 0.36, 0.070])
    else:
        cax = inset_axes(ax, width="35%", height="5.8%", loc="upper right", borderpad=1.35)
    cbar_ax = draw_horizontal_gradient_cbar(cax, norm, CONT_CMAP, FEATURE_LABELS.get(cfeat, cfeat))
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_title(f"{label}   SHAP dependence: {target} ~ {FEATURE_LABELS.get(xfeat, xfeat)}"
                  f"\n(colour = {FEATURE_LABELS.get(cfeat, cfeat)})",
                  fontsize=12, fontweight="bold", loc="left")
    return cbar_ax

axE = fig.add_subplot(gs[1, 0:2])
cbarE = dependence_panel(axE, "TN loss", "Turning times", "Initial C/N (%)", "E")

axF = fig.add_subplot(gs[1, 2:4])
cbarF = dependence_panel(axF, "Final GI", "Initial C/N (%)", "Turning times", "F")

polish_figure(fig, text_scale=1.12)
save_combined_figure(fig, OUT_DIR, "Fig5_response_interaction")


def draw_lever_panel(panel_fig, feat, idx):
    ax = panel_fig.add_subplot(111)
    curves = {}
    for name in TARGET_ORDER:
        X, SV = data[name]
        if feat not in X.columns:
            continue
        x = X[feat].values.astype(float)
        if np.std(x) == 0 or pd.Series(x).nunique() < 5:
            continue
        xr = pd.Series(x).rank(pct=True).values * 100
        y = SV[feat].values.astype(float)
        y = y / (np.std(y) + 1e-9)
        y = y * TARGET_DESIRABLE_DIRECTION[name]
        sm = lowess(y, xr, frac=0.55, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color=TARGET_COLORS[name], lw=2,
                label=TARGET_LABELS[name], alpha=0.9)
        curves[name] = np.interp(XG, sm[:, 0], sm[:, 1])
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{FEATURE_LABELS.get(feat, feat)}\n(within-dataset percentile)", fontsize=10.5)
    ax.set_ylabel("Desirability-adjusted,\nstandardized SHAP value", fontsize=10.5)
    cmat = np.vstack(list(curves.values()))
    frac_pos = (cmat > 0).mean(axis=0)
    strip = ax.inset_axes([0, 1.04, 1, 0.09])
    strip.imshow(frac_pos[np.newaxis, :], aspect="auto", cmap=FAVOURABLE_CMAP, vmin=0, vmax=1,
                 extent=[0, 100, 0, 1])
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    strip.set_ylabel("Favourable\nfraction", fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4,
              frameon=False, fontsize=10.0, title="Endpoint", title_fontsize=10.5)
    return [ax, strip]


def draw_panel_e(panel_fig):
    ax = panel_fig.add_subplot(111)
    cbar_ax = dependence_panel(ax, "TN loss", "Turning times", "Initial C/N (%)", "E", cbar_outside=True)
    return [ax, cbar_ax]


def draw_panel_f(panel_fig):
    ax = panel_fig.add_subplot(111)
    cbar_ax = dependence_panel(ax, "Final GI", "Initial C/N (%)", "Turning times", "F", cbar_outside=True)
    return [ax, cbar_ax]


save_independent_subfigures(OUT_DIR, "Fig5_response_interaction", [
    ("a", (5.2, 4.8), lambda pfig: draw_lever_panel(pfig, LEVERS[0], 0), None),
    ("b", (5.2, 4.8), lambda pfig: draw_lever_panel(pfig, LEVERS[1], 1), None),
    ("c", (5.2, 4.8), lambda pfig: draw_lever_panel(pfig, LEVERS[2], 2), None),
    ("d", (5.2, 4.8), lambda pfig: draw_lever_panel(pfig, LEVERS[3], 3), None),
    ("e", (7.2, 5.8), draw_panel_e, None),
    ("f", (7.2, 5.8), draw_panel_f, None),
])
print("Saved Fig6.")
