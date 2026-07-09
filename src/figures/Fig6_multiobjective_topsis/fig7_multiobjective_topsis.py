"""
Figure 7. Multi-objective decision-support map for compost management:
predicted desirability-adjusted profiles of the 7 endpoints (NH3-N, N2O-N,
CH4-C, CO2-C, TC, TN losses, and Final GI) across a sampled space of
management practices, aggregated into three composite objectives
(N conservation, C/GHG mitigation, product quality) and ranked with TOPSIS
to identify "win-win" management recipes.

Panel A: trade-off landscape (N-conservation vs. C/GHG-mitigation score,
         coloured by quality score), with the TOPSIS-optimal recipe marked.
Panel B: TOPSIS-score distribution across the sampled design space.
Panel C: 7-endpoint desirability radar -- baseline vs. TOPSIS-optimal vs.
         TOPSIS-worst recipes.
Panel D: recommended management adjustments (TOPSIS-optimal vs. baseline),
         normalised to the sampled range of each lever.
Panel E: sensitivity of the three composite objectives to composting period,
         holding all other levers at their TOPSIS-optimal values.
Panel F: 2-D TOPSIS-score design map over composting period x turning
         frequency, holding other levers at their TOPSIS-optimal values.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

try:
    import numpy.random._pickle as _np_random_pickle
    _np_random_pickle.BitGenerators[np.random.MT19937] = np.random.MT19937
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from config import (TARGETS, COMMON_FEATURES, TARGET_ORDER, TARGET_COLORS,
                     TARGET_DESIRABLE_DIRECTION, RESULTS_DIR, FEATURE_LABELS,
                     CMAP_CONT, CMAP_SEQ, TARGET_LABELS)
from plot_style import configure_publication_style, polish_figure, save_combined_figure, save_independent_subfigures

OUT_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")
configure_publication_style()

RNG = np.random.default_rng(42)
N_SAMPLES = 4000

# =====================================================================
# 1. Load the best-performing fitted model per target
# =====================================================================
perf = pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv"))
best_model_name = perf.loc[perf.groupby("Target")["R2_test"].idxmax()].set_index("Target")["Model"]

fitted_models, feat_orders, extra_baseline = {}, {}, {}
for name in TARGET_ORDER:
    fpath, ycol, code = TARGETS[name]
    df = pd.read_csv(fpath)
    X = df.drop(columns=[ycol])
    feat_orders[name] = list(X.columns)
    with open(os.path.join(CKPT_DIR, f"{code}.pkl"), "rb") as f:
        state = pickle.load(f)
    fitted_models[name] = state["fitted"][best_model_name[name]]
    for c in X.columns:
        if c not in COMMON_FEATURES:
            extra_baseline.setdefault(c, {})[name] = float(X[c].median())

print("Best model per target:")
print(best_model_name)

# =====================================================================
# 2. Pooled common-feature space + baseline ("typical practice") vector
# =====================================================================
frames = []
for name in TARGET_ORDER:
    fpath, ycol, code = TARGETS[name]
    df = pd.read_csv(fpath)
    frames.append(df[COMMON_FEATURES])
pooled = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

CATEGORICAL = ['Material_Main', 'Material_2', 'Material_3', 'Additive_1', 'Additive_2',
               'Additive_3', 'Additive_4', ' Composting Method']
baseline = {}
for c in COMMON_FEATURES:
    if c in CATEGORICAL:
        baseline[c] = float(pooled[c].mode().iloc[0])
    else:
        baseline[c] = float(pooled[c].median())

# =====================================================================
# 3. Sample the management-practice design space
#    (continuous levers within the 2nd-98th percentile of the pooled data;
#     binary/categorical aeration switches sampled across observed levels;
#     all other common features fixed at the "typical practice" baseline)
# =====================================================================
LEVERS = {
    'Period (d)':                          (15.0, 133.0),
    'Turning times':                       (0.0, 54.0),
    'Initial C/N (%)':                     (9.7, 43.0),
    'Initial Moisture Content (%)':        (50.0, 75.8),
    'V5_Ventilation rate (L/min/kg iniDW)': (0.0, 7.32),
    'Application Rate (%DW)':              (0.0, 29.0),
    'V4_Ventilation Day':                   (0.0, 100.0),
}
LOG_LEVERS = {'Compost volume (m3)': (0.002, 11.25)}
BINARY_LEVERS = ['M1_is Enclosed']
CAT_LEVERS = {'V1_Ventilation Type': [0, 1, 2]}

LEVER_LABELS = {**FEATURE_LABELS}
SHORT_LEVER_LABELS = {
    'Period (d)': 'Period',
    'Turning times': 'Turning',
    'Initial C/N (%)': 'C/N ratio',
    'Initial Moisture Content (%)': 'Moisture',
    'V5_Ventilation rate (L/min/kg iniDW)': 'Aer. rate',
    'Application Rate (%DW)': 'Bulking rate',
    'V4_Ventilation Day': 'Aer. onset',
    'Compost volume (m3)': 'Pile volume',
    'M1_is Enclosed': 'Enclosed',
    'V1_Ventilation Type': 'Aeration type',
}

sample = pd.DataFrame({c: baseline[c] for c in COMMON_FEATURES}, index=range(N_SAMPLES))
for c, (lo, hi) in LEVERS.items():
    sample[c] = RNG.uniform(lo, hi, N_SAMPLES)
for c, (lo, hi) in LOG_LEVERS.items():
    sample[c] = np.exp(RNG.uniform(np.log(lo), np.log(hi), N_SAMPLES))
for c in BINARY_LEVERS:
    sample[c] = RNG.integers(0, 2, N_SAMPLES).astype(float)
for c, cats in CAT_LEVERS.items():
    sample[c] = RNG.choice(cats, N_SAMPLES).astype(float)

ALL_LEVERS = list(LEVERS.keys()) + list(LOG_LEVERS.keys()) + BINARY_LEVERS + list(CAT_LEVERS.keys())


def predict_all(df_common):
    """Predict the 7 endpoints for a dataframe of COMMON_FEATURES rows."""
    out = pd.DataFrame(index=df_common.index)
    for name in TARGET_ORDER:
        cols = feat_orders[name]
        Xp = pd.DataFrame(index=df_common.index, columns=cols, dtype=float)
        for c in cols:
            if c in COMMON_FEATURES:
                Xp[c] = df_common[c].values
            else:
                Xp[c] = extra_baseline[c][name]
        out[name] = fitted_models[name].predict(Xp[cols])
    return out


preds = predict_all(sample)

# =====================================================================
# 4. Desirability-adjusted scores (min-max over the sampled population)
#    and TOPSIS ranking
# =====================================================================
desir = pd.DataFrame(index=sample.index)
scalers = {}
for name in TARGET_ORDER:
    v = preds[name].values
    vmin, vmax = v.min(), v.max()
    scalers[name] = (vmin, vmax)
    norm = (v - vmin) / (vmax - vmin + 1e-12)
    desir[name] = norm if TARGET_DESIRABLE_DIRECTION[name] == 1 else 1 - norm

sample['N_score'] = desir[['NH3-N loss', 'N2O-N loss', 'TN loss']].mean(axis=1)
sample['C_score'] = desir[['CH4-C loss', 'CO2-C loss', 'TC loss']].mean(axis=1)
sample['Q_score'] = desir['Final GI']

D = desir[TARGET_ORDER].values
ideal, neg_ideal = D.max(axis=0), D.min(axis=0)
dist_pos = np.sqrt(((D - ideal) ** 2).sum(axis=1))
dist_neg = np.sqrt(((D - neg_ideal) ** 2).sum(axis=1))
sample['TOPSIS'] = dist_neg / (dist_pos + dist_neg + 1e-12)
for name in TARGET_ORDER:
    sample[f"pred_{name}"] = preds[name]
    sample[f"desir_{name}"] = desir[name]

best_idx = sample['TOPSIS'].idxmax()
worst_idx = sample['TOPSIS'].idxmin()
top_recipe = sample.loc[best_idx]
worst_recipe = sample.loc[worst_idx]
top_decile = sample['TOPSIS'].quantile(0.95)

# baseline ("typical practice") desirability, on the same scaling as the sample
baseline_df = pd.DataFrame([baseline])
baseline_pred = predict_all(baseline_df)
baseline_desir = {}
for name in TARGET_ORDER:
    vmin, vmax = scalers[name]
    norm = np.clip((baseline_pred[name].iloc[0] - vmin) / (vmax - vmin + 1e-12), 0, 1)
    baseline_desir[name] = norm if TARGET_DESIRABLE_DIRECTION[name] == 1 else 1 - norm

print("\nTOPSIS-optimal recipe (top score = %.3f):" % top_recipe['TOPSIS'])
for c in ALL_LEVERS:
    print(f"   {LEVER_LABELS.get(c, c):28s} baseline={baseline[c]:8.2f}  optimal={top_recipe[c]:8.2f}")
print("\nPredicted endpoint values (optimal vs baseline):")
for name in TARGET_ORDER:
    print(f"   {name:12s} optimal={top_recipe[f'pred_{name}']:7.2f}  baseline={baseline_pred[name].iloc[0]:7.2f}")

# =====================================================================
# Figure layout: 6 panels (3 x 2)
# =====================================================================
fig = plt.figure(figsize=(13.8, 18.2), dpi=300)
gs = fig.add_gridspec(3, 2, width_ratios=[1.10, 1.0], height_ratios=[1, 1, 1],
                      wspace=0.40, hspace=0.46)


def topsis_from_desir(desir_arr):
    """TOPSIS score for an array of (already 0-1 clipped) desirability values."""
    d = np.clip(desir_arr, 0, 1)
    dist_pos = np.sqrt(((d - 1) ** 2).sum(axis=-1))
    dist_neg = np.sqrt((d ** 2).sum(axis=-1))
    return dist_neg / (dist_pos + dist_neg + 1e-12)


def desir_from_pred(pred_df):
    """Convert raw predictions to desirability using the population scalers."""
    out = pd.DataFrame(index=pred_df.index)
    for name in TARGET_ORDER:
        vmin, vmax = scalers[name]
        norm = (pred_df[name].values - vmin) / (vmax - vmin + 1e-12)
        out[name] = norm if TARGET_DESIRABLE_DIRECTION[name] == 1 else 1 - norm
    return out


# ============ Panel A: trade-off landscape ============
axA = fig.add_subplot(gs[0, 0])
sc = axA.scatter(sample['N_score'], sample['C_score'], c=sample['Q_score'], cmap=CMAP_CONT,
                  s=14, alpha=0.55, edgecolors="none", vmin=0, vmax=1)
highlight = sample['TOPSIS'] >= top_decile
axA.scatter(sample.loc[highlight, 'N_score'], sample.loc[highlight, 'C_score'],
            facecolors="none", edgecolors="#C97979", linewidths=0.6, s=24, alpha=0.8,
            label="Top 5% TOPSIS score")
axA.scatter([top_recipe['N_score']], [top_recipe['C_score']], marker="*", s=420,
             color="#E2B36E", edgecolors="black", linewidths=1.2, zorder=5,
             label="TOPSIS-optimal recipe")
baseline_N_score = np.mean([baseline_desir["NH3-N loss"], baseline_desir["N2O-N loss"], baseline_desir["TN loss"]])
baseline_C_score = np.mean([baseline_desir["CH4-C loss"], baseline_desir["CO2-C loss"], baseline_desir["TC loss"]])
axA.scatter([baseline_N_score], [baseline_C_score], marker="s", s=140, color="white",
             edgecolors="black", linewidths=1.4, zorder=5, label="Current typical practice")
caxA = axA.inset_axes([0.14, 1.055, 0.72, 0.045])
cbarA = fig.colorbar(sc, cax=caxA, orientation="horizontal")
cbarA.set_label("Final GI desirability", fontsize=9.5)
cbarA.ax.xaxis.set_label_position("top")
cbarA.ax.tick_params(labelsize=9.2, length=3.4, width=1.0)
axA.set_xlabel("N-conservation score", fontsize=10.8)
axA.set_ylabel("C/GHG-mitigation score", fontsize=10.8)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.legend(loc="lower left", frameon=True, framealpha=0.85, edgecolor="none", fontsize=8.5)
axA.set_title("A   Multi-objective trade-off landscape", fontsize=12, fontweight="bold", loc="left")

# ============ Panel B: TOPSIS-score distribution ============
axB = fig.add_subplot(gs[0, 1])
axB.hist(sample['TOPSIS'], bins=40, color="#78BFC2", edgecolor="black", linewidth=0.4, alpha=0.85)
axB.axvline(top_decile, color="#C97979", lw=1.6, ls="--", label="Top 5% threshold")
baseline_topsis = topsis_from_desir(np.array([[baseline_desir[t] for t in TARGET_ORDER]]))[0]
axB.axvline(baseline_topsis, color="black", lw=1.6, ls="-", label="Current typical practice")
axB.axvline(top_recipe['TOPSIS'], color="#E2B36E", lw=2.0, ls="-", label="TOPSIS-optimal recipe")
axB.set_xlabel("TOPSIS score", fontsize=10.8)
axB.set_ylabel("Number of sampled recipes", fontsize=10.5)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.legend(loc="upper left", frameon=False, fontsize=9)
axB.set_title("B   TOPSIS-score distribution (n=4,000)", fontsize=12, fontweight="bold", loc="left")

# ============ Panel C: 7-endpoint desirability radar ============
axC = fig.add_subplot(gs[1, 0], polar=True)
n_ax = len(TARGET_ORDER)
angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
angles += angles[:1]

series = {
    "TOPSIS-optimal recipe": ([top_recipe[f"desir_{t}"] for t in TARGET_ORDER], "#83AE8F"),
    "Current typical practice": ([baseline_desir[t] for t in TARGET_ORDER], "black"),
    "TOPSIS-worst recipe": ([worst_recipe[f"desir_{t}"] for t in TARGET_ORDER], "#C97979"),
}
for label, (vals, color) in series.items():
    vals = vals + vals[:1]
    ls = "-" if label != "Current typical practice" else "--"
    axC.plot(angles, vals, color=color, linewidth=2, label=label, alpha=0.9, ls=ls)
    axC.fill(angles, vals, color=color, alpha=0.08)

axC.set_xticks(angles[:-1])
axC.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=9.5)
axC.set_ylim(0, 1)
axC.set_yticks([0.25, 0.5, 0.75, 1.0])
axC.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
axC.set_rlabel_position(15)
axC.spines['polar'].set_color('grey')
axC.grid(color="grey", alpha=0.3)
axC.set_title("C   Desirability profile (0=worst, 1=best)",
               fontsize=12, fontweight="bold", loc="left", pad=24)
axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False,
           fontsize=9, ncol=2, columnspacing=1.0, handletextpad=0.45,
           labelspacing=0.35)

# ============ Panel D: recommended management adjustments ============
axD = fig.add_subplot(gs[1, 1])
LEVER_RANGES = {**LEVERS, **LOG_LEVERS, 'M1_is Enclosed': (0, 1), 'V1_Ventilation Type': (0, 2)}


def norm_lever(c, v):
    lo, hi = LEVER_RANGES[c]
    if c in LOG_LEVERS:
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (v - lo) / (hi - lo)


order = ALL_LEVERS
ypos = np.arange(len(order))[::-1]
base_n = [norm_lever(c, baseline[c]) for c in order]
opt_n = [norm_lever(c, top_recipe[c]) for c in order]
for y, b, o in zip(ypos, base_n, opt_n):
    axD.plot([b, o], [y, y], color="grey", lw=2, alpha=0.6, zorder=1)
axD.scatter(base_n, ypos, color="white", edgecolors="black", s=90, zorder=3, label="Current typical practice")
axD.scatter(opt_n, ypos, color="#83AE8F", edgecolors="black", s=90, zorder=3, label="TOPSIS-optimal recipe")
for y, c in zip(ypos, order):
    axD.text(1.04, y, f"{baseline[c]:.2g} -> {top_recipe[c]:.2g}", va="center", fontsize=8.5, color="grey")
axD.set_yticks(ypos)
axD.set_yticklabels([SHORT_LEVER_LABELS.get(c, LEVER_LABELS.get(c, c)) for c in order], fontsize=10)
axD.set_xlim(-0.05, 1.48)
axD.set_xticks([0, 0.5, 1.0])
axD.set_xlabel("Normalised lever value", fontsize=10.8)
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
axD.spines["left"].set_visible(False)
axD.axvline(0, color="grey", lw=0.5)
axD.axvline(1, color="grey", lw=0.5, ls=":")
axD.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Current typical practice"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#83AE8F",
               markeredgecolor="black", markersize=8, label="TOPSIS-optimal recipe"),
    ],
    loc="upper center", bbox_to_anchor=(0.55, -0.17), ncol=2, frameon=False, fontsize=8.6
)
axD.set_title("D   Recommended management adjustments", fontsize=12, fontweight="bold", loc="left")

# ============ Panel E: sensitivity to composting period ============
axE = fig.add_subplot(gs[2, 0])
period_lo, period_hi = LEVERS['Period (d)']
period_grid = np.linspace(period_lo, period_hi, 60)
sweep = pd.DataFrame({c: top_recipe[c] for c in COMMON_FEATURES}, index=range(len(period_grid)))
sweep['Period (d)'] = period_grid
sweep_pred = predict_all(sweep)
sweep_desir = desir_from_pred(sweep_pred)
sweep_N = sweep_desir[['NH3-N loss', 'N2O-N loss', 'TN loss']].mean(axis=1)
sweep_C = sweep_desir[['CH4-C loss', 'CO2-C loss', 'TC loss']].mean(axis=1)
sweep_Q = sweep_desir['Final GI']
sweep_T = topsis_from_desir(sweep_desir[TARGET_ORDER].values)

axE.plot(period_grid, sweep_N, color="#C97979", lw=2, label="N-conservation score")
axE.plot(period_grid, sweep_C, color="#777FBC", lw=2, label="C/GHG-mitigation score")
axE.plot(period_grid, sweep_Q, color="#83AE8F", lw=2, label="Quality score")
axE.plot(period_grid, sweep_T, color="black", lw=2, ls="--", label="TOPSIS score")
axE.axvline(top_recipe['Period (d)'], color="#E2B36E", lw=2,
            label=f"TOPSIS-optimal period ({top_recipe['Period (d)']:.0f} d)")
axE.set_xlabel("Composting period (d)", fontsize=10.8)
axE.set_ylabel("Score (0-1)", fontsize=10.5)
axE.set_ylim(0, 1)
axE.spines["top"].set_visible(False)
axE.spines["right"].set_visible(False)
axE.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False,
           fontsize=9, ncol=3, columnspacing=0.85, handletextpad=0.42)
axE.set_title("E   Sensitivity to composting period", fontsize=12, fontweight="bold", loc="left")

# ============ Panel F: 2-D design map (period x turning frequency) ============
axF = fig.add_subplot(gs[2, 1])
turn_lo, turn_hi = LEVERS['Turning times']
g1 = np.linspace(period_lo, period_hi, 35)
g2 = np.linspace(turn_lo, turn_hi, 35)
G1, G2 = np.meshgrid(g1, g2)
grid_df = pd.DataFrame({c: top_recipe[c] for c in COMMON_FEATURES}, index=range(G1.size))
grid_df['Period (d)'] = G1.ravel()
grid_df['Turning times'] = G2.ravel()
grid_pred = predict_all(grid_df)
grid_desir = desir_from_pred(grid_pred)
grid_T = topsis_from_desir(grid_desir[TARGET_ORDER].values).reshape(G1.shape)

im = axF.pcolormesh(G1, G2, grid_T, cmap=CMAP_SEQ, shading="gouraud", vmin=grid_T.min(), vmax=grid_T.max())
cs = axF.contour(G1, G2, grid_T, colors="white", linewidths=0.6, alpha=0.6, levels=6)
axF.scatter([top_recipe['Period (d)']], [top_recipe['Turning times']], marker="*", s=380,
             color="#E2B36E", edgecolors="black", linewidths=1.2, zorder=5, label="TOPSIS-optimal recipe")
axF.scatter([baseline['Period (d)']], [baseline['Turning times']], marker="s", s=120,
             color="white", edgecolors="black", linewidths=1.4, zorder=5, label="Current typical practice")
caxF = axF.inset_axes([1.025, 0.16, 0.035, 0.68])
cbarF = fig.colorbar(im, cax=caxF, orientation="vertical")
cbarF.set_label("TOPSIS score", fontsize=9.5, labelpad=7)
cbarF.ax.tick_params(labelsize=9.2, length=3.4, width=1.0)
axF.set_xlabel("Composting period (d)", fontsize=10.5)
axF.set_ylabel("Turning frequency (times)", fontsize=10.5)
axF.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False,
           fontsize=9, ncol=1, columnspacing=0.85, handletextpad=0.42)
axF.set_title("F   Design map: period x turning frequency", fontsize=12, fontweight="bold", loc="left")

polish_figure(fig)
save_combined_figure(fig, OUT_DIR, "Fig6_multiobjective_topsis")


def draw_panel_a(panel_fig):
    ax = panel_fig.add_subplot(111)
    sc_local = ax.scatter(sample['N_score'], sample['C_score'], c=sample['Q_score'], cmap=CMAP_CONT,
                          s=14, alpha=0.55, edgecolors="none", vmin=0, vmax=1)
    ax.scatter(sample.loc[highlight, 'N_score'], sample.loc[highlight, 'C_score'],
               facecolors="none", edgecolors="#C97979", linewidths=0.6, s=24, alpha=0.8,
               label="Top 5% TOPSIS score")
    ax.scatter([top_recipe['N_score']], [top_recipe['C_score']], marker="*", s=420,
               color="#E2B36E", edgecolors="black", linewidths=1.2, zorder=5,
               label="TOPSIS-optimal recipe")
    ax.scatter([baseline_N_score], [baseline_C_score], marker="s", s=140, color="white",
               edgecolors="black", linewidths=1.4, zorder=5, label="Current typical practice")
    cax = ax.inset_axes([0.14, 1.055, 0.72, 0.045])
    cb = panel_fig.colorbar(sc_local, cax=cax, orientation="horizontal")
    cb.set_label("Final GI desirability", fontsize=9.8)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=9.4, length=3.4, width=1.0)
    ax.set_xlabel("N-conservation score", fontsize=11.2)
    ax.set_ylabel("C/GHG-mitigation score", fontsize=11.2)
    ax.legend(loc="lower left", frameon=True, framealpha=0.85, edgecolor="none", fontsize=8.8)
    return [ax, cb.ax]


def draw_panel_b(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.hist(sample['TOPSIS'], bins=40, color="#78BFC2", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(top_decile, color="#C97979", lw=1.6, ls="--", label="Top 5% threshold")
    ax.axvline(baseline_topsis, color="black", lw=1.6, ls="-", label="Current typical practice")
    ax.axvline(top_recipe['TOPSIS'], color="#E2B36E", lw=2.0, ls="-", label="TOPSIS-optimal recipe")
    ax.set_xlabel("TOPSIS score", fontsize=11.2)
    ax.set_ylabel("Number of sampled recipes", fontsize=10.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9.2)
    return ax


def draw_panel_c(panel_fig):
    ax = panel_fig.add_subplot(111, polar=True)
    for label, (vals, color) in series.items():
        vals2 = vals + vals[:1]
        ax.plot(angles, vals2, color=color, linewidth=2, label=label, alpha=0.9,
                ls="-" if label != "Current typical practice" else "--")
        ax.fill(angles, vals2, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER], fontsize=9.8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5, color="grey")
    ax.set_rlabel_position(15)
    ax.spines["polar"].set_color("grey")
    ax.grid(color="grey", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False,
              fontsize=9.2, ncol=2, columnspacing=1.0, handletextpad=0.45,
              labelspacing=0.35)
    return ax


def draw_panel_d(panel_fig):
    ax = panel_fig.add_subplot(111)
    for y, b, o in zip(ypos, base_n, opt_n):
        ax.plot([b, o], [y, y], color="grey", lw=2, alpha=0.6, zorder=1)
    ax.scatter(base_n, ypos, color="white", edgecolors="black", s=90, zorder=3)
    ax.scatter(opt_n, ypos, color="#83AE8F", edgecolors="black", s=90, zorder=3)
    for y, c in zip(ypos, order):
        ax.text(1.04, y, f"{baseline[c]:.2g} -> {top_recipe[c]:.2g}", va="center", fontsize=8.8, color="grey")
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_LEVER_LABELS.get(c, LEVER_LABELS.get(c, c)) for c in order], fontsize=10.4)
    ax.set_xlim(-0.05, 1.48)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Normalised lever value", fontsize=11.0)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color="grey", lw=0.5)
    ax.axvline(1, color="grey", lw=0.5, ls=":")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="black", markersize=8, label="Current typical practice"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#83AE8F",
                   markeredgecolor="black", markersize=8, label="TOPSIS-optimal recipe"),
        ],
        loc="upper center", bbox_to_anchor=(0.55, -0.17), ncol=2, frameon=False, fontsize=8.8
    )
    return ax


def draw_panel_e(panel_fig):
    ax = panel_fig.add_subplot(111)
    ax.plot(period_grid, sweep_N, color="#C97979", lw=2, label="N-conservation score")
    ax.plot(period_grid, sweep_C, color="#777FBC", lw=2, label="C/GHG-mitigation score")
    ax.plot(period_grid, sweep_Q, color="#83AE8F", lw=2, label="Quality score")
    ax.plot(period_grid, sweep_T, color="black", lw=2, ls="--", label="TOPSIS score")
    ax.axvline(top_recipe['Period (d)'], color="#E2B36E", lw=2,
               label=f"TOPSIS-optimal period ({top_recipe['Period (d)']:.0f} d)")
    ax.set_xlabel("Composting period (d)", fontsize=11.0)
    ax.set_ylabel("Score (0-1)", fontsize=10.8)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9.2, ncol=2)
    return ax


def draw_panel_f(panel_fig):
    ax = panel_fig.add_subplot(111)
    im_local = ax.pcolormesh(G1, G2, grid_T, cmap=CMAP_SEQ, shading="gouraud",
                             vmin=grid_T.min(), vmax=grid_T.max())
    ax.contour(G1, G2, grid_T, colors="white", linewidths=0.6, alpha=0.6, levels=6)
    ax.scatter([top_recipe['Period (d)']], [top_recipe['Turning times']], marker="*", s=380,
               color="#E2B36E", edgecolors="black", linewidths=1.2, zorder=5, label="TOPSIS-optimal recipe")
    ax.scatter([baseline['Period (d)']], [baseline['Turning times']], marker="s", s=120,
               color="white", edgecolors="black", linewidths=1.4, zorder=5, label="Current typical practice")
    cax = ax.inset_axes([0.16, 1.055, 0.68, 0.045])
    cb = panel_fig.colorbar(im_local, cax=cax, orientation="horizontal")
    cb.set_label("TOPSIS score", fontsize=9.8)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=9.4, length=3.4, width=1.0)
    ax.set_xlabel("Composting period (d)", fontsize=10.8)
    ax.set_ylabel("Turning frequency (times)", fontsize=10.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9.2, ncol=2)
    return [ax, cb.ax]


save_independent_subfigures(OUT_DIR, "Fig6_multiobjective_topsis", [
    ("a", (6.6, 5.7), draw_panel_a, None),
    ("b", (5.9, 5.0), draw_panel_b, None),
    ("c", (5.6, 5.6), draw_panel_c, None),
    ("d", (6.5, 5.9), draw_panel_d, None),
    ("e", (6.1, 5.0), draw_panel_e, None),
    ("f", (6.1, 5.0), draw_panel_f, None),
])
print("Saved Fig7.")

# ============ Export tables for the manuscript ============
top10 = sample.sort_values("TOPSIS", ascending=False).head(10)
export_cols = ALL_LEVERS + ["N_score", "C_score", "Q_score", "TOPSIS"] + [f"pred_{t}" for t in TARGET_ORDER]
top10[export_cols].round(3).to_csv(os.path.join(OUT_DIR, "Table_topsis_top10_recipes.csv"), index=False)

compare = pd.DataFrame({
    "Lever": [LEVER_LABELS.get(c, c) for c in ALL_LEVERS],
    "Typical practice": [baseline[c] for c in ALL_LEVERS],
    "TOPSIS-optimal recipe": [top_recipe[c] for c in ALL_LEVERS],
})
compare.round(3).to_csv(os.path.join(OUT_DIR, "Table_baseline_vs_optimal_levers.csv"), index=False)

endpoint_compare = pd.DataFrame({
    "Endpoint": TARGET_ORDER,
    "Typical practice (predicted)": [baseline_pred[t].iloc[0] for t in TARGET_ORDER],
    "TOPSIS-optimal (predicted)": [top_recipe[f"pred_{t}"] for t in TARGET_ORDER],
    "Typical practice (desirability)": [baseline_desir[t] for t in TARGET_ORDER],
    "TOPSIS-optimal (desirability)": [top_recipe[f"desir_{t}"] for t in TARGET_ORDER],
})
endpoint_compare.round(3).to_csv(os.path.join(OUT_DIR, "Table_baseline_vs_optimal_endpoints.csv"), index=False)
print("Saved Fig7 tables.")
