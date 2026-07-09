import os
import pickle

import numpy as np
import pandas as pd

from config import (
    COMMON_FEATURES,
    FEATURE_LABELS,
    RESULTS_DIR,
    TARGETS,
    TARGET_DESIRABLE_DIRECTION,
    TARGET_ORDER,
)


CATEGORICAL = [
    "Material_Main",
    "Material_2",
    "Material_3",
    "Additive_1",
    "Additive_2",
    "Additive_3",
    "Additive_4",
    " Composting Method",
]

LEVERS = {
    "Period (d)": (15.0, 133.0),
    "Turning times": (0.0, 54.0),
    "Initial C/N (%)": (9.7, 43.0),
    "Initial Moisture Content (%)": (50.0, 75.8),
    "V5_Ventilation rate (L/min/kg iniDW)": (0.0, 7.32),
    "Application Rate (%DW)": (0.0, 29.0),
    "V4_Ventilation Day": (0.0, 100.0),
}
LOG_LEVERS = {"Compost volume (m3)": (0.002, 11.25)}
BINARY_LEVERS = ["M1_is Enclosed"]
CAT_LEVERS = {"V1_Ventilation Type": [0, 1, 2]}

ALL_LEVERS = (
    list(LEVERS.keys())
    + list(LOG_LEVERS.keys())
    + BINARY_LEVERS
    + list(CAT_LEVERS.keys())
)

LEVER_RANGES = {
    **LEVERS,
    **LOG_LEVERS,
    "M1_is Enclosed": (0.0, 1.0),
    "V1_Ventilation Type": (0.0, 2.0),
}

SHORT_LEVER_LABELS = {
    "Period (d)": "Period",
    "Turning times": "Turning",
    "Initial C/N (%)": "C/N ratio",
    "Initial Moisture Content (%)": "Moisture",
    "V5_Ventilation rate (L/min/kg iniDW)": "Aer. rate",
    "Application Rate (%DW)": "Bulking rate",
    "V4_Ventilation Day": "Aer. onset",
    "Compost volume (m3)": "Pile volume",
    "M1_is Enclosed": "Enclosed",
    "V1_Ventilation Type": "Aeration type",
}

OBJECTIVE_GROUPS = {
    "N conservation": ["NH3-N loss", "N2O-N loss", "TN loss"],
    "C/GHG mitigation": ["CH4-C loss", "CO2-C loss", "TC loss"],
    "Quality": ["Final GI"],
}


def load_context():
    perf = pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv"))
    best_model_name = (
        perf.loc[perf.groupby("Target")["R2_test"].idxmax()]
        .set_index("Target")["Model"]
    )

    fitted_models, feat_orders, extra_baseline = {}, {}, {}
    ckpt_dir = os.path.join(RESULTS_DIR, "ckpt")
    frames = []

    for name in TARGET_ORDER:
        fpath, ycol, code = TARGETS[name]
        df = pd.read_csv(fpath)
        X = df.drop(columns=[ycol])
        feat_orders[name] = list(X.columns)
        frames.append(df[COMMON_FEATURES])

        with open(os.path.join(ckpt_dir, f"{code}.pkl"), "rb") as f:
            state = pickle.load(f)
        fitted_models[name] = state["fitted"][best_model_name[name]]

        for col in X.columns:
            if col not in COMMON_FEATURES:
                extra_baseline.setdefault(col, {})[name] = float(X[col].median())

    pooled = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    baseline = {}
    for col in COMMON_FEATURES:
        if col in CATEGORICAL:
            baseline[col] = float(pooled[col].mode().iloc[0])
        else:
            baseline[col] = float(pooled[col].median())

    return {
        "best_model_name": best_model_name,
        "fitted_models": fitted_models,
        "feat_orders": feat_orders,
        "extra_baseline": extra_baseline,
        "pooled": pooled,
        "baseline": baseline,
    }


def sample_design_space(context, n_samples=4000, seed=42):
    rng = np.random.default_rng(seed)
    baseline = context["baseline"]
    sample = pd.DataFrame({col: baseline[col] for col in COMMON_FEATURES}, index=range(n_samples))

    for col, (lo, hi) in LEVERS.items():
        sample[col] = rng.uniform(lo, hi, n_samples)
    for col, (lo, hi) in LOG_LEVERS.items():
        sample[col] = np.exp(rng.uniform(np.log(lo), np.log(hi), n_samples))
    for col in BINARY_LEVERS:
        sample[col] = rng.integers(0, 2, n_samples).astype(float)
    for col, cats in CAT_LEVERS.items():
        sample[col] = rng.choice(cats, n_samples).astype(float)

    return sample


def predict_all(context, df_common):
    out = pd.DataFrame(index=df_common.index)
    for name in TARGET_ORDER:
        cols = context["feat_orders"][name]
        Xp = pd.DataFrame(index=df_common.index, columns=cols, dtype=float)
        for col in cols:
            if col in COMMON_FEATURES:
                Xp[col] = df_common[col].values
            else:
                Xp[col] = context["extra_baseline"][col][name]
        out[name] = context["fitted_models"][name].predict(Xp[cols])
    return out


def fit_desirability_scalers(preds):
    return {name: (float(preds[name].min()), float(preds[name].max())) for name in TARGET_ORDER}


def desirability_from_predictions(preds, scalers):
    desir = pd.DataFrame(index=preds.index)
    for name in TARGET_ORDER:
        vmin, vmax = scalers[name]
        norm = (preds[name].values - vmin) / (vmax - vmin + 1e-12)
        norm = np.clip(norm, 0, 1)
        desir[name] = norm if TARGET_DESIRABLE_DIRECTION[name] == 1 else 1 - norm
    return desir


def objective_scores(desir):
    return pd.DataFrame(
        {
            "N_score": desir[OBJECTIVE_GROUPS["N conservation"]].mean(axis=1),
            "C_score": desir[OBJECTIVE_GROUPS["C/GHG mitigation"]].mean(axis=1),
            "Q_score": desir[OBJECTIVE_GROUPS["Quality"]].mean(axis=1),
        },
        index=desir.index,
    )


def endpoint_weights_from_objective_weights(n_weight, c_weight, q_weight):
    weights = pd.Series(0.0, index=TARGET_ORDER)
    for endpoint in OBJECTIVE_GROUPS["N conservation"]:
        weights[endpoint] = n_weight / len(OBJECTIVE_GROUPS["N conservation"])
    for endpoint in OBJECTIVE_GROUPS["C/GHG mitigation"]:
        weights[endpoint] = c_weight / len(OBJECTIVE_GROUPS["C/GHG mitigation"])
    for endpoint in OBJECTIVE_GROUPS["Quality"]:
        weights[endpoint] = q_weight / len(OBJECTIVE_GROUPS["Quality"])
    total = float(weights.sum())
    return (weights / total).loc[TARGET_ORDER].values


def equal_endpoint_objective_weights():
    n = len(OBJECTIVE_GROUPS["N conservation"]) / len(TARGET_ORDER)
    c = len(OBJECTIVE_GROUPS["C/GHG mitigation"]) / len(TARGET_ORDER)
    q = len(OBJECTIVE_GROUPS["Quality"]) / len(TARGET_ORDER)
    return np.array([n, c, q], dtype=float)


def topsis_scores(desir, endpoint_weights=None):
    d = np.asarray(desir[TARGET_ORDER] if isinstance(desir, pd.DataFrame) else desir, dtype=float)
    if endpoint_weights is None:
        endpoint_weights = np.full(d.shape[1], 1.0 / d.shape[1])
    weights = np.asarray(endpoint_weights, dtype=float)
    weights = weights / weights.sum()
    dist_pos = np.sqrt(((d - 1.0) ** 2 * weights).sum(axis=1))
    dist_neg = np.sqrt((d**2 * weights).sum(axis=1))
    return dist_neg / (dist_pos + dist_neg + 1e-12)


def score_candidate_table(sample, preds, desir):
    scored = sample.copy()
    obj = objective_scores(desir)
    for col in obj.columns:
        scored[col] = obj[col]
    scored["TOPSIS"] = topsis_scores(desir)
    for name in TARGET_ORDER:
        scored[f"pred_{name}"] = preds[name]
        scored[f"desir_{name}"] = desir[name]
    return scored


def baseline_profiles(context, scalers):
    baseline_df = pd.DataFrame([context["baseline"]])
    baseline_pred = predict_all(context, baseline_df)
    baseline_desir = desirability_from_predictions(baseline_pred, scalers)
    baseline_scores = objective_scores(baseline_desir)
    baseline_scores["TOPSIS"] = topsis_scores(baseline_desir)
    return baseline_pred, baseline_desir, baseline_scores


def build_analysis(n_samples=4000, seed=42):
    context = load_context()
    sample = sample_design_space(context, n_samples=n_samples, seed=seed)
    preds = predict_all(context, sample)
    scalers = fit_desirability_scalers(preds)
    desir = desirability_from_predictions(preds, scalers)
    scored = score_candidate_table(sample, preds, desir)
    baseline_pred, baseline_desir, baseline_scores = baseline_profiles(context, scalers)
    return context, sample, preds, desir, scored, baseline_pred, baseline_desir, baseline_scores, scalers


def normalized_lever_matrix(df):
    arr = []
    for col in ALL_LEVERS:
        vals = np.asarray(df[col], dtype=float)
        lo, hi = LEVER_RANGES[col]
        if col in LOG_LEVERS:
            vals = np.log(np.clip(vals, lo, hi))
            lo, hi = np.log(lo), np.log(hi)
        arr.append((vals - lo) / (hi - lo + 1e-12))
    return np.column_stack(arr)


def normalized_lever_series(row):
    return pd.Series(normalized_lever_matrix(pd.DataFrame([row]))[0], index=ALL_LEVERS)


def nearest_observed_distances(candidates, observed, chunk_size=256):
    cand_x = normalized_lever_matrix(candidates)
    obs_x = normalized_lever_matrix(observed)
    nearest = np.empty(cand_x.shape[0], dtype=float)
    nearest_idx = np.empty(cand_x.shape[0], dtype=int)
    for start in range(0, cand_x.shape[0], chunk_size):
        end = min(start + chunk_size, cand_x.shape[0])
        diff = cand_x[start:end, None, :] - obs_x[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        nearest[start:end] = dist.min(axis=1)
        nearest_idx[start:end] = dist.argmin(axis=1)
    return nearest, nearest_idx


def lever_label(col):
    return FEATURE_LABELS.get(col, col)
