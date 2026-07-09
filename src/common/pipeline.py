"""
Core modeling pipeline for the 7-target nutrient-carbon-quality nexus analysis.
Trains ML algorithms per target (with checkpointing so each can run across
multiple short calls), evaluates performance, and fits a tuned Random Forest
'interpretation model' per target for SHAP analysis.
"""
import os, json, warnings, time, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import TARGETS, COMMON_FEATURES, TARGET_ORDER, RESULTS_DIR

RANDOM_STATE = 42
N_ITER = 6
CV = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

def get_models():
    models = {}
    models["RandomForest"] = (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1,2,4]})
    models["ExtraTrees"] = (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1,2,4]})
    models["GradientBoosting"] = (GradientBoostingRegressor(random_state=RANDOM_STATE), {
        "n_estimators": [100, 150], "max_depth": [2,3], "learning_rate": [0.05,0.1]})
    models["HistGradientBoosting"] = (HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
        "max_iter": [100,150], "max_depth": [None,3,6], "learning_rate": [0.05,0.1]})
    models["Ridge"] = (Pipeline([("sc", StandardScaler()), ("m", Ridge())]), {
        "m__alpha": [0.01,0.1,1,10,100]})
    models["ElasticNet"] = (Pipeline([("sc", StandardScaler()), ("m", ElasticNet(max_iter=5000))]), {
        "m__alpha": [0.001,0.01,0.1,1], "m__l1_ratio": [0.1,0.5,0.9]})
    models["MLP"] = (Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(max_iter=600, random_state=RANDOM_STATE))]), {
        "m__hidden_layer_sizes": [(32,), (64,), (32,16)], "m__alpha": [0.0001,0.001,0.01]})
    models["SVR"] = (Pipeline([("sc", StandardScaler()), ("m", SVR(max_iter=20000))]), {
        "m__C": [1,10,50], "m__gamma": ["scale","auto"], "m__epsilon": [0.05,0.2]})
    return models

def evaluate(model, Xtr, ytr, Xte, yte):
    pred_tr = model.predict(Xtr)
    pred_te = model.predict(Xte)
    return {
        "R2_train": r2_score(ytr, pred_tr),
        "R2_test": r2_score(yte, pred_te),
        "RMSE_train": np.sqrt(mean_squared_error(ytr, pred_tr)),
        "RMSE_test": np.sqrt(mean_squared_error(yte, pred_te)),
        "MAE_test": mean_absolute_error(yte, pred_te),
    }

def run_target_step(target_name, time_budget=35):
    """Train models for one target, checkpointing progress. Returns True when fully done."""
    fpath, ycol, code = TARGETS[target_name]
    df = pd.read_csv(fpath)
    y = df[ycol].values
    X = df.drop(columns=[ycol])
    feat_names = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    ckpt_path = os.path.join(CKPT_DIR, f"{code}.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
    else:
        state = {"metrics": [], "fitted": {}, "done_models": []}

    t0 = time.time()
    for name, (est, grid) in get_models().items():
        if name in state["done_models"]:
            continue
        if time.time() - t0 > time_budget:
            break
        n_iter = min(N_ITER, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=CV, scoring="r2",
                                     random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        m = evaluate(best, Xtr, ytr, Xte, yte)
        m.update({"Target": target_name, "Model": name, "CV_R2": search.best_score_})
        state["metrics"].append(m)
        state["fitted"][name] = best
        state["done_models"].append(name)
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp_path, ckpt_path)
        print(f"  [{code}] done model: {name} (R2_test={m['R2_test']:.3f}) t={time.time()-t0:.1f}s", flush=True)

    if len(state["done_models"]) < len(get_models()):
        print(f"  [{code}] progress {len(state['done_models'])}/{len(get_models())} -- rerun to continue", flush=True)
        return False

    # ---- All models done: finalize (SHAP + correlations) ----
    metrics_df = pd.DataFrame(state["metrics"])
    rf = state["fitted"]["RandomForest"]
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values_all = explainer.shap_values(X)
    shap_importance = np.abs(shap_values_all).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": feat_names, "MeanAbsSHAP": shap_importance})
    imp_df["Importance_pct"] = 100 * imp_df["MeanAbsSHAP"] / imp_df["MeanAbsSHAP"].sum()
    imp_df = imp_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    imp_df["Target"] = target_name

    corr_rows = []
    for f in COMMON_FEATURES:
        if f in df.columns:
            rho, p = spearmanr(df[f], df[ycol])
            corr_rows.append({"Feature": f, "Target": target_name, "rho": rho, "p": p})
    corr_df = pd.DataFrame(corr_rows)

    common_idx = [feat_names.index(f) for f in COMMON_FEATURES if f in feat_names]
    np.savez(os.path.join(RESULTS_DIR, f"shap_{code}.npz"),
             shap_values=shap_values_all[:, common_idx],
             X=X[[f for f in COMMON_FEATURES if f in feat_names]].values,
             feature_names=np.array([f for f in COMMON_FEATURES if f in feat_names]),
             y=y, X_full=X.values, feat_names_full=np.array(feat_names),
             shap_values_full=shap_values_all)

    metrics_df.to_csv(os.path.join(RESULTS_DIR, f"metrics_{code}.csv"), index=False)
    imp_df.to_csv(os.path.join(RESULTS_DIR, f"shapimp_{code}.csv"), index=False)
    corr_df.to_csv(os.path.join(RESULTS_DIR, f"corr_{code}.csv"), index=False)
    summ = {"n_samples": len(df), "n_features": len(feat_names), "code": code, "Target": target_name}
    with open(os.path.join(RESULTS_DIR, f"summary_{code}.json"), "w") as f:
        json.dump(summ, f)
    print("FULLY DONE:", target_name)
    return True

if __name__ == "__main__":
    target = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    run_target_step(target, time_budget=budget)
