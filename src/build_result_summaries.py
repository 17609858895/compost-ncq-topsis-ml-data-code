from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))

from config import RESULTS_DIR, TARGETS, TARGET_ORDER  # noqa: E402


def main() -> None:
    results = Path(RESULTS_DIR)
    metrics_frames = []
    shap_frames = []
    corr_frames = []
    summary_rows = []

    for target in TARGET_ORDER:
        _, _, code = TARGETS[target]

        metrics_path = results / f"metrics_{code}.csv"
        shap_path = results / f"shapimp_{code}.csv"
        corr_path = results / f"corr_{code}.csv"
        summary_path = results / f"summary_{code}.json"

        missing = [p for p in [metrics_path, shap_path, corr_path, summary_path] if not p.exists()]
        if missing:
            names = ", ".join(str(p.relative_to(ROOT)) for p in missing)
            raise FileNotFoundError(
                f"Missing model-output files for {target}: {names}. "
                "Run `python src/run_model_pipeline.py` first."
            )

        metrics_frames.append(pd.read_csv(metrics_path))
        shap_frames.append(pd.read_csv(shap_path))
        corr_frames.append(pd.read_csv(corr_path))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary_rows.append(json.load(handle))

    pd.concat(metrics_frames, ignore_index=True).to_csv(results / "model_performance.csv", index=False)
    pd.concat(shap_frames, ignore_index=True).to_csv(results / "shap_importance.csv", index=False)
    pd.concat(corr_frames, ignore_index=True).to_csv(results / "feature_target_corr.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(results / "dataset_summary.csv", index=False)
    print(f"Wrote summary tables to {results.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
