from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

FIGURE_SCRIPTS = [
    ROOT / "src" / "figures" / "Fig1_dataset_structure" / "fig2_distribution_correlation.py",
    ROOT / "src" / "figures" / "Fig2_model_performance" / "fig3_model_performance.py",
    ROOT / "src" / "figures" / "Fig3_shap_importance" / "fig4_shap_importance.py",
    ROOT / "src" / "figures" / "Fig4_synergy_tradeoff" / "fig5_tradeoff_network.py",
    ROOT / "src" / "figures" / "Fig5_response_interaction" / "fig6_dependence_interaction.py",
    ROOT / "src" / "figures" / "Fig6_multiobjective_topsis" / "fig7_multiobjective_topsis.py",
    ROOT / "src" / "figures" / "Fig7_decision_matrix" / "fig8_decision_matrix.py",
    ROOT / "src" / "figures" / "Fig9a_weight_sensitivity_applicability" / "fig9a_weight_sensitivity_applicability_recovery_note_source.py",
    ROOT / "src" / "figures" / "Fig9b_single_objective_comparison" / "fig9b_single_objective_comparison_recovery_note_source.py",
    ROOT / "src" / "figures" / "Fig8_decision_robustness" / "fig9_decision_robustness.py",
]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / "build_result_summaries.py")], cwd=ROOT, check=True)
    for script in FIGURE_SCRIPTS:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
