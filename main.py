"""
================================================================================
MAIN PIPELINE
================================================================================
Hybrid Quantum-Classical Optimization using QAOA for Medical Feature Selection

End-to-end pipeline:
  Phase 1: QAOA Feature Selection (Steps 1-5)
  Phase 2: Dataset + ML Integration (Steps 6-9)
  Phase 3: Classical Baselines (Steps 10-11)
  Phase 4: Visualization (Step 12)
  Phase 5: System Design (Steps 13-14) — see docs/architecture.md
  Phase 6: Research Output (Step 15)

Usage:
  python main.py
================================================================================
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import load_medical_dataset, preprocess_data, split_data
from src.qaoa.feature_selector import QAOAFeatureSelector
from src.qaoa.hamiltonian import compute_relevance_scores
from src.ml.classifier import compare_feature_sets
from src.ml.baselines import run_all_baselines
from src.visualization.plots import (
    plot_comprehensive_results,
    plot_correlation_heatmap,
)
from src.research.report import generate_research_report


def main():
    """Run the complete QAOA feature selection pipeline."""

    total_start = time.time()

    print("\n" + "█" * 70)
    print("█  HYBRID QUANTUM-CLASSICAL OPTIMIZATION USING QAOA")
    print("█  FOR MEDICAL FEATURE SELECTION")
    print("█" * 70)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2, Step 6: Load and preprocess dataset
    # (We do data loading first to inform the QAOA setup)
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 2 — STEP 6: DATASET LOADING & PREPROCESSING")
    print("═" * 70)

    X, y, feature_names, target_names = load_medical_dataset()
    X_normalized, scaler = preprocess_data(X, feature_names)
    X_train, X_test, y_train, y_test = split_data(X_normalized, y)

    # Compute full relevance for visualization later
    full_relevance = compute_relevance_scores(X_normalized, y)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1, Steps 1-5: QAOA Feature Selection
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 1 — STEPS 1-5: QAOA FEATURE SELECTION")
    print("═" * 70)

    qaoa_selector = QAOAFeatureSelector(
        n_candidates=10,     # Pre-filter to top-10 by MI
        lambda_param=0.5,    # Relevance-redundancy trade-off
        p=1,                 # QAOA depth
        shots=1024,          # Measurement shots
        optimizer_method="COBYLA",
        maxiter=150,
        random_state=42,
    )

    qaoa_selector.fit(X_train, y_train, feature_names)
    qaoa_indices = qaoa_selector.get_selected_features()
    qaoa_summary = qaoa_selector.get_summary()
    convergence = qaoa_selector.get_convergence_history()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3, Steps 10-11: Classical Baselines
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 3 — STEPS 10-11: CLASSICAL BASELINES")
    print("═" * 70)

    baselines = run_all_baselines(X_train, X_test, y_train, y_test)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2, Steps 7-9: ML Training & Evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 2 — STEPS 7-9: ML TRAINING & EVALUATION")
    print("═" * 70)

    # Prepare feature selection dictionary
    feature_selections = {
        "All Features": None,
        "QAOA": qaoa_indices,
    }

    # Add baseline selections (for those that return indices)
    for name, baseline in baselines.items():
        if baseline.get("indices") is not None:
            feature_selections[name] = baseline["indices"]

    # Compare all methods
    all_ml_results = compare_feature_sets(
        X_train, X_test, y_train, y_test,
        feature_selections, feature_names
    )

    # For PCA (creates new features, not a subset), evaluate separately
    if "PCA" in baselines and baselines["PCA"]["indices"] is None:
        from src.ml.classifier import train_and_evaluate
        pca_results = {}
        pca_data = baselines["PCA"]
        for model_name in ["logistic_regression", "random_forest"]:
            res = train_and_evaluate(
                pca_data["X_train"], pca_data["X_test"],
                y_train, y_test, model_name
            )
            pca_results[model_name] = res
        all_ml_results["PCA"] = pca_results

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4, Step 12: Visualization
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 4 — STEP 12: VISUALIZATION")
    print("═" * 70)

    # Feature counts for comparison
    feature_counts = {"All Features": len(feature_names), "QAOA": len(qaoa_indices)}
    time_dict = {"QAOA": qaoa_summary.get("computation_time", 0)}

    for name, baseline in baselines.items():
        feature_counts[name] = baseline["n_features"]
        time_dict[name] = baseline["time"]

    # Correlation heatmap
    plot_correlation_heatmap(
        X_normalized, feature_names,
        save_path="outputs/correlation_heatmap.png",
        max_features=15
    )

    # Comprehensive results
    plot_comprehensive_results(
        all_ml_results=all_ml_results,
        convergence_history=convergence["costs"],
        relevance_scores=full_relevance,
        feature_names=feature_names,
        selected_indices=qaoa_indices,
        feature_counts=feature_counts,
        time_dict=time_dict,
        save_dir="outputs",
    )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6, Step 15: Research Report
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 6 — STEP 15: RESEARCH REPORT")
    print("═" * 70)

    generate_research_report(
        qaoa_summary=qaoa_summary,
        all_ml_results=all_ml_results,
        baselines=baselines,
        feature_names=feature_names,
        save_path="docs/research_paper.md",
    )

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start

    print("\n\n" + "█" * 70)
    print("█  PIPELINE COMPLETE")
    print("█" * 70)
    print(f"\n  Total execution time: {total_time:.1f}s")
    print(f"\n  QAOA selected {len(qaoa_indices)} features from {len(feature_names)} total")
    print(f"  Selected feature indices: {qaoa_indices.tolist()}")
    print(f"  Selected feature names: {[feature_names[i] for i in qaoa_indices]}")
    print(f"\n  Generated outputs:")
    print(f"    📊 outputs/feature_importance.png")
    print(f"    📊 outputs/accuracy_comparison.png")
    print(f"    📊 outputs/qaoa_convergence.png")
    print(f"    📊 outputs/correlation_heatmap.png")
    print(f"    📊 outputs/feature_count_comparison.png")
    print(f"    📊 outputs/computation_time.png")
    print(f"    📄 docs/research_paper.md")
    print(f"\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    main()
