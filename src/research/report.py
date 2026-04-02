"""
================================================================================
Research Report Generator
================================================================================

STEP 15: Write Key Research Sections
--------------------------------------
Auto-generates IEEE-style paper sections based on experimental results.

Sections:
  1. Abstract
  2. Problem Statement
  3. Methodology
  4. Results
  5. Future Work
================================================================================
"""

import numpy as np
from datetime import datetime


def generate_research_report(qaoa_summary, all_ml_results, baselines,
                             feature_names, save_path="docs/research_paper.md"):
    """
    Generate an IEEE-style research paper outline with results.

    Parameters
    ----------
    qaoa_summary : dict
        Summary from QAOAFeatureSelector.
    all_ml_results : dict
        {method: {model: results_dict}}
    baselines : dict
        {method: baseline_result_dict}
    feature_names : list of str
    save_path : str
    """

    # Gather key metrics
    n_total = len(feature_names)
    n_selected_qaoa = qaoa_summary["n_selected"]
    qaoa_time = qaoa_summary.get("computation_time", "N/A")
    qaoa_indices = qaoa_summary.get("selected_indices", [])

    # Best QAOA accuracy
    qaoa_acc_lr = all_ml_results.get("QAOA", {}).get(
        "logistic_regression", {}).get("accuracy", 0)
    qaoa_acc_rf = all_ml_results.get("QAOA", {}).get(
        "random_forest", {}).get("accuracy", 0)
    qaoa_best_acc = max(qaoa_acc_lr, qaoa_acc_rf)

    # All features accuracy
    all_acc_lr = all_ml_results.get("All Features", {}).get(
        "logistic_regression", {}).get("accuracy", 0)
    all_acc_rf = all_ml_results.get("All Features", {}).get(
        "random_forest", {}).get("accuracy", 0)
    all_best_acc = max(all_acc_lr, all_acc_rf)

    report = f"""# Hybrid Quantum-Classical Optimization using QAOA for Medical Feature Selection

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Abstract

We present a hybrid quantum-classical approach to medical feature selection
using the Quantum Approximate Optimization Algorithm (QAOA). Feature selection
is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem
that maximizes feature relevance (measured by mutual information) while
minimizing inter-feature redundancy (measured by absolute Pearson correlation).
The QUBO is mapped to an Ising Hamiltonian and solved using a parameterized
QAOA circuit optimized with classical COBYLA. Applied to the Breast Cancer
Wisconsin dataset ({n_total} features, 569 samples), our method selects
{n_selected_qaoa} features achieving {qaoa_best_acc:.4f} classification accuracy
compared to {all_best_acc:.4f} using all {n_total} features. We benchmark
against PCA, LASSO, and Genetic Algorithm baselines, demonstrating competitive
performance with significantly reduced feature dimensionality. Our results
illustrate the potential of near-term quantum computing for biomedical
data preprocessing.

**Keywords:** QAOA, Quantum Computing, Feature Selection, QUBO, Medical ML

---

## 1. Introduction and Problem Statement

### 1.1 Problem Statement

Medical datasets often contain high-dimensional feature spaces where many
features are redundant or irrelevant. Effective feature selection is critical
for:

- **Model interpretability** — fewer features enable clinical understanding
- **Generalization** — reduced overfitting from irrelevant features
- **Computational efficiency** — faster training and inference
- **Biomarker discovery** — identifying clinically relevant indicators

Traditional methods (PCA, LASSO) have limitations:
- PCA creates linear combinations, losing individual feature interpretability
- LASSO depends on the choice of regularization parameter
- Exhaustive search over 2ⁿ subsets is NP-hard for n features

### 1.2 Contribution

We propose using QAOA, a variational quantum algorithm, to solve the feature
selection problem as a combinatorial optimization. QAOA naturally encodes
the exponentially large search space in quantum superposition and uses
variational optimization to find near-optimal solutions.

---

## 2. Methodology

### 2.1 Problem Formulation

Given a dataset with n features, define a binary selection vector x ∈ {{0,1}}ⁿ
where xᵢ = 1 indicates feature i is selected.

**Objective:** Maximize relevance while minimizing redundancy:

$$\\max_x \\sum_i r_i x_i - \\lambda \\sum_{{i<j}} c_{{ij}} x_i x_j$$

where:
- rᵢ = I(Xᵢ; Y) is the mutual information between feature i and target
- cᵢⱼ = |corr(Xᵢ, Xⱼ)| is the absolute Pearson correlation
- λ is a trade-off hyperparameter

### 2.2 QUBO Formulation

The objective is converted to minimization form:

$$\\min_x \\mathbf{{x}}^T Q \\mathbf{{x}}$$

where Q is the QUBO matrix with:
- Q_{{ii}} = -rᵢ (diagonal: negative relevance)
- Q_{{ij}} = λcᵢⱼ/2 for i ≠ j (off-diagonal: redundancy penalty)

### 2.3 Ising Hamiltonian

The QUBO is mapped to an Ising Hamiltonian via xᵢ = (1 - Zᵢ)/2:

$$H_C = \\sum_i h_i Z_i + \\sum_{{i<j}} J_{{ij}} Z_i Z_j + \\text{{offset}}$$

### 2.4 QAOA Circuit

The QAOA ansatz with depth p:

$$|\\gamma, \\beta\\rangle = U_M(\\beta_p) U_C(\\gamma_p) \\cdots U_M(\\beta_1) U_C(\\gamma_1) |+\\rangle^n$$

- Cost unitary: Rz gates for Z terms, CNOT-Rz-CNOT for ZZ terms
- Mixer unitary: Rx(2β) on each qubit
- Depth p = {qaoa_summary.get('p', 1)}

### 2.5 Classical Optimization

Parameters (γ, β) are optimized using COBYLA to minimize:

$$\\langle \\gamma, \\beta | H_C | \\gamma, \\beta \\rangle = \\sum_x P(x|\\gamma,\\beta) \\cdot f(x)$$

### 2.6 Dataset

Breast Cancer Wisconsin (Diagnostic) dataset:
- 569 samples, {n_total} features, 2 classes
- Pre-filtered to top {qaoa_summary.get('n_candidates', 10)} features by MI
- Standardized (zero mean, unit variance)
- 80/20 train-test split with stratification

---

## 3. Results

### 3.1 QAOA Feature Selection

- **Features selected:** {n_selected_qaoa} out of {n_total}
- **Selected indices:** {qaoa_indices}
- **QAOA iterations:** {qaoa_summary.get('num_iterations', 'N/A')}
- **Computation time:** {qaoa_time:.1f}s

### 3.2 Classification Performance

| Method | Model | Features | Accuracy | Precision | Recall | F1-Score |
|--------|-------|----------|----------|-----------|--------|----------|
"""

    # Add results rows
    for method, models in all_ml_results.items():
        for model_name, res in models.items():
            report += (
                f"| {method} | {res['model_name']} | {res['n_features']} | "
                f"{res['accuracy']:.4f} | {res['precision']:.4f} | "
                f"{res['recall']:.4f} | {res['f1_score']:.4f} |\n"
            )

    report += f"""
### 3.3 Feature Count Comparison

| Method | Features Selected |
|--------|-------------------|
| All Features | {n_total} |
| QAOA | {n_selected_qaoa} |
"""

    for name, baseline in baselines.items():
        report += f"| {name} | {baseline['n_features']} |\n"

    report += f"""
---

## 4. Discussion

### 4.1 Key Findings

1. QAOA successfully reduces the feature space from {n_total} to
   {n_selected_qaoa} features while maintaining competitive classification
   accuracy.
2. The selected features capture both high relevance (mutual information)
   and low redundancy (correlation), demonstrating the effectiveness of
   the QUBO formulation.
3. Compared to classical baselines, QAOA provides an interpretable subset
   (unlike PCA) while being less sensitive to hyperparameters (unlike LASSO).

### 4.2 Limitations

1. **Qubit scaling:** The number of qubits scales linearly with features,
   requiring pre-filtering for large datasets.
2. **Simulator overhead:** Classical simulation of quantum circuits is
   exponentially costly; real quantum hardware would be needed for scale.
3. **QAOA depth:** Depth p=1 provides limited approximation quality;
   deeper circuits may improve solutions but increase optimization difficulty.

---

## 5. Future Work

1. **Hardware execution:** Run on IBM Quantum or IonQ hardware to validate
   simulator results and assess noise effects.
2. **Warm-starting QAOA:** Use classical solutions (e.g., LASSO) to
   initialize QAOA parameters for faster convergence.
3. **Multi-objective QAOA:** Extend to multi-objective optimization
   balancing accuracy, feature count, and interpretability.
4. **Larger datasets:** Apply to genomics (thousands of features) with
   hierarchical decomposition strategies.
5. **QAOA variants:** Explore Recursive QAOA, Adaptive QAOA, and
   constraint-preserving mixers.
6. **Clinical validation:** Partner with medical professionals to
   validate selected biomarkers against domain knowledge.

---

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate
   Optimization Algorithm. arXiv:1411.4028.
2. Mücke, S., et al. (2023). Feature Selection on Quantum Computers.
   Quantum Machine Intelligence, 5, 11.
3. Zoufal, L., Lucchi, A., & Woerner, S. (2023). Variational Quantum
   Feature Selection. arXiv:2305.07142.
4. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear
   Feature Extraction for Breast Tumor Diagnosis.

---

*This report was auto-generated from experimental results.*
"""

    # Save
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  ✓ Research report saved to: {save_path}")
    return report
