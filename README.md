# 🧬 Hybrid Quantum-Classical Optimization using QAOA for Medical Feature Selection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end research project that uses the **Quantum Approximate Optimization Algorithm (QAOA)** to perform feature selection on a medical dataset (Breast Cancer Wisconsin), benchmarked against classical methods (PCA, LASSO, Genetic Algorithm).

---

## 🎯 Project Overview

Feature selection in medical datasets is a **combinatorial optimization problem** — with n features, there are 2ⁿ possible subsets to evaluate. This project formulates feature selection as a **QUBO (Quadratic Unconstrained Binary Optimization)** problem and solves it using QAOA, a variational quantum algorithm.

### Key Contributions
- **QUBO formulation** that maximizes feature relevance (mutual information) while minimizing redundancy (correlation)
- **Manual QAOA circuit construction** with educational gate-level decomposition
- **Comprehensive benchmarking** against PCA, LASSO, and Genetic Algorithm
- **Publication-quality visualizations** and IEEE-style research sections

---

## 📁 Project Structure

```
QAOA/
├── main.py                            # End-to-end pipeline runner
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── src/
│   ├── qaoa/
│   │   ├── hamiltonian.py             # QUBO & Ising Hamiltonian (Steps 1-2)
│   │   ├── circuit.py                 # QAOA circuit construction (Step 3)
│   │   ├── optimizer.py               # Classical optimization loop (Step 4)
│   │   └── feature_selector.py        # Feature selection wrapper (Steps 5, 7)
│   ├── data/
│   │   └── dataset.py                 # Dataset loading & preprocessing (Step 6)
│   ├── ml/
│   │   ├── classifier.py              # ML training & evaluation (Steps 8-9)
│   │   └── baselines.py               # PCA, LASSO, GA (Steps 10-11)
│   ├── visualization/
│   │   └── plots.py                   # All visualizations (Step 12)
│   └── research/
│       └── report.py                  # Research paper generator (Step 15)
├── notebooks/
│   └── QAOA_Medical_Feature_Selection.py  # 📓 Complete Colab notebook
├── outputs/                           # Generated plots & results
└── docs/
    ├── architecture.md                # System architecture (Steps 13-14)
    └── research_paper.md              # Generated research paper
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python main.py
```

### 3. Google Colab
Upload `notebooks/QAOA_Medical_Feature_Selection.py` to Colab and run all cells.

---

## 🧠 Methodology

### QAOA for Feature Selection

1. **Compute feature statistics**: mutual information (relevance) and pairwise correlation (redundancy)
2. **Build QUBO matrix**: Q where Q_ii = -rᵢ, Q_ij = λ·cᵢⱼ/2
3. **Map to Ising Hamiltonian**: xᵢ = (1 - Zᵢ)/2
4. **Build QAOA circuit**: alternating cost (ZZ, Z gates) and mixer (Rx gates) layers
5. **Optimize parameters**: COBYLA minimizes ⟨H_C⟩
6. **Extract solution**: most probable bitstring → selected features

### Mathematical Formulation

**Objective:**
```
max  Σᵢ rᵢ·xᵢ  −  λ · Σᵢ<ⱼ cᵢⱼ·xᵢ·xⱼ
      ↑ relevance      ↑ redundancy
```

**QUBO:** `min x^T Q x`

**Ising:** `H_C = Σᵢ hᵢZᵢ + Σᵢ<ⱼ JᵢⱼZᵢZⱼ + offset`

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `outputs/feature_importance.png` | Feature importance bar chart (MI scores) |
| `outputs/qaoa_convergence.png` | QAOA optimization convergence plot |
| `outputs/accuracy_comparison.png` | ML metrics across all methods |
| `outputs/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/feature_count_comparison.png` | Features selected per method |
| `outputs/computation_time.png` | Computational cost comparison |
| `docs/research_paper.md` | Auto-generated IEEE-style paper sections |

---

## ⚙️ Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_candidates` | 10 | Number of candidate features (qubits) |
| `lambda_param` | 0.5 | Relevance-redundancy trade-off |
| `p` | 1 | QAOA circuit depth |
| `shots` | 1024 | Measurement shots per evaluation |
| `maxiter` | 150 | Maximum COBYLA iterations |

---

## 🔬 Technology Stack

| Component | Technology |
|-----------|-----------|
| Quantum Circuit | Qiskit ≥ 1.0 |
| Simulation | Qiskit Aer |
| Optimization | SciPy (COBYLA) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Machine Learning | Scikit-learn (LogReg, RF) |
| Visualization | Matplotlib, Seaborn |

---

## 📄 Citation

If you use this project in your research:

```bibtex
@misc{qaoa_feature_selection,
  title={Hybrid Quantum-Classical Optimization using QAOA for Medical Feature Selection},
  year={2026},
  note={Available at: https://github.com/username/QAOA}
}
```

---

## 📚 References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028.
2. Mücke, S., et al. (2023). *Feature Selection on Quantum Computers.* Quantum Machine Intelligence, 5, 11.
3. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear Feature Extraction for Breast Tumor Diagnosis.*

---

## License

MIT License
