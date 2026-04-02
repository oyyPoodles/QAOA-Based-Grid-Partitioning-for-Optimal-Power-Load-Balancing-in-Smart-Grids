# ⚡ QAOA-Based Grid Partitioning for Optimal Power Load Balancing in Smart Grids

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)

A hybrid quantum-classical approach to smart grid partitioning using the **Quantum Approximate Optimization Algorithm (QAOA)**. The grid partitioning problem is formulated as a **QUBO** combining MAX-CUT (minimizing transmission cost) with a load balancing penalty, then solved via a parameterized QAOA circuit.

---

## 🎯 Problem

Modern power grids need intelligent partitioning to:
- **Minimize transmission losses** across sub-grid boundaries
- **Balance electrical load** for grid stability
- **Enable decentralized control** for faster fault response

Grid partitioning is NP-hard (2ⁿ possible partitions for n nodes). QAOA provides a quantum-native approach to this combinatorial optimization.

---

## 🧪 Methodology

1. **Dataset → Graph**: PJM hourly energy data → weighted graph (nodes = regions, edge weights = 1-|correlation|)
2. **QUBO Formulation**: MAX-CUT objective + load balancing penalty → QUBO matrix Q
3. **Ising Mapping**: x_i = (1-Z_i)/2 → Hamiltonian H_C
4. **QAOA Circuit**: Parameterized cost + mixer layers optimized by COBYLA
5. **Partition**: Optimal bitstring → balanced grid partition

---

## 📁 Project Structure

```
QAOA/
├── main.py                         # End-to-end pipeline
├── requirements.txt                # Dependencies
├── dataset/                        # PJM hourly energy CSVs
│   ├── AEP_hourly.csv
│   ├── COMED_hourly.csv
│   ├── DAYTON_hourly.csv
│   └── ... (11 regional files)
├── src/
│   ├── data/dataset.py             # Steps 1-2: Graph construction
│   ├── qaoa/
│   │   ├── hamiltonian.py          # Steps 3-5: QUBO & Ising
│   │   ├── circuit.py              # Step 6: QAOA circuit
│   │   └── optimizer.py            # Steps 7-8: Optimization & extraction
│   ├── visualization/plots.py     # Steps 11-12: All plots
│   └── research/report.py         # Step 15: Research paper
├── notebooks/
│   └── QAOA_Smart_Grid.py         # Google Colab notebook
├── outputs/                        # Generated plots
└── docs/
    ├── architecture.md             # Steps 13-14: System design
    └── research_paper.md           # Auto-generated paper
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

---

## 📊 Dataset

**PJM Interconnection Hourly Energy Consumption** (Kaggle):
- 11 regional power zones in the US Eastern Interconnection
- Hourly MW consumption data spanning multiple years
- Regions: AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW

---

## 🔬 Technology Stack

| Component | Technology |
|-----------|-----------|
| Quantum Circuit | Qiskit ≥ 1.0 |
| Simulation | Qiskit Aer |
| Graph Modeling | NetworkX |
| Optimization | SciPy (COBYLA) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## 📚 References

1. Farhi et al. (2014). *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028
2. PJM Hourly Energy Consumption Dataset, Kaggle

## License
MIT License
