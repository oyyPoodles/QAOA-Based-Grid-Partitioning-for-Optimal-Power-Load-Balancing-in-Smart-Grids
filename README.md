# ⚡ QAOA-Based Grid Partitioning for Optimal Power Load Balancing in Smart Grids

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid quantum-classical approach to smart grid partitioning using the **Quantum Approximate Optimization Algorithm (QAOA)**. The grid partitioning problem is formulated as a **QUBO** combining MAX-CUT (minimizing transmission cost) with a load balancing penalty, then solved via a parameterized QAOA circuit.

---

## 🎯 Problem

Modern power grids need intelligent partitioning to:
- **Minimize transmission losses** across sub-grid boundaries.
- **Balance electrical load** for grid stability.
- **Enable decentralized control** for faster fault response.

Grid partitioning is NP-hard ($2^n$ possible partitions for $n$ nodes). QAOA provides a quantum-native approach to this combinatorial optimization.

---

## 🧪 Methodology

1. **Dataset → Graph**: PJM hourly energy data → weighted graph (nodes = regions, edge weights = $1 - |\text{correlation}|$).
2. **QUBO Formulation**: MAX-CUT objective + load balancing penalty → QUBO matrix $Q$.
3. **Ising Mapping**: $x_i = (1 - Z_i) / 2$ → Hamiltonian $H_C$.
4. **QAOA Circuit**: Parameterized cost + mixer layers optimized by COBYLA classical optimizer.
5. **Partition**: Optimal bitstring → mathematically balanced grid partition.

---

## 🚀 Quick Start (Local)

1. Clone the repository and install the dependencies:
```bash
git clone https://github.com/oyyPoodles/QAOA-Based-Grid-Partitioning-for-Optimal-Power-Load-Balancing-in-Smart-Grids.git
cd QAOA-Based-Grid-Partitioning-for-Optimal-Power-Load-Balancing-in-Smart-Grids
pip install -r requirements.txt
```

2. Run the end-to-end Python pipeline:
```bash
python main.py
```

---

## 📓 Google Colab Ready

Prefer to run the project entirely in the cloud without installing any local dependencies? 
We have bundled the entire 15-step pipeline into a single, interactive Google Colab notebook!

1. Open [notebooks/QAOA_Smart_Grid.py](notebooks/QAOA_Smart_Grid.py).
2. Copy the code into a new **[Google Colab Notebook](https://colab.research.google.com/)**.
3. Upload the `dataset/` folder contents to your Colab environment in a folder named `dataset/`.
4. Run all cells sequentially.

---

## 📁 Project Structure

```
.
├── main.py                         # End-to-end pipeline
├── requirements.txt                # Dependencies
├── dataset/                        # PJM hourly energy CSVs
│   ├── AEP_hourly.csv
│   └── ... (11 regional files)
├── src/
│   ├── data/dataset.py             # Steps 1-2: Graph construction
│   ├── qaoa/
│   │   ├── hamiltonian.py          # Steps 3-5: QUBO & Ising
│   │   ├── circuit.py              # Step 6: QAOA circuit
│   │   └── optimizer.py            # Steps 7-8: Optimization & extraction
│   ├── visualization/plots.py      # Steps 11-12: All plots
│   └── research/report.py          # Step 15: Research paper output
├── notebooks/
│   └── QAOA_Smart_Grid.py          # Complete Google Colab notebook
├── outputs/                        # High-resolution generated plots
└── docs/
    ├── architecture.md             # System design & logical workflow
    └── research_paper.md           # Auto-generated comprehensive paper
```

---

## 📈 Key Visualizations & Outputs

When you run the pipeline (`python main.py`), it automatically generates several key figures in the `outputs/` directory:

*   **`original_grid.png`**: Visualizes the power grid layout, drawing thicker transmission lines between regions with less load correlation.
*   **`partitioned_grid.png`**: Displays the final QAOA solution, highlighting exactly which edges between regions were cut to form the two distinct, balanced partitions.
*   **`load_balance_comparison.png`**: Provides a bar chart directly contrasting the severe load imbalance of a random partitioning method against the highly balanced QAOA output.
*   **`load_distribution.png`**: A granular breakdown of mean loads (in MW) assigned to Partition A vs Partition B.
*   **`correlation_heatmap.png`**: A statistical heatmap showing how similar each PJM region's load consumption profile is to every other region over time.
*   **`qaoa_convergence.png`**: Tracks the COBYLA optimizer as it minimizes the expected cost $\langle H_C \rangle$ at each iteration loop.

An IEEE-style research report is also automatically aggregated with your particular run outputs and saved as `docs/research_paper.md`.

---

## 📊 Dataset

**PJM Interconnection Hourly Energy Consumption** (Kaggle):
- **11 regional power zones** in the US Eastern Interconnection.
- Hourly MW consumption data spanning multiple years.
- Regions modeled: AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW.

---

## 🔬 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Quantum Circuit** | Qiskit ≥ 1.0.0 |
| **Simulation** | Qiskit Statevector Simulator (`qiskit-aer`) |
| **Graph Modeling** | NetworkX |
| **Optimization** | SciPy (COBYLA algorithm) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |

---

## 📚 References

1. Farhi et al. (2014). *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028
2. PJM Hourly Energy Consumption Dataset, Kaggle

## License
[MIT License](LICENSE)
