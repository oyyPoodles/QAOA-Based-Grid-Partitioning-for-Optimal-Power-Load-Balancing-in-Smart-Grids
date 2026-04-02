<div align="center">

# ⚡ QAOA-Based Grid Partitioning
### Optimal Power Load Balancing in Smart Grids

<br/>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit 1.0+](https://img.shields.io/badge/Qiskit-1.0+-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7CA18?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Colab Ready](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br/>

> *A hybrid quantum-classical approach to smart grid partitioning using the*
> ***Quantum Approximate Optimization Algorithm (QAOA)***, *formulated as a QUBO*
> *combining MAX-CUT with a load balancing penalty.*

<br/>

</div>

---

## 🌐 Overview

Modern power grids require intelligent partitioning to operate efficiently and resiliently. This project tackles the **NP-hard** problem of grid partitioning using a quantum-native approach — bringing together quantum circuit design, graph theory, and classical optimization into a seamless pipeline.

| Challenge | Solution |
|-----------|----------|
| 🔴 Minimize transmission losses | MAX-CUT objective in QUBO |
| 🟡 Balance electrical load across sub-grids | Load-balancing penalty term |
| 🟢 Enable decentralized fault response | Quantum-optimal binary partitioning |
| ⚫ $2^n$ possible partitions (NP-hard) | QAOA circuit with COBYLA optimizer |

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph DATA["Data Layer"]
        D1[("PJM Hourly\nEnergy CSVs")]:::data
        D2["Dataset Loader\ndataset.py"]:::data
    end

    subgraph GC["Graph Construction"]
        G1["Correlation\nMatrix"]:::gnode
        G2["Weighted Graph\nNetworkX"]:::gnode
        G3["Edge Weights\n1 - abs(corr)"]:::gnode
    end

    subgraph QF["QUBO Formulation"]
        Q1["MAX-CUT\nObjective"]:::qubo
        Q2["Load Balance\nPenalty lambda"]:::qubo
        Q3["QUBO Matrix Q"]:::qubo
    end

    subgraph IS["Quantum Mapping"]
        I1["Ising Mapping\nxi = (1 - Zi) / 2"]:::ising
        I2["Cost Hamiltonian\nH_C"]:::ising
        I3["Mixer Hamiltonian\nH_B = sum Xi"]:::ising
    end

    subgraph QA["QAOA Circuit"]
        C1["Initial State\nPlus state x n qubits"]:::circuit
        C2["Cost Layer\nexp(-i gamma H_C)"]:::circuit
        C3["Mixer Layer\nexp(-i beta H_B)"]:::circuit
        C4["p Repetitions"]:::circuit
    end

    subgraph CL["Classical Optimization"]
        O1["COBYLA\nOptimizer"]:::opt
        O2["Parameter Update\ngamma, beta"]:::opt
        O3["Expectation Value\nH_C"]:::opt
    end

    subgraph OUT["Output"]
        R1["Optimal\nBitstring"]:::outnode
        R2["Grid Partitions\nA and B"]:::outnode
        R3["Visualizations\nand Research Paper"]:::outnode
    end

    D1 --> D2
    D2 --> G1 --> G2 --> G3
    G3 --> Q1
    G3 --> Q2
    Q1 --> Q3
    Q2 --> Q3
    Q3 --> I1
    Q3 --> I3
    I1 --> I2
    I2 --> C2
    I3 --> C3
    C1 --> C2 --> C3 --> C4
    C4 -->|"measure"| O3
    O3 --> O1 --> O2
    O2 -->|"update params"| C2
    O1 -->|"converged"| R1
    R1 --> R2 --> R3

    classDef data     fill:#1e3a5f,stroke:#4a9eff,color:#e8f4ff
    classDef gnode    fill:#1a3a2a,stroke:#4aff8a,color:#e8fff0
    classDef qubo     fill:#3a1a3a,stroke:#cc4aff,color:#f8e8ff
    classDef ising    fill:#3a2a1a,stroke:#ff9a4a,color:#fff4e8
    classDef circuit  fill:#1a2a3a,stroke:#4ac8ff,color:#e8f8ff
    classDef opt      fill:#3a1a1a,stroke:#ff4a4a,color:#ffe8e8
    classDef outnode  fill:#2a3a1a,stroke:#c8ff4a,color:#f8ffe8
```

---

## 🔄 End-to-End Workflow

```mermaid
sequenceDiagram
    autonumber

    participant MP  as main.py
    participant DL  as DataLoader
    participant GB  as GraphBuilder
    participant QF  as QUBOEngine
    participant IM  as IsingMapper
    participant QC  as QAOACircuit
    participant SIM as Simulator
    participant COPT as COBYLAOptimizer
    participant PD  as PartitionDecoder
    participant VIZ as Visualizer

    Note over MP,GB: Step 1–2: Data Ingestion and Graph Construction
    MP  ->> DL  : load data
    DL  -->> MP : dataframes
    MP  ->> GB  : build graph
    GB  ->> GB  : compute correlations and weights
    GB  -->> MP : graph G(V, E, W)

    Note over MP,IM: Step 3–5: QUBO Formulation and Ising Mapping
    MP  ->> QF  : build QUBO
    QF  -->> MP : matrix Q
    MP  ->> IM  : map to Ising model
    IM  -->> MP : H_C, H_B

    Note over MP,SIM: Step 6: QAOA Circuit Construction
    MP  ->> QC  : build circuit
    QC  -->> MP : parameterized circuit

    Note over MP,COPT: Step 7–8: Optimization Loop
    MP  ->> COPT : optimize
    loop until convergence
        COPT ->> SIM : execute circuit
        SIM -->> COPT : results
        COPT ->> COPT : update parameters
    end
    COPT -->> MP : optimal parameters

    Note over MP,PD: Step 9–10: Partition Extraction
    MP  ->> SIM : execute with optimal parameters
    SIM -->> MP : distribution
    MP  ->> PD  : decode solution
    PD  -->> MP : partitions and metrics

    Note over MP,VIZ: Step 11–15: Visualization and Reporting
    MP  ->> VIZ : generate plots
    VIZ -->> MP : outputs saved
    MP  ->> MP  : generate report
```

## 🚀 Quick Start

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/oyyPoodles/QAOA-Based-Grid-Partitioning-for-Optimal-Power-Load-Balancing-in-Smart-Grids.git
cd QAOA-Based-Grid-Partitioning-for-Optimal-Power-Load-Balancing-in-Smart-Grids

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

```

### ☁️ Google Colab (No Setup Required)

1. Open [`notebooks/QAOA_Smart_Grid.py`](notebooks/QAOA_Smart_Grid.py)
2. Paste the code into a new **[Google Colab Notebook](https://colab.research.google.com/)**
3. Upload the `dataset/` folder to your Colab environment
4. Run all cells sequentially ▶️

---

## 📁 Project Structure

```
📦 QAOA-Grid-Partitioning
├── 🐍 main.py                          ← End-to-end pipeline entry point
├── 📋 requirements.txt
│
├── 📂 dataset/                         ← PJM Hourly Energy CSVs (11 regions)
│   ├── AEP_hourly.csv
│   └── ... (10 more regional files)
│
├── 📂 src/
│   ├── 📂 data/
│   │   └── dataset.py                  ← Steps 1–2 · Graph construction
│   ├── 📂 qaoa/
│   │   ├── hamiltonian.py              ← Steps 3–5 · QUBO & Ising mapping
│   │   ├── circuit.py                  ← Step  6  · QAOA circuit builder
│   │   └── optimizer.py                ← Steps 7–8 · Optimization & extraction
│   ├── 📂 visualization/
│   │   └── plots.py                    ← Steps 11–12 · All visualizations
│   └── 📂 research/
│       └── report.py                   ← Step  15 · Research paper generator
│
├── 📂 notebooks/
│   └── QAOA_Smart_Grid.py              ← Complete Google Colab notebook
│
├── 📂 outputs/                         ← High-resolution generated plots
└── 📂 docs/
    ├── architecture.md
    └── research_paper.md               ← Auto-generated IEEE-style paper
```

---

## 📈 Generated Outputs

After running `python main.py`, the following artifacts are saved to `outputs/`:

| File | Description |
|------|-------------|
| `original_grid.png` | Power grid graph — thicker edges = lower correlation between regions |
| `partitioned_grid.png` | QAOA solution — cut edges highlighted between Partition A and B |
| `load_balance_comparison.png` | Bar chart: random partition vs. QAOA-balanced partition |
| `load_distribution.png` | Mean MW load breakdown per partition |
| `correlation_heatmap.png` | Statistical heatmap of inter-region load similarity |
| `qaoa_convergence.png` | COBYLA optimizer convergence curve across iterations |
| `docs/research_paper.md` | Auto-generated IEEE-style research paper |

---

## 📊 Dataset

**[PJM Interconnection Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) — Kaggle**

- 🗺️ **11 regional power zones** in the US Eastern Interconnection
- 📅 Hourly MW consumption spanning multiple years
- 🔌 Regions: `AEP` · `COMED` · `DAYTON` · `DEOK` · `DOM` · `DUQ` · `EKPC` · `FE` · `NI` · `PJME` · `PJMW`

---

## 🔬 Technology Stack

```mermaid
graph TB
    A[QAOA Grid]

    A --> B[Quantum]
    B --> B1[Qiskit 1.0]
    B --> B2[qiskit-aer Statevector]

    A --> C[Graph]
    C --> C1[NetworkX]
    C --> C2[Correlation Mapping]

    A --> D[Optimization]
    D --> D1[SciPy COBYLA]
    D --> D2[QUBO Formulation]

    A --> E[Data]
    E --> E1[NumPy]
    E --> E2[Pandas]

    A --> F[Visualization]
    F --> F1[Matplotlib]
    F --> F2[Seaborn]
```

---

## 🧪 QUBO Formulation

The optimization problem is cast as a **Quadratic Unconstrained Binary Optimization**:

$$Q = \underbrace{\sum_{(i,j) \in E} w_{ij} \cdot x_i(1 - x_j)}_{\text{MAX-CUT (transmission cost)}} + \underbrace{\lambda \left(\sum_{i} L_i x_i - \frac{L_{total}}{2}\right)^2}_{\text{Load Balance Penalty}}$$

The Ising mapping $x_i = \frac{1 - Z_i}{2}$ converts this to the cost Hamiltonian $H_C$ acting on qubits.

---

## 📚 References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A Quantum Approximate Optimization Algorithm.* [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
2. *PJM Hourly Energy Consumption Dataset*, Kaggle — [robikscube/hourly-energy-consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

---

<div align="center">

**[MIT License](LICENSE)** · Made with ⚡ and a few qubits

</div>
