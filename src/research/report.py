"""
================================================================================
Research Report Generator — Smart Grid Partitioning
================================================================================

STEP 15: IEEE-Style Paper Sections
================================================================================
"""

import os
from datetime import datetime


def generate_research_report(G, nodes, partition_result, optimization_result,
                              before_metrics, save_path="docs/research_paper.md"):
    """Generate IEEE-style research paper with experimental results."""

    metrics = partition_result["metrics"]
    n_nodes = len(nodes)
    n_edges = G.number_of_edges()
    n_iters = optimization_result["num_iterations"]

    import numpy as np
    before_var = np.var([before_metrics["load_0"], before_metrics["load_1"]])
    after_var = np.var([metrics["load_0"], metrics["load_1"]])
    improvement = (1 - after_var / before_var) * 100 if before_var > 0 else 0

    report = f"""# QAOA-Based Grid Partitioning for Optimal Power Load Balancing in Smart Grids

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Abstract

We present a quantum computing approach to smart grid partitioning using the
Quantum Approximate Optimization Algorithm (QAOA). The grid partitioning
problem is formulated as a QUBO (Quadratic Unconstrained Binary Optimization)
combining MAX-CUT for minimizing interconnection costs with a load balancing
penalty. Using the PJM Interconnection hourly energy consumption dataset
({n_nodes} regional nodes, {n_edges} transmission edges), our QAOA-based
approach achieves a load imbalance of {metrics['load_imbalance']*100:.2f}%
between partitions, representing a {improvement:.1f}% variance reduction
compared to random partitioning. The approach demonstrates the viability
of near-term quantum algorithms for power systems optimization.

**Keywords:** QAOA, Smart Grid, Graph Partitioning, QUBO, Load Balancing,
Quantum Computing

---

## 1. Introduction and Problem Statement

### 1.1 Problem Statement

Modern power grids require intelligent partitioning to:
- **Minimize transmission losses** across partition boundaries
- **Balance electrical load** evenly for grid stability
- **Enable decentralized control** for faster fault response
- **Optimize resource allocation** within sub-grids

Grid partitioning is a graph bisection problem, which is NP-hard.
With n nodes, there are 2^n possible partitions to evaluate.

### 1.2 Contribution

We formulate grid partitioning as a QUBO problem combining:
1. **MAX-CUT objective** — separate dissimilar regions to minimize
   inter-partition power flow
2. **Load balancing penalty** — ensure equal total load per partition

The QUBO is mapped to an Ising Hamiltonian and solved using QAOA,
a variational quantum algorithm.

---

## 2. Methodology

### 2.1 Grid Graph Construction

The PJM Interconnection dataset provides hourly energy consumption for
{n_nodes} regional zones. We construct a weighted graph:

- **Nodes** = power regions with load attribute (mean MW)
- **Edges** = transmission interconnections
- **Edge weight** = 1 - |correlation| between regions

Highly correlated regions (similar patterns) have low edge weight,
meaning it's cheap to cut them apart (they don't need to exchange power).

### 2.2 QUBO Formulation

Binary variable x_i ∈ {{0,1}} assigns node i to partition A (0) or B (1).

**MAX-CUT component:**
$$C_{{cut}} = \\sum_{{(i,j)\\in E}} w_{{ij}} (x_i + x_j - 2x_i x_j)$$

**Load balancing:**
$$C_{{balance}} = \\alpha \\cdot \\left(\\sum_i l_i x_i - \\frac{{L}}{{2}}\\right)^2$$

**Combined QUBO:** minimize $-C_{{cut}} + \\alpha \\cdot C_{{balance}}$

### 2.3 Ising Mapping

Substitution $x_i = (1 - Z_i)/2$ yields:
$$H_C = \\sum_i h_i Z_i + \\sum_{{i<j}} J_{{ij}} Z_i Z_j + \\text{{offset}}$$

### 2.4 QAOA Circuit

The QAOA ansatz with depth p=1:
$$|\\gamma, \\beta\\rangle = U_M(\\beta) U_C(\\gamma) |+\\rangle^n$$

- Cost unitary: Rz and CNOT-Rz-CNOT gates
- Mixer unitary: Rx gates
- Optimized with COBYLA

---

## 3. Dataset

**PJM Interconnection Hourly Energy Consumption** (Kaggle):
- {n_nodes} regional zones in the US Eastern Interconnection
- Hourly MW consumption data
- Time range: multiple years of historical data

Regions and their mean loads:
"""

    for node in nodes:
        load = G.nodes[node]["load"]
        partition = "A" if partition_result["bits"][nodes.index(node)] == 0 else "B"
        report += f"| {node} | {load:,.0f} MW | Partition {partition} |\n"

    report += f"""
---

## 4. Results

### 4.1 QAOA Optimization
- **Qubits:** {n_nodes}
- **Circuit depth (p):** 1
- **Optimizer iterations:** {n_iters}
- **Optimal bitstring:** {partition_result['bitstring']}

### 4.2 Partition Quality

| Metric | Random Partition | QAOA Partition |
|--------|-----------------|----------------|
| Load A | {before_metrics['load_0']:,.0f} MW | {metrics['load_0']:,.0f} MW |
| Load B | {before_metrics['load_1']:,.0f} MW | {metrics['load_1']:,.0f} MW |
| Imbalance | {before_metrics['load_imbalance']*100:.2f}% | {metrics['load_imbalance']*100:.2f}% |
| Load Variance | {before_var:,.0f} | {after_var:,.0f} |
| Edges Cut | {before_metrics['cut_edges']} | {metrics['cut_edges']} |
| Cut Weight | {before_metrics['cut_weight']:.4f} | {metrics['cut_weight']:.4f} |

**Variance Improvement: {improvement:.1f}%**

### 4.3 Partition Assignment

**Partition A:** {metrics['partition_0']}
**Partition B:** {metrics['partition_1']}

---

## 5. Discussion

1. QAOA successfully identifies a partition that balances load across the
   two sub-grids while minimizing inter-partition transmission.
2. The correlation-based edge weights effectively capture the benefit of
   keeping similarly-patterned regions together.
3. The load balancing penalty (α) controls the trade-off between cut
   quality and load balance.

### 5.1 Limitations
1. Limited to small graphs (≤ 10 nodes) due to quantum simulation cost.
2. QAOA depth p=1 provides limited approximation quality.
3. Binary partitioning only; real grids may need k-way partitioning.

---

## 6. Future Scope

1. **Multi-way partitioning** using multiple binary QAOA rounds or
   higher-order encodings
2. **Real quantum hardware** execution on IBM Quantum
3. **Dynamic repartitioning** based on real-time load changes
4. **Integration with renewable energy** sources and storage
5. **Larger graphs** using recursive QAOA or graph coarsening
6. **Time-varying optimization** adapting partitions to demand patterns

---

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate
   Optimization Algorithm. arXiv:1411.4028.
2. Guerrero, J., et al. (2020). Smart Grid Partitioning Using Metaheuristic
   Algorithms. Energies, 13(18), 4849.
3. PJM Interconnection Hourly Energy Consumption Dataset, Kaggle.

---

*Auto-generated from experimental results.*
"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  ✓ Research report saved: {save_path}")
    return report
