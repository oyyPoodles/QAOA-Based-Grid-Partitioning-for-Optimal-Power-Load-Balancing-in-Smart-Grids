"""
================================================================================
MAIN PIPELINE - QAOA Grid Partitioning for Smart Grids
================================================================================
End-to-end:
  Phase 1: Problem Formulation (Steps 1-5)
  Phase 2: QAOA Implementation (Steps 6-8)
  Phase 3: Practical Interpretation (Steps 9-10)
  Phase 4: Visualization (Steps 11-12)
  Phase 5: System Design (Steps 13-14) - see docs/architecture.md
  Phase 6: Research Output (Step 15)

Usage: python main.py
================================================================================
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import build_smart_grid
from src.qaoa.hamiltonian import (
    build_grid_partition_qubo,
    qubo_to_ising,
    evaluate_partition,
)
from src.qaoa.optimizer import QAOAGridOptimizer
from src.visualization.plots import generate_all_plots
from src.research.report import generate_research_report


def main():
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  QAOA-BASED GRID PARTITIONING FOR OPTIMAL POWER")
    print("  LOAD BALANCING IN SMART GRIDS")
    print("=" * 70)

    # ==================================================================
    # PHASE 1: STEPS 1-2 - Build Smart Grid Graph from Dataset
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 1 - STEPS 1-2: GRID GRAPH CONSTRUCTION")
    print("=" * 70)

    G, aligned_data, mean_loads, corr_matrix = build_smart_grid(
        dataset_dir="dataset",
        n_nodes=8,     # Keep manageable for QAOA simulation
        freq="D",      # Daily aggregation
    )

    nodes_list = sorted(G.nodes())

    # ==================================================================
    # PHASE 1: STEPS 3-5 - QUBO & Ising Formulation
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 1 - STEPS 3-5: QUBO & ISING FORMULATION")
    print("=" * 70)

    Q, nodes = build_grid_partition_qubo(G, alpha=2.0)
    h, J, offset = qubo_to_ising(Q)

    print(f"\n  Ising Hamiltonian:")
    print(f"    h (linear):    {np.round(h, 4)}")
    print(f"    Non-zero J:    {np.sum(np.abs(J) > 1e-10)}")
    print(f"    Offset:        {offset:.4f}")

    # ==================================================================
    # PHASE 2: STEPS 6-8 - QAOA Optimization
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 2 - STEPS 6-8: QAOA OPTIMIZATION")
    print("=" * 70)

    optimizer = QAOAGridOptimizer(
        G=G, Q=Q, nodes=nodes,
        p=1, shots=1024, maxiter=200, random_state=42,
    )

    opt_result = optimizer.optimize()
    partition_result = optimizer.extract_partition(shots=8192)

    # ==================================================================
    # PHASE 3: STEPS 9-10 - Practical Interpretation & Evaluation
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 3 - STEPS 9-10: EVALUATION")
    print("=" * 70)

    # Random partition baseline
    np.random.seed(0)
    random_bits = np.random.randint(0, 2, len(nodes))
    if random_bits.sum() == 0:
        random_bits[0] = 1
    elif random_bits.sum() == len(nodes):
        random_bits[0] = 0

    random_metrics = evaluate_partition(G, nodes, random_bits)
    qaoa_metrics = partition_result["metrics"]

    # Load variance comparison
    rand_var = np.var([random_metrics["load_0"], random_metrics["load_1"]])
    qaoa_var = np.var([qaoa_metrics["load_0"], qaoa_metrics["load_1"]])

    print(f"\n  PERFORMANCE COMPARISON")
    print(f"  {'-'*55}")
    print(f"  {'Metric':<20s} {'Random':>15s} {'QAOA':>15s}")
    print(f"  {'-'*55}")
    print(f"  {'Load A':<20s} {random_metrics['load_0']:>12,.0f} MW {qaoa_metrics['load_0']:>12,.0f} MW")
    print(f"  {'Load B':<20s} {random_metrics['load_1']:>12,.0f} MW {qaoa_metrics['load_1']:>12,.0f} MW")
    print(f"  {'Imbalance':<20s} {random_metrics['load_imbalance']*100:>12.2f} %  {qaoa_metrics['load_imbalance']*100:>12.2f} %")
    print(f"  {'Load Variance':<20s} {rand_var:>15,.0f} {qaoa_var:>15,.0f}")
    print(f"  {'Edges Cut':<20s} {random_metrics['cut_edges']:>15d} {qaoa_metrics['cut_edges']:>15d}")
    print(f"  {'Cut Weight':<20s} {random_metrics['cut_weight']:>15.4f} {qaoa_metrics['cut_weight']:>15.4f}")
    print(f"  {'-'*55}")

    improvement = 0
    if rand_var > 0:
        improvement = (1 - qaoa_var / rand_var) * 100
        print(f"\n  Variance improvement: {improvement:.1f}%")

    # Estimated transmission loss
    rand_loss = random_metrics["cut_weight"] * abs(random_metrics["load_0"] - random_metrics["load_1"])
    qaoa_loss = qaoa_metrics["cut_weight"] * abs(qaoa_metrics["load_0"] - qaoa_metrics["load_1"])
    print(f"\n  Estimated transmission loss proxy:")
    print(f"    Random: {rand_loss:,.0f}")
    print(f"    QAOA:   {qaoa_loss:,.0f}")
    if rand_loss > 0:
        print(f"    Reduction: {(1 - qaoa_loss / rand_loss) * 100:.1f}%")

    # ==================================================================
    # PHASE 4: STEPS 11-12 - Visualization
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 4 - STEPS 11-12: VISUALIZATION")
    print("=" * 70)

    # Filter corr_matrix to only include nodes in the graph
    corr_sub = corr_matrix.loc[
        [n for n in nodes if n in corr_matrix.index],
        [n for n in nodes if n in corr_matrix.columns]
    ]

    generate_all_plots(
        G=G,
        nodes=nodes,
        partition_bits=partition_result["bits"],
        cost_history=opt_result["convergence_history"],
        corr_matrix=corr_sub,
        before_metrics=random_metrics,
        after_metrics=qaoa_metrics,
        save_dir="outputs",
    )

    # ==================================================================
    # PHASE 6: STEP 15 - Research Report
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("  PHASE 6 - STEP 15: RESEARCH REPORT")
    print("=" * 70)

    generate_research_report(
        G=G,
        nodes=nodes,
        partition_result=partition_result,
        optimization_result=opt_result,
        before_metrics=random_metrics,
        save_path="docs/research_paper.md",
    )

    # ==================================================================
    # SUMMARY
    # ==================================================================
    total_time = time.time() - total_start

    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total time:        {total_time:.1f}s")
    print(f"  Grid:              {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  QAOA partition:    {partition_result['bitstring']}")
    print(f"  Partition A:       {qaoa_metrics['partition_0']}")
    print(f"  Partition B:       {qaoa_metrics['partition_1']}")
    print(f"  Load imbalance:    {qaoa_metrics['load_imbalance']*100:.2f}%")
    print(f"\n  Generated outputs:")
    print(f"    outputs/original_grid.png")
    print(f"    outputs/partitioned_grid.png")
    print(f"    outputs/load_distribution.png")
    print(f"    outputs/qaoa_convergence.png")
    print(f"    outputs/correlation_heatmap.png")
    print(f"    outputs/load_balance_comparison.png")
    print(f"    docs/research_paper.md")
    print(f"\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
