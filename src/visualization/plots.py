"""
================================================================================
Visualization Module for Smart Grid Partitioning
================================================================================

STEP 11: Graph Visualization
STEP 12: Metrics Visualization
  • Original vs partitioned grid
  • Load distribution chart
  • QAOA optimization convergence
  • Load balance comparison
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})


def plot_original_grid(G, save_path=None):
    """Plot the original power grid graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node sizes proportional to load
    loads = [G.nodes[n]["load"] for n in G.nodes()]
    max_load = max(loads)
    node_sizes = [800 + 2000 * (l / max_load) for l in loads]

    # Edge widths inversely proportional to weight (higher corr = thicker)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [3.0 * (1 - w / max_w) + 0.5 for w in edge_weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           edge_color="#95a5a6", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#3498db",
                           alpha=0.85, edgecolors="#2c3e50", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_title("Original Power Grid Network\n(Node size ∝ load, Edge = correlation-based weight)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()
    return pos


def plot_partitioned_grid(G, nodes, partition_bits, pos=None, save_path=None):
    """Plot the grid with partition coloring."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=2)

    loads = [G.nodes[n]["load"] for n in G.nodes()]
    max_load = max(loads)
    node_sizes = [800 + 2000 * (l / max_load) for l in loads]

    # Color by partition
    colors = ["#e74c3c" if partition_bits[i] == 0 else "#2ecc71"
              for i, n in enumerate(nodes)]

    # Draw edges (highlight cut edges)
    node_idx = {n: i for i, n in enumerate(nodes)}
    for u, v, d in G.edges(data=True):
        ui, vi = node_idx[u], node_idx[v]
        is_cut = partition_bits[ui] != partition_bits[vi]
        style = "--" if is_cut else "-"
        color = "#e74c3c" if is_cut else "#2ecc71"
        alpha = 0.8 if is_cut else 0.4
        width = 2.5 if is_cut else 1.5
        ax.annotate("", xy=pos[v], xytext=pos[u],
                     arrowprops=dict(arrowstyle="-", color=color,
                                     alpha=alpha, lw=width,
                                     linestyle=style))

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors,
                           alpha=0.85, edgecolors="#2c3e50", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="Partition A"),
        mpatches.Patch(facecolor="#2ecc71", label="Partition B"),
        plt.Line2D([0], [0], color="#e74c3c", linestyle="--", lw=2, label="Cut Edge"),
        plt.Line2D([0], [0], color="#2ecc71", linestyle="-", lw=2, label="Internal Edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title("QAOA-Optimized Grid Partition\n(Dashed = cut edges between partitions)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_load_distribution(G, nodes, partition_bits, save_path=None):
    """Bar chart of load per node, colored by partition."""
    fig, ax = plt.subplots(figsize=(12, 5))

    loads = [G.nodes[n]["load"] for n in nodes]
    colors = ["#e74c3c" if partition_bits[i] == 0 else "#2ecc71"
              for i in range(len(nodes))]

    bars = ax.bar(range(len(nodes)), loads, color=colors, edgecolor="white",
                  linewidth=1.5)

    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=30, ha="right")
    ax.set_ylabel("Mean Load (MW)")
    ax.set_title("Load Distribution by Partition Assignment")

    # Add load values
    for bar, load in zip(bars, loads):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 50,
                f"{load:,.0f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    # Partition totals
    load_a = sum(loads[i] for i in range(len(nodes)) if partition_bits[i] == 0)
    load_b = sum(loads[i] for i in range(len(nodes)) if partition_bits[i] == 1)
    ax.axhline(y=0, color="black", linewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label=f"Partition A: {load_a:,.0f} MW"),
        mpatches.Patch(facecolor="#2ecc71", label=f"Partition B: {load_b:,.0f} MW"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_convergence(cost_history, save_path=None):
    """Plot QAOA optimization convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))

    iters = range(1, len(cost_history) + 1)
    ax.plot(iters, cost_history, "b-", alpha=0.4, linewidth=0.8,
            label="Cost per iteration")

    running_min = np.minimum.accumulate(cost_history)
    ax.plot(iters, running_min, "r-", linewidth=2, label="Running minimum")

    if len(cost_history) > 10:
        w = min(10, len(cost_history) // 5)
        ma = np.convolve(cost_history, np.ones(w) / w, mode="valid")
        ax.plot(range(w, len(cost_history) + 1), ma, "g--", linewidth=1.5,
                alpha=0.7, label=f"Moving avg (w={w})")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Expected Cost ⟨H_C⟩")
    ax.set_title("QAOA Optimization Convergence — Grid Partitioning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    min_idx = np.argmin(cost_history)
    ax.annotate(f"Min: {cost_history[min_idx]:.4f}\n(iter {min_idx + 1})",
                xy=(min_idx + 1, cost_history[min_idx]),
                xytext=(min_idx + 1 + len(cost_history) * 0.15,
                        cost_history[min_idx] + 0.02),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_load_balance_comparison(before_variance, after_variance,
                                  before_loads, after_loads, save_path=None):
    """Compare load balance before and after partitioning."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before/After partition loads
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35
    ax1.bar(x - width / 2, before_loads, width, label="Before (Random)",
            color="#e74c3c", alpha=0.8)
    ax1.bar(x + width / 2, after_loads, width, label="After (QAOA)",
            color="#2ecc71", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Partition A", "Partition B"])
    ax1.set_ylabel("Load (MW)")
    ax1.set_title("Partition Load: Random vs QAOA")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, (b, a) in enumerate(zip(before_loads, after_loads)):
        ax1.text(i - width / 2, b + 100, f"{b:,.0f}", ha="center", fontsize=8)
        ax1.text(i + width / 2, a + 100, f"{a:,.0f}", ha="center", fontsize=8)

    # Variance comparison
    ax2 = axes[1]
    methods = ["Random\nPartition", "QAOA\nPartition"]
    variances = [before_variance, after_variance]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax2.bar(methods, variances, color=colors, edgecolor="white",
                   linewidth=1.5)
    ax2.set_ylabel("Load Variance (MW²)")
    ax2.set_title("Load Variance: Random vs QAOA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, variances):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 100,
                 f"{v:,.0f}", ha="center", fontweight="bold")

    improvement = (1 - after_variance / before_variance) * 100 if before_variance > 0 else 0
    ax2.text(0.5, 0.95, f"Improvement: {improvement:.1f}%",
             transform=ax2.transAxes, ha="center", fontsize=11,
             fontweight="bold", color="#27ae60")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_correlation_heatmap(corr_matrix, save_path=None):
    """Plot feature correlation heatmap."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                ax=ax, vmin=-1, vmax=1,
                annot_kws={"fontsize": 9})
    ax.set_title("Inter-Region Load Correlation\n(High corr → similar patterns → keep together)",
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def generate_all_plots(G, nodes, partition_bits, cost_history, corr_matrix,
                        before_metrics, after_metrics, save_dir="outputs"):
    """Generate all visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # 1. Original grid
    pos = plot_original_grid(
        G, save_path=os.path.join(save_dir, "original_grid.png")
    )

    # 2. Partitioned grid
    plot_partitioned_grid(
        G, nodes, partition_bits, pos=pos,
        save_path=os.path.join(save_dir, "partitioned_grid.png")
    )

    # 3. Load distribution
    plot_load_distribution(
        G, nodes, partition_bits,
        save_path=os.path.join(save_dir, "load_distribution.png")
    )

    # 4. Convergence
    plot_convergence(
        cost_history,
        save_path=os.path.join(save_dir, "qaoa_convergence.png")
    )

    # 5. Correlation heatmap
    plot_correlation_heatmap(
        corr_matrix,
        save_path=os.path.join(save_dir, "correlation_heatmap.png")
    )

    # 6. Load balance comparison
    before_var = np.var([before_metrics["load_0"], before_metrics["load_1"]])
    after_var = np.var([after_metrics["load_0"], after_metrics["load_1"]])
    plot_load_balance_comparison(
        before_var, after_var,
        [before_metrics["load_0"], before_metrics["load_1"]],
        [after_metrics["load_0"], after_metrics["load_1"]],
        save_path=os.path.join(save_dir, "load_balance_comparison.png")
    )

    print(f"\n  All plots saved to: {save_dir}/")
    print(f"{'='*60}")
