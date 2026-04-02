"""
================================================================================
Visualization Module
================================================================================

STEP 12: Visualization
-----------------------
Generate publication-quality plots for research analysis:

  1. Feature Importance Graph — Bar chart of mutual information scores
  2. Accuracy Comparison — Grouped bar chart across methods and models
  3. Convergence Plot — QAOA cost function over optimizer iterations
  4. Correlation Heatmap — Feature-feature correlation structure
  5. Comprehensive Results — Multi-panel summary figure
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/script
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Global style settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})


def plot_feature_importance(relevance_scores, feature_names, selected_indices=None,
                            save_path=None, top_n=None):
    """
    Bar chart of feature importance (mutual information scores).

    Features selected by QAOA are highlighted in a different color.

    Parameters
    ----------
    relevance_scores : np.ndarray
        Mutual information scores.
    feature_names : list of str
    selected_indices : np.ndarray, optional
        Indices of QAOA-selected features (for highlighting).
    save_path : str, optional
        Path to save the figure.
    top_n : int, optional
        Show only the top-n features.
    """
    n = len(relevance_scores)
    if top_n is not None:
        sort_idx = np.argsort(relevance_scores)[::-1][:top_n]
    else:
        sort_idx = np.argsort(relevance_scores)[::-1]

    sorted_scores = relevance_scores[sort_idx]
    sorted_names = [feature_names[i] for i in sort_idx]

    # Color: highlight selected features
    colors = []
    for idx in sort_idx:
        if selected_indices is not None and idx in selected_indices:
            colors.append("#2ecc71")  # Green for selected
        else:
            colors.append("#3498db")  # Blue for not selected

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(sorted_scores)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Mutual Information Score (Normalized)")
    ax.set_title("Feature Importance — Mutual Information with Target")
    ax.invert_yaxis()

    # Legend
    if selected_indices is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="QAOA Selected"),
            Patch(facecolor="#3498db", label="Not Selected"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_accuracy_comparison(all_results, save_path=None):
    """
    Grouped bar chart comparing accuracy across feature selection methods.

    Parameters
    ----------
    all_results : dict
        {method_name: {model_name: results_dict}}
    save_path : str, optional
    """
    methods = list(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())

    # Extract metrics
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(len(methods))
        width = 0.35

        for i, model_name in enumerate(model_names):
            values = [all_results[m][model_name][metric] for m in methods]
            display_name = all_results[methods[0]][model_name]["model_name"]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=display_name,
                          alpha=0.85, edgecolor="white", linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Feature Selection Method")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylim(0.8, 1.02)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("ML Performance Comparison Across Feature Selection Methods",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_convergence(cost_history, save_path=None):
    """
    Plot QAOA optimization convergence.

    Shows the cost function value over optimizer iterations, with
    running minimum to visualize convergence behavior.

    Parameters
    ----------
    cost_history : list of float
        Cost values per iteration.
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = range(1, len(cost_history) + 1)

    # Raw cost values
    ax.plot(iterations, cost_history, "b-", alpha=0.4, linewidth=0.8,
            label="Cost per iteration")

    # Running minimum
    running_min = np.minimum.accumulate(cost_history)
    ax.plot(iterations, running_min, "r-", linewidth=2,
            label="Running minimum")

    # Moving average
    if len(cost_history) > 10:
        window = min(10, len(cost_history) // 5)
        moving_avg = np.convolve(cost_history,
                                  np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(cost_history) + 1), moving_avg,
                "g--", linewidth=1.5, alpha=0.7, label=f"Moving avg (w={window})")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Expected Cost ⟨H_C⟩")
    ax.set_title("QAOA Optimization Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate minimum
    min_idx = np.argmin(cost_history)
    min_val = cost_history[min_idx]
    ax.annotate(f"Min: {min_val:.4f}\n(iter {min_idx+1})",
                xy=(min_idx+1, min_val),
                xytext=(min_idx+1 + len(cost_history)*0.1, min_val),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_correlation_heatmap(X, feature_names, save_path=None, max_features=20):
    """
    Heatmap of feature-feature correlations.

    Visualizes the redundancy structure that QAOA aims to minimize.

    Parameters
    ----------
    X : np.ndarray
    feature_names : list of str
    save_path : str, optional
    max_features : int
        Maximum features to display.
    """
    if X.shape[1] > max_features:
        X = X[:, :max_features]
        feature_names = feature_names[:max_features]

    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                xticklabels=feature_names, yticklabels=feature_names,
                ax=ax, vmin=-1, vmax=1,
                annot_kws={"fontsize": 6})

    ax.set_title("Feature Correlation Matrix\n(Redundancy Structure)",
                 fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_feature_count_comparison(results_dict, save_path=None):
    """
    Bar chart comparing number of features selected by each method.

    Parameters
    ----------
    results_dict : dict
        {method_name: n_features}
    save_path : str, optional
    """
    methods = list(results_dict.keys())
    counts = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(methods))
    bars = ax.bar(methods, counts, color=colors, edgecolor="white",
                  linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Number of Features")
    ax.set_title("Features Selected by Each Method")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_computation_time(time_dict, save_path=None):
    """
    Bar chart comparing computation time across methods.

    Parameters
    ----------
    time_dict : dict
        {method_name: time_seconds}
    save_path : str, optional
    """
    methods = list(time_dict.keys())
    times = list(time_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("magma", len(methods))
    bars = ax.bar(methods, times, color=colors, edgecolor="white",
                  linewidth=1.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Computation Time (seconds)")
    ax.set_title("Computational Cost Comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_comprehensive_results(all_ml_results, convergence_history,
                               relevance_scores, feature_names,
                               selected_indices, feature_counts,
                               time_dict, save_dir="outputs"):
    """
    Generate all plots and save to the specified directory.

    Parameters
    ----------
    all_ml_results : dict
    convergence_history : list
    relevance_scores : np.ndarray
    feature_names : list
    selected_indices : np.ndarray
    feature_counts : dict
    time_dict : dict
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # 1. Feature importance
    plot_feature_importance(
        relevance_scores, feature_names, selected_indices,
        save_path=os.path.join(save_dir, "feature_importance.png")
    )

    # 2. Accuracy comparison
    plot_accuracy_comparison(
        all_ml_results,
        save_path=os.path.join(save_dir, "accuracy_comparison.png")
    )

    # 3. Convergence plot
    plot_convergence(
        convergence_history,
        save_path=os.path.join(save_dir, "qaoa_convergence.png")
    )

    # 4. Feature count comparison
    plot_feature_count_comparison(
        feature_counts,
        save_path=os.path.join(save_dir, "feature_count_comparison.png")
    )

    # 5. Computation time
    plot_computation_time(
        time_dict,
        save_path=os.path.join(save_dir, "computation_time.png")
    )

    print(f"\n  All plots saved to: {save_dir}/")
    print(f"{'='*60}\n")
