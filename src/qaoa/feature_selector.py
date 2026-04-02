"""
================================================================================
QAOA-Based Feature Selector
================================================================================

STEP 5: Extract Optimal Bitstring
-----------------------------------
After QAOA optimization, we extract the solution by:

1. Run the optimized circuit with high shot count
2. Collect measurement statistics (probability distribution)
3. Identify the most probable bitstring
4. Interpret: bit i = 1 → feature i is SELECTED

The most probable bitstring approximates the optimal solution to the
QUBO problem, i.e., the feature subset that maximizes relevance while
minimizing redundancy.

STEP 7: Apply QAOA Feature Selection
--------------------------------------
Integration with the data pipeline:
  1. Pre-filter to top-k features (by mutual information)
  2. Build QUBO from filtered features
  3. Run QAOA optimization
  4. Extract selected feature indices (mapping back to original)
================================================================================
"""

import numpy as np
import time
from .hamiltonian import (
    compute_relevance_scores,
    compute_redundancy_matrix,
    build_qubo_matrix,
    qubo_to_ising,
    qubo_objective,
    get_top_features_by_relevance,
)
from .optimizer import QAOAOptimizer


class QAOAFeatureSelector:
    """
    High-level feature selection using QAOA.

    This class wraps the entire QAOA pipeline:
      Data → Pre-filter → QUBO → Ising → Circuit → Optimize → Select

    Parameters
    ----------
    n_candidates : int, default=10
        Number of candidate features to consider (pre-filtered by MI).
        This determines the number of qubits.
    lambda_param : float, default=0.5
        Relevance-redundancy trade-off. Higher → fewer features selected.
    p : int, default=1
        QAOA circuit depth.
    shots : int, default=1024
        Measurement shots per circuit evaluation.
    optimizer_method : str, default='COBYLA'
        Classical optimizer to use.
    maxiter : int, default=200
        Maximum optimizer iterations.
    random_state : int, default=42
        Random seed.
    """

    def __init__(self, n_candidates=10, lambda_param=0.5, p=1, shots=1024,
                 optimizer_method="COBYLA", maxiter=200, random_state=42):
        self.n_candidates = n_candidates
        self.lambda_param = lambda_param
        self.p = p
        self.shots = shots
        self.optimizer_method = optimizer_method
        self.maxiter = maxiter
        self.random_state = random_state

        # Internal state
        self.Q = None
        self.h = None
        self.J = None
        self.offset = None
        self.relevance = None
        self.redundancy = None
        self.candidate_indices = None
        self.selected_mask = None
        self.selected_indices = None
        self.optimizer = None
        self.optimization_result = None
        self.top_bitstrings = None
        self.computation_time = None

    def fit(self, X, y, feature_names=None):
        """
        Run the full QAOA feature selection pipeline.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features. If None, uses indices.

        Returns
        -------
        self : QAOAFeatureSelector
            Fitted selector.
        """
        start_time = time.time()
        n_features = X.shape[1]

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        print(f"\n{'='*60}")
        print(f"QAOA FEATURE SELECTION")
        print(f"{'='*60}")
        print(f"  Total features:     {n_features}")
        print(f"  Candidate features: {self.n_candidates}")
        print(f"  Samples:            {X.shape[0]}")
        print(f"  λ (trade-off):      {self.lambda_param}")

        # ── Step 1: Pre-filter to top candidates ─────────────────────
        print(f"\n▸ Pre-filtering to top {self.n_candidates} features by MI...")
        self.candidate_indices, X_filtered, self.relevance = \
            get_top_features_by_relevance(
                X, y, n_top=self.n_candidates,
                random_state=self.random_state
            )
        candidate_names = [feature_names[i] for i in self.candidate_indices]
        print(f"  Candidates: {candidate_names}")
        print(f"  Relevance:  {np.round(self.relevance, 4)}")

        # ── Step 2: Build QUBO ───────────────────────────────────────
        print(f"\n▸ Building QUBO matrix...")
        self.redundancy = compute_redundancy_matrix(X_filtered)
        self.Q = build_qubo_matrix(
            self.relevance, self.redundancy, self.lambda_param
        )
        print(f"  QUBO matrix shape: {self.Q.shape}")
        print(f"  QUBO diagonal (relevance): {np.round(np.diag(self.Q), 4)}")

        # ── Step 3: Convert to Ising ─────────────────────────────────
        print(f"\n▸ Converting QUBO → Ising Hamiltonian...")
        self.h, self.J, self.offset = qubo_to_ising(self.Q)
        print(f"  Linear coefficients (h): {np.round(self.h, 4)}")
        n_nonzero_J = np.sum(np.abs(self.J) > 1e-10)
        print(f"  Non-zero ZZ couplings:   {n_nonzero_J}")
        print(f"  Energy offset:           {self.offset:.4f}")

        # ── Step 4: Run QAOA Optimization ────────────────────────────
        self.optimizer = QAOAOptimizer(
            Q=self.Q, h=self.h, J=self.J, p=self.p,
            shots=self.shots, optimizer_method=self.optimizer_method,
            maxiter=self.maxiter, random_state=self.random_state,
        )
        self.optimization_result = self.optimizer.optimize()

        # ── Step 5: Extract Optimal Bitstring ────────────────────────
        print(f"\n▸ Extracting optimal solution...")
        counts = self.optimizer.get_optimal_distribution(shots=4096)

        # Sort by count (most probable first)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.top_bitstrings = sorted_counts[:10]

        print(f"\n  Top 5 measurement outcomes:")
        print(f"  {'Bitstring':<20} {'Count':>6} {'Prob':>8} {'QUBO Cost':>12}")
        print(f"  {'-'*48}")
        for bitstring, count in sorted_counts[:5]:
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            prob = count / 4096
            print(f"  {bitstring:<20} {count:>6} {prob:>8.4f} {cost:>12.6f}")

        # Best bitstring = lowest QUBO cost
        best_bitstring = None
        best_cost = float('inf')
        for bitstring, count in sorted_counts[:20]:
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            if cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring

        # Decode solution
        self.selected_mask = np.array(
            [int(b) for b in reversed(best_bitstring)]
        )

        # Map back to original feature indices
        self.selected_indices = self.candidate_indices[self.selected_mask == 1]

        self.computation_time = time.time() - start_time

        # ── Summary ──────────────────────────────────────────────────
        selected_names = [feature_names[i] for i in self.selected_indices]
        print(f"\n{'='*60}")
        print(f"  QAOA SOLUTION")
        print(f"{'='*60}")
        print(f"  Best bitstring:     {best_bitstring}")
        print(f"  QUBO cost:          {best_cost:.6f}")
        print(f"  Features selected:  {len(self.selected_indices)} / {n_features}")
        print(f"  Selected features:  {selected_names}")
        print(f"  Selected indices:   {self.selected_indices.tolist()}")
        print(f"  Computation time:   {self.computation_time:.1f}s")
        print(f"{'='*60}\n")

        return self

    def get_selected_features(self):
        """
        Get the indices of selected features.

        Returns
        -------
        indices : np.ndarray
            Indices into the original feature matrix.
        """
        if self.selected_indices is None:
            raise RuntimeError("Must call fit() first.")
        return self.selected_indices

    def get_selected_mask(self):
        """
        Get a boolean mask for selected features (original dimensionality).

        Returns
        -------
        mask : np.ndarray, shape (n_original_features,)
            Boolean mask where True indicates selection.
        """
        if self.selected_indices is None:
            raise RuntimeError("Must call fit() first.")
        # We only know the candidate subset mask; caller must use indices
        return self.selected_mask

    def transform(self, X):
        """
        Apply feature selection to a data matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_selected : np.ndarray, shape (n_samples, n_selected)
        """
        return X[:, self.selected_indices]

    def fit_transform(self, X, y, feature_names=None):
        """Fit and transform in one call."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_convergence_history(self):
        """Get optimization convergence data."""
        if self.optimizer is None:
            raise RuntimeError("Must call fit() first.")
        return self.optimizer.get_convergence_history()

    def get_summary(self):
        """Get a summary dict of results."""
        return {
            "n_candidates": self.n_candidates,
            "n_selected": len(self.selected_indices) if self.selected_indices is not None else 0,
            "selected_indices": self.selected_indices.tolist() if self.selected_indices is not None else [],
            "lambda_param": self.lambda_param,
            "p": self.p,
            "optimal_cost": self.optimization_result["optimal_cost"] if self.optimization_result else None,
            "num_iterations": self.optimization_result["num_iterations"] if self.optimization_result else None,
            "computation_time": self.computation_time,
        }
