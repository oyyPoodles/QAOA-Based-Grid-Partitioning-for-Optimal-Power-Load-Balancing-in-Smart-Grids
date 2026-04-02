"""
================================================================================
QAOA Hamiltonian Construction for Feature Selection
================================================================================

STEP 1: QAOA Explanation
------------------------
The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-
classical variational algorithm introduced by Farhi, Goldstone, and Gutmann
(2014). It is designed to find approximate solutions to combinatorial
optimization problems.

**Key Components:**

1. COST HAMILTONIAN (H_C):
   Encodes the objective function of the optimization problem. For a
   combinatorial problem with objective f(x), the cost Hamiltonian is
   diagonal in the computational basis:
       H_C |x⟩ = f(x) |x⟩
   The ground state of H_C corresponds to the optimal solution.

2. MIXER HAMILTONIAN (H_M):
   Drives transitions between computational basis states to explore the
   solution space. The standard mixer is the transverse-field Hamiltonian:
       H_M = Σᵢ Xᵢ
   where Xᵢ is the Pauli-X operator on qubit i.

3. PARAMETERIZED QUANTUM CIRCUIT:
   The QAOA ansatz with depth p applies alternating layers:
       |γ, β⟩ = Uₘ(βₚ) Uc(γₚ) ... Uₘ(β₁) Uc(γ₁) |+⟩ⁿ
   where:
       Uc(γ) = e^{-iγH_C}   (cost unitary)
       Uₘ(β) = e^{-iβH_M}   (mixer unitary)

**Feature Selection as Optimization:**
   Feature selection can be naturally mapped to a combinatorial optimization
   problem. Given n features, we define a binary vector x ∈ {0,1}ⁿ where
   xᵢ = 1 indicates feature i is selected. The objective is to find x that
   maximizes predictive relevance while minimizing feature redundancy.

STEP 2: Mathematical Formulation
---------------------------------
Binary Feature Selection Vector:
   x ∈ {0, 1}ⁿ where n = number of features

Objective Function:
   maximize  f(x) = Σᵢ rᵢ·xᵢ  -  λ · Σᵢ<ⱼ cᵢⱼ·xᵢ·xⱼ
                     ↑ Relevance      ↑ Redundancy

   where:
     rᵢ  = mutual information between feature i and target variable
     cᵢⱼ = |correlation(feature_i, feature_j)| (absolute Pearson correlation)
     λ   = trade-off parameter controlling relevance vs redundancy

QUBO Formulation:
   We convert to minimization:  minimize g(x) = -f(x) = x^T Q x

   where Q is the QUBO matrix:
     Q_{ii} = -rᵢ                    (diagonal: negative relevance)
     Q_{ij} = λ·cᵢⱼ / 2  for i ≠ j  (off-diagonal: redundancy penalty)

Ising Hamiltonian:
   Substituting xᵢ = (I - Zᵢ)/2  (mapping binary → spin variables):

   H_C = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ + offset

   where:
     Jᵢⱼ = Wᵢⱼ / 2                             (ZZ coupling strength)
     hᵢ  = -Wᵢᵢ/2 - Σⱼ≠ᵢ Wᵢⱼ/2                (Z field strength)
     Wᵢⱼ = (Qᵢⱼ + Qⱼᵢ)/2  (symmetrized QUBO)
     offset = Σᵢ Wᵢᵢ/2 + Σᵢ<ⱼ Wᵢⱼ/2
================================================================================
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_relevance_scores(X, y, random_state=42):
    """
    Compute feature relevance scores using mutual information.

    Mutual information I(Xᵢ; Y) measures the statistical dependence between
    feature Xᵢ and target Y. Higher values indicate more informative features.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    relevance : np.ndarray, shape (n_features,)
        Mutual information scores for each feature.
    """
    relevance = mutual_info_classif(X, y, random_state=random_state)

    # Normalize to [0, 1] for numerical stability in QUBO
    if relevance.max() > 0:
        relevance = relevance / relevance.max()

    return relevance


def compute_redundancy_matrix(X):
    """
    Compute pairwise feature redundancy using absolute Pearson correlation.

    |corr(Xᵢ, Xⱼ)| measures linear dependency between features. High values
    indicate redundant feature pairs that carry overlapping information.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    redundancy : np.ndarray, shape (n_features, n_features)
        Absolute correlation matrix with diagonal set to zero.
    """
    corr = np.corrcoef(X.T)
    redundancy = np.abs(corr)

    # Zero out diagonal (self-correlation is not redundancy)
    np.fill_diagonal(redundancy, 0.0)

    return redundancy


def build_qubo_matrix(relevance, redundancy, lambda_param=0.5):
    """
    Construct the QUBO matrix Q for the feature selection problem.

    QUBO Objective (minimization):
        g(x) = x^T Q x = Σᵢ Q_{ii} xᵢ + Σᵢ≠ⱼ Q_{ij} xᵢ xⱼ

    where:
        Q_{ii} = -rᵢ                      (feature relevance, negated)
        Q_{ij} = λ · cᵢⱼ / 2  for i ≠ j   (redundancy penalty)

    The minimizer of g(x) selects features that are highly relevant
    and minimally redundant.

    Parameters
    ----------
    relevance : np.ndarray, shape (n_features,)
        Relevance scores for each feature.
    redundancy : np.ndarray, shape (n_features, n_features)
        Pairwise redundancy matrix.
    lambda_param : float
        Trade-off parameter. Higher λ → stronger redundancy penalty.

    Returns
    -------
    Q : np.ndarray, shape (n_features, n_features)
        QUBO matrix (symmetric).
    """
    n = len(relevance)
    Q = np.zeros((n, n))

    # Diagonal: negative relevance (we minimize, so negate to maximize relevance)
    for i in range(n):
        Q[i, i] = -relevance[i]

    # Off-diagonal: redundancy penalty (symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = lambda_param * redundancy[i, j] / 2.0
            Q[j, i] = lambda_param * redundancy[i, j] / 2.0

    return Q


def qubo_to_ising(Q):
    """
    Convert a QUBO matrix to Ising Hamiltonian coefficients.

    The substitution xᵢ = (1 - Zᵢ)/2 transforms the QUBO into:
        H_C = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ + offset

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Symmetric QUBO matrix.

    Returns
    -------
    h : np.ndarray, shape (n,)
        Linear (Z) coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic (ZZ) coupling matrix (upper triangular).
    offset : float
        Constant energy offset.
    """
    n = Q.shape[0]

    # Symmetrize: W = (Q + Q^T) / 2
    W = (Q + Q.T) / 2.0

    # Compute Ising coefficients
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        # Linear field: h_i = -W_ii/2 - (1/2) Σ_{j≠i} W_ij
        h[i] = -W[i, i] / 2.0 - np.sum(W[i, :]) / 2.0 + W[i, i] / 2.0
        # Simplifies to: h_i = -W_ii/2 - Σ_{j≠i} W_ij / 2
        h[i] = -W[i, i] / 2.0
        for j in range(n):
            if j != i:
                h[i] -= W[i, j] / 2.0

        # Offset contribution from diagonal
        offset += W[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            # ZZ coupling: J_ij = W_ij / 2
            J[i, j] = W[i, j] / 2.0
            # Offset contribution from off-diagonal
            offset += W[i, j] / 2.0

    return h, J, offset


def qubo_objective(x, Q):
    """
    Evaluate the QUBO objective function for a binary vector x.

    f(x) = x^T Q x

    Parameters
    ----------
    x : np.ndarray or list, shape (n,)
        Binary vector (0s and 1s).
    Q : np.ndarray, shape (n, n)
        QUBO matrix.

    Returns
    -------
    cost : float
        Objective function value.
    """
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)


def get_top_features_by_relevance(X, y, n_top=10, random_state=42):
    """
    Pre-filter features by mutual information to reduce qubit requirements.

    For practical quantum simulation, we limit the number of qubits by
    selecting the top-n most relevant features as candidates for QAOA.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    n_top : int
        Number of top features to retain.
    random_state : int

    Returns
    -------
    top_indices : np.ndarray, shape (n_top,)
        Indices of the top features.
    X_filtered : np.ndarray, shape (n_samples, n_top)
        Filtered feature matrix.
    relevance_filtered : np.ndarray, shape (n_top,)
        Relevance scores for filtered features.
    """
    relevance = mutual_info_classif(X, y, random_state=random_state)
    top_indices = np.argsort(relevance)[::-1][:n_top]
    top_indices = np.sort(top_indices)  # Keep original order

    X_filtered = X[:, top_indices]
    relevance_filtered = relevance[top_indices]

    # Normalize
    if relevance_filtered.max() > 0:
        relevance_filtered = relevance_filtered / relevance_filtered.max()

    return top_indices, X_filtered, relevance_filtered
