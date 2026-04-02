"""
================================================================================
Cost Function & QUBO Formulation for Grid Partitioning
================================================================================

STEP 3: Define Cost Function
------------------------------
The grid partitioning problem assigns each node i to one of two partitions
(binary: x_i ∈ {0, 1}). The objective has two components:

1. EDGE CUT PENALTY:
   When two connected nodes are in different partitions, the transmission
   line (edge) between them is "cut". The cost of cutting edge (i,j) is
   proportional to its weight w_ij.

   C_cut = Σ_{(i,j)∈E} w_ij · (x_i ⊕ x_j)

   where x_i ⊕ x_j = x_i + x_j - 2·x_i·x_j = 1 when in different partitions.

   This is the standard MAX-CUT formulation (but we MINIMIZE it for
   grid partitioning, keeping correlated regions together).

   Actually for our problem, we MAXIMIZE edge cut of edges with HIGH weight
   (low correlation → different patterns → separate them), which is
   equivalent to the standard MAX-CUT problem.

   Alternatively, we minimize the cut cost of highly correlated edges.

2. LOAD BALANCING TERM:
   To ensure both partitions have roughly equal total load:

   C_balance = α · (Σ_i l_i·x_i - L/2)²

   where l_i = load of node i, L = total load, α = balancing penalty.

   Expanding: C_balance = α · (Σ_i l_i·x_i)² - α·L·(Σ_i l_i·x_i) + α·L²/4

COMBINED OBJECTIVE:
   minimize  C = -C_cut + α · C_balance

   The negative sign on C_cut means we MAXIMIZE the cut (separate dissimilar
   regions) while the balance term keeps partitions even.

STEP 4: QUBO Formulation
--------------------------
QUBO form: minimize x^T Q x + c^T x + const

For MAX-CUT (edge cut component):
   C_cut = Σ_{(i,j)∈E} w_ij · (x_i + x_j - 2·x_i·x_j)

   Maximizing this is equivalent to minimizing:
   -C_cut = Σ_{(i,j)∈E} w_ij · (2·x_i·x_j - x_i - x_j)

   QUBO contributions:
     Q_ij += 2·w_ij  (off-diagonal, for i<j, both directions)
     Q_ii += -w_ij   (diagonal, from -x_i for each edge containing i)

   Wait, let me be more careful. For the MAX-CUT formulation:
   We want to MAXIMIZE: Σ_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)
   Equivalently MINIMIZE: -Σ_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)
                        = Σ_{(i,j)} w_ij (2 x_i x_j - x_i - x_j)

   Q_ii += -Σ_{j:(i,j)∈E} w_ij   (from -x_i terms)
   Q_ij += w_ij                    (from 2 x_i x_j, split symmetrically)

For Load balancing:
   C_balance = α · (Σ_i l_i x_i - L/2)²
             = α · [Σ_i Σ_j l_i l_j x_i x_j - L·Σ_i l_i x_i + L²/4]

   Q_ii += α · l_i² - α · L · l_i   (from l_i² x_i² and -L l_i x_i)
   Q_ij += α · l_i · l_j             (from l_i l_j x_i x_j, for i≠j)

STEP 5: Map QUBO → Hamiltonian
---------------------------------
Substitute x_i = (1 - Z_i)/2:
   H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + offset
================================================================================
"""

import numpy as np
import networkx as nx


def build_maxcut_qubo(G):
    """
    Build the QUBO matrix for the MAX-CUT component.

    MAX-CUT: maximize Σ_{(i,j)} w_ij (x_i + x_j - 2x_i x_j)
    Minimization form: minimize Σ_{(i,j)} w_ij (2x_i x_j - x_i - x_j)

    Parameters
    ----------
    G : nx.Graph
        Weighted graph with 'weight' edge attribute.

    Returns
    -------
    Q_cut : np.ndarray, shape (n, n)
        QUBO matrix for the edge cut component.
    nodes : list
        Ordered node list (defines qubit mapping).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    Q_cut = np.zeros((n, n))

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        i = node_idx[u]
        j = node_idx[v]

        # Diagonal: -w_ij for each endpoint
        Q_cut[i, i] -= w
        Q_cut[j, j] -= w

        # Off-diagonal: +w_ij (symmetric)
        Q_cut[i, j] += w
        Q_cut[j, i] += w

    return Q_cut, nodes


def build_balance_qubo(G, nodes, alpha=1.0):
    """
    Build the QUBO matrix for the load balancing penalty.

    C_balance = α · (Σ_i l_i x_i - L/2)²

    Expansion:
      = α [Σ_i l_i² x_i² + 2·Σ_{i<j} l_i l_j x_i x_j - L·Σ_i l_i x_i + L²/4]
      = α [Σ_i (l_i² - L·l_i) x_i + 2·Σ_{i<j} l_i l_j x_i x_j] + const

    Parameters
    ----------
    G : nx.Graph
        Graph with 'load' node attribute.
    nodes : list
        Ordered node list.
    alpha : float
        Balancing penalty strength.

    Returns
    -------
    Q_bal : np.ndarray, shape (n, n)
        QUBO matrix for balancing.
    """
    n = len(nodes)
    loads = np.array([G.nodes[node]["load"] for node in nodes])
    L = loads.sum()

    # Normalize loads for numerical stability
    loads_norm = loads / L

    Q_bal = np.zeros((n, n))

    for i in range(n):
        # Diagonal: α · (l_i² - L·l_i) → using normalized: α·L² · (l̂_i² - l̂_i)
        Q_bal[i, i] = alpha * (loads_norm[i] ** 2 - loads_norm[i])

    for i in range(n):
        for j in range(i + 1, n):
            # Off-diagonal: α · 2·l_i·l_j → α · 2·L²·l̂_i·l̂_j
            val = alpha * 2.0 * loads_norm[i] * loads_norm[j]
            Q_bal[i, j] = val
            Q_bal[j, i] = val

    return Q_bal


def build_grid_partition_qubo(G, alpha=2.0):
    """
    Build the complete QUBO matrix for grid partitioning.

    Combined objective:
        minimize  Q_cut + α · Q_balance

    Parameters
    ----------
    G : nx.Graph
        Power grid graph with 'load' node and 'weight' edge attributes.
    alpha : float
        Load balancing penalty weight. Higher = more balanced partitions.

    Returns
    -------
    Q : np.ndarray, shape (n, n)
        Combined QUBO matrix.
    nodes : list
        Ordered node list (qubit mapping).
    """
    Q_cut, nodes = build_maxcut_qubo(G)
    Q_bal = build_balance_qubo(G, nodes, alpha=alpha)

    # Normalize Q_cut to have comparable scale with Q_bal
    cut_scale = np.abs(Q_cut).max()
    bal_scale = np.abs(Q_bal).max()

    if cut_scale > 0 and bal_scale > 0:
        # Scale so both components contribute comparably
        Q_cut_norm = Q_cut / cut_scale
        Q_bal_norm = Q_bal / bal_scale
        Q = Q_cut_norm + alpha * Q_bal_norm
    else:
        Q = Q_cut + alpha * Q_bal

    print(f"\n  QUBO Matrix ({Q.shape}):")
    print(f"    Cut component scale:     {cut_scale:.6f}")
    print(f"    Balance component scale: {bal_scale:.6f}")
    print(f"    α (balance weight):      {alpha}")
    print(f"    Combined Q range:        [{Q.min():.4f}, {Q.max():.4f}]")

    return Q, nodes


def qubo_to_ising(Q):
    """
    Convert QUBO matrix to Ising Hamiltonian coefficients.

    Substitution: x_i = (1 - Z_i) / 2

    H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + offset

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)

    Returns
    -------
    h : np.ndarray, shape (n,)
        Linear Z coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic ZZ couplings (upper triangular).
    offset : float
        Constant energy offset.
    """
    n = Q.shape[0]
    W = (Q + Q.T) / 2.0  # Symmetrize

    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        h[i] = -W[i, i] / 2.0
        for j in range(n):
            if j != i:
                h[i] -= W[i, j] / 2.0
        offset += W[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = W[i, j] / 2.0
            offset += W[i, j] / 2.0

    return h, J, offset


def qubo_objective(x, Q):
    """Evaluate QUBO cost: f(x) = x^T Q x."""
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)


def evaluate_partition(G, nodes, partition_bitstring):
    """
    Evaluate a partition assignment.

    Parameters
    ----------
    G : nx.Graph
    nodes : list
    partition_bitstring : array-like of 0s and 1s

    Returns
    -------
    metrics : dict
    """
    x = np.array(partition_bitstring, dtype=int)
    n = len(nodes)

    # Partition assignments
    partition_0 = [nodes[i] for i in range(n) if x[i] == 0]
    partition_1 = [nodes[i] for i in range(n) if x[i] == 1]

    # Load per partition
    load_0 = sum(G.nodes[node]["load"] for node in partition_0)
    load_1 = sum(G.nodes[node]["load"] for node in partition_1)
    total_load = load_0 + load_1
    imbalance = abs(load_0 - load_1) / total_load if total_load > 0 else 0

    # Edge cut
    cut_weight = 0.0
    cut_edges = 0
    total_edge_weight = 0.0
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        total_edge_weight += w
        ui = nodes.index(u)
        vi = nodes.index(v)
        if x[ui] != x[vi]:
            cut_weight += w
            cut_edges += 1

    return {
        "partition_0": partition_0,
        "partition_1": partition_1,
        "load_0": load_0,
        "load_1": load_1,
        "total_load": total_load,
        "load_imbalance": imbalance,
        "cut_edges": cut_edges,
        "cut_weight": cut_weight,
        "total_edge_weight": total_edge_weight,
    }
