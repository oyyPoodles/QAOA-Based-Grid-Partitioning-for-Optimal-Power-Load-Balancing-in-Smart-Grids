"""
================================================================================
Classical Optimization Loop for QAOA
================================================================================

STEP 7: Classical Optimization
--------------------------------
Variational loop:
  1. Initialize parameters (γ, β) randomly
  2. Run QAOA circuit on simulator → measurement counts
  3. Compute expected cost: ⟨H_C⟩ = Σ_x P(x) · QUBO_cost(x)
  4. COBYLA optimizer proposes new (γ, β)
  5. Repeat until convergence

STEP 8: Extract Optimal Partition
-----------------------------------
After optimization:
  1. Run circuit with optimal (γ*, β*) with many shots
  2. Find most probable bitstring
  3. Also find bitstring with lowest QUBO cost
  4. Map bitstring to partition: bit i = 0 → Partition A, bit i = 1 → Partition B
================================================================================
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import transpile

try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

from .circuit import build_qaoa_circuit
from .hamiltonian import qubo_objective, qubo_to_ising, evaluate_partition


class QAOAGridOptimizer:
    """
    QAOA optimizer for grid partitioning.

    Parameters
    ----------
    G : nx.Graph
        Power grid graph.
    Q : np.ndarray
        QUBO matrix.
    nodes : list
        Ordered node list.
    p : int
        QAOA depth.
    shots : int
        Measurement shots per evaluation.
    maxiter : int
        Max optimizer iterations.
    random_state : int
    """

    def __init__(self, G, Q, nodes, p=1, shots=1024, maxiter=200,
                 random_state=42):
        self.G = G
        self.Q = Q
        self.nodes = nodes
        self.num_qubits = len(nodes)
        self.p = p
        self.shots = shots
        self.maxiter = maxiter
        self.random_state = random_state

        # Convert QUBO → Ising
        self.h, self.J, self.offset = qubo_to_ising(Q)

        # Build circuit
        self.circuit, self.gamma_params, self.beta_params = \
            build_qaoa_circuit(self.num_qubits, p, self.h, self.J)

        # Simulator
        self.backend = AerSimulator()

        # Tracking
        self.cost_history = []
        self.iteration_count = 0
        self.optimal_params = None
        self.optimal_counts = None
        self.optimal_partition = None

    def _evaluate_cost(self, params):
        """Compute ⟨H_C⟩ from circuit measurements."""
        gammas = params[:self.p]
        betas = params[self.p:]

        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        bound_circuit = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        counts = job.result().get_counts()

        expected_cost = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            expected_cost += (count / total) * cost

        self.cost_history.append(expected_cost)
        self.iteration_count += 1

        if self.iteration_count % 20 == 0:
            print(f"    Iter {self.iteration_count}: ⟨Cost⟩ = {expected_cost:.6f}")

        return expected_cost

    def optimize(self):
        """
        Run the COBYLA optimization loop.

        Returns
        -------
        result : dict
        """
        print(f"\n{'='*60}")
        print(f"  QAOA OPTIMIZATION (Grid Partitioning)")
        print(f"{'='*60}")
        print(f"  Qubits:    {self.num_qubits}")
        print(f"  Depth (p): {self.p}")
        print(f"  Shots:     {self.shots}")
        print(f"  Max iter:  {self.maxiter}")

        self.cost_history = []
        self.iteration_count = 0

        np.random.seed(self.random_state)
        init_params = np.concatenate([
            np.random.uniform(0, 2 * np.pi, self.p),
            np.random.uniform(0, np.pi, self.p),
        ])

        print(f"\n  Optimizing...")
        result = minimize(
            self._evaluate_cost, init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.5},
        )

        self.optimal_params = result.x
        print(f"\n  ✓ Converged after {self.iteration_count} iterations")
        print(f"  ✓ Final ⟨Cost⟩: {self.cost_history[-1]:.6f}")

        return {
            "optimal_params": self.optimal_params,
            "optimal_cost": self.cost_history[-1],
            "num_iterations": self.iteration_count,
            "convergence_history": self.cost_history.copy(),
        }

    def extract_partition(self, shots=8192):
        """
        Extract the optimal partition from the optimized circuit.

        Returns
        -------
        partition : dict
            Contains partition assignment, metrics, top bitstrings.
        """
        if self.optimal_params is None:
            raise RuntimeError("Must call optimize() first.")

        print(f"\n  Extracting optimal partition ({shots} shots)...")

        gammas = self.optimal_params[:self.p]
        betas = self.optimal_params[self.p:]
        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        bound = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        self.optimal_counts = job.result().get_counts()

        # Find best bitstring by QUBO cost
        sorted_counts = sorted(
            self.optimal_counts.items(), key=lambda x: x[1], reverse=True
        )

        print(f"\n  Top 10 measurement outcomes:")
        print(f"  {'Bitstring':<15} {'Count':>6} {'Prob':>8} {'QUBO Cost':>12}")
        print(f"  {'-'*43}")

        best_bitstring = None
        best_cost = float("inf")

        for bitstring, count in sorted_counts[:10]:
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            prob = count / shots
            marker = ""
            if cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring
                marker = " ← best"
            print(f"  {bitstring:<15} {count:>6} {prob:>8.4f} {cost:>12.6f}{marker}")

        # Decode partition
        partition_bits = np.array([int(b) for b in reversed(best_bitstring)])
        metrics = evaluate_partition(self.G, self.nodes, partition_bits)

        self.optimal_partition = {
            "bitstring": best_bitstring,
            "bits": partition_bits,
            "qubo_cost": best_cost,
            "metrics": metrics,
        }

        print(f"\n{'='*60}")
        print(f"  OPTIMAL PARTITION")
        print(f"{'='*60}")
        print(f"  Bitstring:   {best_bitstring}")
        print(f"  QUBO cost:   {best_cost:.6f}")
        print(f"\n  Partition A: {metrics['partition_0']}")
        print(f"  Partition B: {metrics['partition_1']}")
        print(f"  Load A:      {metrics['load_0']:,.0f} MW")
        print(f"  Load B:      {metrics['load_1']:,.0f} MW")
        print(f"  Imbalance:   {metrics['load_imbalance']*100:.2f}%")
        print(f"  Edges cut:   {metrics['cut_edges']} / {self.G.number_of_edges()}")
        print(f"  Cut weight:  {metrics['cut_weight']:.4f} / {metrics['total_edge_weight']:.4f}")
        print(f"{'='*60}")

        return self.optimal_partition
