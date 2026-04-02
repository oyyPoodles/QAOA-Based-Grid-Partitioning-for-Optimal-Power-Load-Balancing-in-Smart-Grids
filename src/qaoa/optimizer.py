"""
================================================================================
Classical Optimization Loop for QAOA
================================================================================

STEP 4: Classical Optimization
-------------------------------
QAOA is a variational algorithm: the quantum circuit is parameterized by
angles (γ₁,...,γₚ, β₁,...,βₚ), and a classical optimizer tunes these
parameters to minimize the expectation value of the cost Hamiltonian:

    min_{γ,β}  ⟨γ,β| H_C |γ,β⟩  =  min_{γ,β} Σₓ P(x|γ,β) · f(x)

Optimization Procedure:
    1. Initialize parameters (γ, β) randomly or heuristically
    2. Construct and run the QAOA circuit with current parameters
    3. Sample bitstrings from measurement outcomes
    4. Compute expected cost: E[f] = Σₓ (count(x)/shots) · QUBO_cost(x)
    5. Feed E[f] back to the classical optimizer
    6. Optimizer proposes new parameters
    7. Repeat until convergence

Supported Optimizers:
    • COBYLA: Constrained Optimization BY Linear Approximation
      - Derivative-free, works well with noisy objective functions
      - Good for quantum optimization with shot noise
    • SPSA: Simultaneous Perturbation Stochastic Approximation
      - Uses random perturbations to estimate gradients
      - Robust to noise, suitable for real quantum hardware
================================================================================
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import transpile

# Try to import AerSimulator; fall back gracefully
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

from .circuit import build_qaoa_circuit_with_measurement
from .hamiltonian import qubo_objective


class QAOAOptimizer:
    """
    Classical optimization loop for QAOA parameter tuning.

    This class handles:
      - Building the parameterized QAOA circuit
      - Running it on a quantum simulator
      - Computing the cost function from measurement samples
      - Interfacing with a classical optimizer (COBYLA/SPSA)
      - Tracking convergence history

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        QUBO matrix encoding the optimization problem.
    h : np.ndarray, shape (n,)
        Linear Ising coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic Ising coupling matrix.
    p : int, default=1
        QAOA circuit depth (number of layers).
    shots : int, default=1024
        Number of measurement shots per circuit evaluation.
    optimizer_method : str, default='COBYLA'
        Classical optimizer ('COBYLA' or 'SPSA').
    maxiter : int, default=200
        Maximum number of optimizer iterations.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, Q, h, J, p=1, shots=1024, optimizer_method="COBYLA",
                 maxiter=200, random_state=42):
        self.Q = Q
        self.h = h
        self.J = J
        self.num_qubits = Q.shape[0]
        self.p = p
        self.shots = shots
        self.optimizer_method = optimizer_method
        self.maxiter = maxiter
        self.random_state = random_state

        # Build the parameterized circuit
        self.circuit, self.gamma_params, self.beta_params = \
            build_qaoa_circuit_with_measurement(self.num_qubits, p, h, J)

        # Initialize simulator backend
        self.backend = AerSimulator()

        # Convergence tracking
        self.cost_history = []
        self.param_history = []
        self.iteration_count = 0

        # Results
        self.optimal_params = None
        self.optimal_cost = None
        self.optimal_counts = None

    def _evaluate_cost(self, params):
        """
        Evaluate the expected cost function for given parameters.

        Process:
          1. Bind parameter values to the QAOA circuit
          2. Transpile and run on simulator
          3. Compute weighted average cost over sampled bitstrings

        Parameters
        ----------
        params : np.ndarray, shape (2*p,)
            Parameter vector [γ₁,...,γₚ, β₁,...,βₚ].

        Returns
        -------
        expected_cost : float
            ⟨H_C⟩ = Σₓ P(x) · f(x)
        """
        # Split parameters
        gammas = params[:self.p]
        betas = params[self.p:]

        # Create parameter binding dictionary
        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        # Bind parameters to circuit
        bound_circuit = self.circuit.assign_parameters(param_dict)

        # Transpile and run
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Compute expected cost from samples
        expected_cost = 0.0
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            # Qiskit returns bitstrings in reverse order (qubit 0 is rightmost)
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            expected_cost += (count / total_shots) * cost

        # Track convergence
        self.cost_history.append(expected_cost)
        self.param_history.append(params.copy())
        self.iteration_count += 1

        if self.iteration_count % 25 == 0:
            print(f"  Iteration {self.iteration_count}: "
                  f"Expected Cost = {expected_cost:.6f}")

        return expected_cost

    def optimize(self):
        """
        Run the classical optimization loop.

        Initializes parameters and runs the chosen optimizer to find
        the optimal (γ*, β*) that minimize ⟨H_C⟩.

        Returns
        -------
        result : dict
            Optimization result containing:
              - 'optimal_params': best parameter values
              - 'optimal_cost': minimum expected cost
              - 'num_iterations': total iterations
              - 'convergence_history': cost values per iteration
        """
        print(f"\n{'='*60}")
        print(f"QAOA Optimization")
        print(f"{'='*60}")
        print(f"  Qubits:    {self.num_qubits}")
        print(f"  Depth (p): {self.p}")
        print(f"  Shots:     {self.shots}")
        print(f"  Optimizer: {self.optimizer_method}")
        print(f"  Max Iter:  {self.maxiter}")
        print(f"{'='*60}")

        # Reset tracking
        self.cost_history = []
        self.param_history = []
        self.iteration_count = 0

        # Initialize parameters
        np.random.seed(self.random_state)
        # γ ∈ [0, 2π], β ∈ [0, π] (standard QAOA parameter ranges)
        initial_gammas = np.random.uniform(0, 2 * np.pi, self.p)
        initial_betas = np.random.uniform(0, np.pi, self.p)
        initial_params = np.concatenate([initial_gammas, initial_betas])

        print(f"\n  Initial γ: {initial_gammas}")
        print(f"  Initial β: {initial_betas}")
        print(f"\n  Optimizing...\n")

        # Run optimizer
        if self.optimizer_method == "COBYLA":
            result = minimize(
                self._evaluate_cost,
                initial_params,
                method="COBYLA",
                options={"maxiter": self.maxiter, "rhobeg": 0.5},
            )
        elif self.optimizer_method == "SPSA":
            # Simple SPSA implementation
            result = self._run_spsa(initial_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_method}")

        self.optimal_params = result.x if hasattr(result, 'x') else result['x']
        self.optimal_cost = self.cost_history[-1] if self.cost_history else None

        print(f"\n  Optimization complete!")
        print(f"  Final Cost:  {self.optimal_cost:.6f}")
        print(f"  Iterations:  {self.iteration_count}")

        return {
            "optimal_params": self.optimal_params,
            "optimal_cost": self.optimal_cost,
            "num_iterations": self.iteration_count,
            "convergence_history": self.cost_history.copy(),
        }

    def _run_spsa(self, initial_params, a=0.1, c=0.1, alpha=0.602, gamma_decay=0.101):
        """
        Simultaneous Perturbation Stochastic Approximation (SPSA).

        SPSA estimates the gradient using only 2 function evaluations
        per iteration regardless of the parameter dimension.

        Parameters
        ----------
        initial_params : np.ndarray
        a, c : float
            Step size parameters.
        alpha, gamma_decay : float
            Decay rates for step sizes.

        Returns
        -------
        result : dict with 'x' key containing optimal parameters.
        """
        params = initial_params.copy()
        n_params = len(params)

        for k in range(1, self.maxiter + 1):
            ak = a / (k ** alpha)
            ck = c / (k ** gamma_decay)

            # Random perturbation direction (Bernoulli ±1)
            delta = np.random.choice([-1, 1], size=n_params)

            # Evaluate at perturbed points
            cost_plus = self._evaluate_cost(params + ck * delta)
            cost_minus = self._evaluate_cost(params - ck * delta)

            # Estimate gradient
            gradient = (cost_plus - cost_minus) / (2 * ck * delta)

            # Update parameters
            params = params - ak * gradient

        return {"x": params}

    def get_optimal_distribution(self, shots=4096):
        """
        Run the optimized circuit with more shots to get a reliable
        probability distribution over solutions.

        Parameters
        ----------
        shots : int
            Number of measurement shots.

        Returns
        -------
        counts : dict
            Measurement counts {bitstring: count}.
        """
        if self.optimal_params is None:
            raise RuntimeError("Must call optimize() first.")

        gammas = self.optimal_params[:self.p]
        betas = self.optimal_params[self.p:]

        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        bound_circuit = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        result = job.result()
        self.optimal_counts = result.get_counts()

        return self.optimal_counts

    def get_convergence_history(self):
        """Get the optimization convergence history."""
        return {
            "costs": self.cost_history.copy(),
            "params": self.param_history.copy(),
        }
