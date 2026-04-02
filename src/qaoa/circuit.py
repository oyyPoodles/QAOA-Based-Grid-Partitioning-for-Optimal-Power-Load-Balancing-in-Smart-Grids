"""
================================================================================
QAOA Circuit Construction for Grid Partitioning
================================================================================

STEP 6: Build QAOA Circuit
----------------------------
The QAOA ansatz for graph partitioning:

    |γ, β⟩ = Uₘ(βₚ) Uc(γₚ) ... Uₘ(β₁) Uc(γ₁) |+⟩ⁿ

Where:
    |+⟩ⁿ = H⊗ⁿ |0⟩ⁿ  (uniform superposition)

Cost Unitary — Uc(γ) = e^{-iγH_C}:
    H_C = Σᵢ hᵢZᵢ + Σᵢ<ⱼ JᵢⱼZᵢZⱼ

    Decomposition (all terms commute since H_C is diagonal):
      e^{-iγhᵢZᵢ}      → Rz(2γhᵢ) on qubit i
      e^{-iγJᵢⱼZᵢZⱼ}   → CNOT(i,j), Rz(2γJᵢⱼ)(j), CNOT(i,j)

Mixer Unitary — Uₘ(β) = e^{-iβΣᵢXᵢ}:
    = Πᵢ Rx(2β)  on each qubit

Each qubit represents a grid node. The measurement outcome determines
which partition (0 or 1) each node is assigned to.
================================================================================
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def build_qaoa_circuit(num_qubits, p, h, J):
    """
    Build the parameterized QAOA circuit for grid partitioning.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (= number of grid nodes).
    p : int
        QAOA depth (number of alternating layers).
    h : np.ndarray, shape (n,)
        Linear Ising coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic Ising coupling matrix (upper triangular).

    Returns
    -------
    qc : QuantumCircuit
        Parameterized QAOA circuit with measurement.
    gamma_params : list of Parameter
    beta_params : list of Parameter
    """
    gamma_params = [Parameter(f"gamma_{k}") for k in range(p)]
    beta_params = [Parameter(f"beta_{k}") for k in range(p)]

    qc = QuantumCircuit(num_qubits, num_qubits)

    # ── Initial superposition |+⟩^n ──
    for i in range(num_qubits):
        qc.h(i)
    qc.barrier()

    # ── Alternating QAOA layers ──
    for k in range(p):
        # Cost layer: Uc(γ_k) = exp(-iγ H_C)
        has_cost_gate = False

        # ZZ terms
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2.0 * gamma_params[k] * J[i, j], j)
                    qc.cx(i, j)
                    has_cost_gate = True

        # Z terms
        for i in range(num_qubits):
            if abs(h[i]) > 1e-10:
                qc.rz(2.0 * gamma_params[k] * h[i], i)
                has_cost_gate = True

        # Ensure gamma parameter is always used (identity rotation if needed)
        if not has_cost_gate:
            qc.rz(gamma_params[k] * 0.0001, 0)

        qc.barrier()

        # Mixer layer: Um(β_k) = exp(-iβ Σ Xi)
        for i in range(num_qubits):
            qc.rx(2.0 * beta_params[k], i)
        qc.barrier()

    # Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    return qc, gamma_params, beta_params


def get_circuit_info(qc):
    """Get summary statistics of the QAOA circuit."""
    return {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "num_gates": qc.size(),
        "gate_counts": dict(qc.count_ops()),
    }
