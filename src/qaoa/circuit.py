"""
================================================================================
QAOA Circuit Construction
================================================================================

STEP 3: Build QAOA Circuit
---------------------------
The QAOA circuit implements the variational ansatz:

    |γ, β⟩ = Uₘ(βₚ) Uc(γₚ) ... Uₘ(β₁) Uc(γ₁) |+⟩ⁿ

Where:
    |+⟩ⁿ  = H⊗ⁿ |0⟩ⁿ  (uniform superposition via Hadamard gates)

Cost Unitary — Uc(γ) = e^{-iγH_C}:
    Since H_C = Σᵢ hᵢZᵢ + Σᵢ<ⱼ JᵢⱼZᵢZⱼ is diagonal, the exponential
    decomposes into a product of commuting terms:

    e^{-iγH_C} = Πᵢ e^{-iγhᵢZᵢ}  ×  Πᵢ<ⱼ e^{-iγJᵢⱼZᵢZⱼ}

    Implementation:
      • e^{-iγhZᵢ} → Rz(2γh) on qubit i
      • e^{-iγJZᵢZⱼ} → CNOT(i,j), Rz(2γJ) on j, CNOT(i,j)

Mixer Unitary — Uₘ(β) = e^{-iβH_M}:
    H_M = Σᵢ Xᵢ  (transverse field)
    e^{-iβH_M} = Πᵢ e^{-iβXᵢ} = Πᵢ Rx(2β)

    Implementation:
      • Rx(2β) on each qubit
================================================================================
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def build_initial_state(num_qubits):
    """
    Prepare the initial state |+⟩ⁿ = H⊗ⁿ |0⟩ⁿ.

    The uniform superposition is the starting point for QAOA, ensuring
    that all 2ⁿ possible solutions have equal initial amplitude.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (= number of candidate features).

    Returns
    -------
    qc : QuantumCircuit
        Circuit with Hadamard gates applied to all qubits.
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc


def build_cost_layer(qc, gamma, h, J, num_qubits):
    """
    Apply the cost unitary Uc(γ) = e^{-iγH_C}.

    Decomposes into:
      1. Single-qubit Rz rotations for linear (Z) terms
      2. Two-qubit CNOT-Rz-CNOT sequences for quadratic (ZZ) terms

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append gates to.
    gamma : Parameter or float
        Cost layer variational parameter.
    h : np.ndarray, shape (n,)
        Linear Ising coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic Ising coupling matrix (upper triangular).
    num_qubits : int
        Number of qubits.

    Returns
    -------
    qc : QuantumCircuit
        Circuit with cost layer appended.
    """
    # ZZ interaction terms: e^{-iγJ_{ij}Z_iZ_j}
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if abs(J[i, j]) > 1e-10:  # Skip negligible couplings
                qc.cx(i, j)
                qc.rz(2.0 * gamma * J[i, j], j)
                qc.cx(i, j)

    # Z field terms: e^{-iγh_iZ_i}
    for i in range(num_qubits):
        if abs(h[i]) > 1e-10:  # Skip negligible fields
            qc.rz(2.0 * gamma * h[i], i)

    return qc


def build_mixer_layer(qc, beta, num_qubits):
    """
    Apply the mixer unitary Uₘ(β) = e^{-iβH_M} = Πᵢ Rx(2β).

    The mixer drives transitions between computational basis states,
    enabling the algorithm to explore the solution space.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append gates to.
    beta : Parameter or float
        Mixer layer variational parameter.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    qc : QuantumCircuit
        Circuit with mixer layer appended.
    """
    for i in range(num_qubits):
        qc.rx(2.0 * beta, i)

    return qc


def build_qaoa_circuit(num_qubits, p, h, J):
    """
    Construct the complete parameterized QAOA ansatz.

    Circuit structure:
        1. Initialize |+⟩ⁿ (Hadamard on all qubits)
        2. For layer k = 1, ..., p:
           a. Apply cost unitary   Uc(γₖ)
           b. Apply mixer unitary  Uₘ(βₖ)
        3. Measure all qubits

    Parameters
    ----------
    num_qubits : int
        Number of qubits (= number of candidate features).
    p : int
        QAOA depth (number of alternating layers).
    h : np.ndarray, shape (n,)
        Linear Ising coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic Ising coupling matrix.

    Returns
    -------
    qc : QuantumCircuit
        Parameterized QAOA circuit (without measurement gates).
    gamma_params : list of Parameter
        List of γ parameters (one per layer).
    beta_params : list of Parameter
        List of β parameters (one per layer).
    """
    # Create parameterized variables
    gamma_params = [Parameter(f"γ_{k}") for k in range(p)]
    beta_params = [Parameter(f"β_{k}") for k in range(p)]

    # Build circuit
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: Initial superposition
    for i in range(num_qubits):
        qc.h(i)

    qc.barrier()

    # Step 2: Alternating cost and mixer layers
    for k in range(p):
        # Cost layer
        build_cost_layer(qc, gamma_params[k], h, J, num_qubits)
        qc.barrier()

        # Mixer layer
        build_mixer_layer(qc, beta_params[k], num_qubits)
        qc.barrier()

    return qc, gamma_params, beta_params


def build_qaoa_circuit_with_measurement(num_qubits, p, h, J):
    """
    Build QAOA circuit with measurement gates appended.

    Parameters
    ----------
    num_qubits : int
    p : int
    h : np.ndarray
    J : np.ndarray

    Returns
    -------
    qc : QuantumCircuit
        QAOA circuit with measurements.
    gamma_params : list of Parameter
    beta_params : list of Parameter
    """
    qc, gamma_params, beta_params = build_qaoa_circuit(num_qubits, p, h, J)

    # Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    return qc, gamma_params, beta_params


def get_circuit_info(qc):
    """
    Get summary information about the QAOA circuit.

    Parameters
    ----------
    qc : QuantumCircuit

    Returns
    -------
    info : dict
        Circuit statistics.
    """
    return {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "num_gates": qc.size(),
        "gate_counts": dict(qc.count_ops()),
    }
