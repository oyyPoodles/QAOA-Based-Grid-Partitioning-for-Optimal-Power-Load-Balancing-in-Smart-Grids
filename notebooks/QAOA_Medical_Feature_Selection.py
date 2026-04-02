# %% [markdown]
# # 🧬 Hybrid Quantum-Classical Optimization using QAOA
# # for Medical Feature Selection
#
# **Complete End-to-End Pipeline — Google Colab Ready**
#
# This notebook implements a hybrid quantum-classical approach to medical
# feature selection using the Quantum Approximate Optimization Algorithm (QAOA).
#
# ---
#
# ## Table of Contents
# 1. **Phase 1 — Quantum Foundation**
#    - Step 1: QAOA Explanation
#    - Step 2: Mathematical Formulation
#    - Step 3: Build QAOA Circuit
#    - Step 4: Classical Optimization Loop
#    - Step 5: Extract Optimal Bitstring
# 2. **Phase 2 — Data & ML Integration**
#    - Step 6: Dataset Handling
#    - Step 7: Apply QAOA Feature Selection
#    - Step 8: Train ML Model
#    - Step 9: Evaluation
# 3. **Phase 3 — Classical Baseline Comparison**
#    - Step 10: Implement Classical Methods
#    - Step 11: Benchmarking
# 4. **Phase 4 — Visualization & Analysis**
#    - Step 12: Comprehensive Visualizations
# 5. **Phase 5 — System Design** (see docs/architecture.md)
# 6. **Phase 6 — Research Output**
#    - Step 15: Research Paper Sections

# %% [markdown]
# ## 📦 Setup & Installation
# Run this cell first to install required dependencies.

# %%
# Install dependencies (uncomment for Google Colab)
# !pip install qiskit qiskit-aer qiskit-algorithms qiskit-optimization
# !pip install scikit-learn numpy pandas matplotlib seaborn scipy tabulate

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

# Scikit-learn imports
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from scipy.optimize import minimize

print("✅ All imports successful!")
print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ---
# # 🧠 PHASE 1 — QUANTUM FOUNDATION
# ---

# %% [markdown]
# ## Step 1: QAOA Explanation
#
# ### What is QAOA?
#
# The **Quantum Approximate Optimization Algorithm (QAOA)** is a hybrid
# quantum-classical variational algorithm designed to solve combinatorial
# optimization problems. Introduced by Farhi, Goldstone, and Gutmann (2014).
#
# ### Key Components
#
# **1. Cost Hamiltonian (H_C):**
# Encodes the objective function. For objective f(x):
# $$H_C |x\rangle = f(x) |x\rangle$$
# The ground state of H_C is the optimal solution.
#
# **2. Mixer Hamiltonian (H_M):**
# Drives exploration of the solution space:
# $$H_M = \sum_i X_i$$
#
# **3. Parameterized Quantum Circuit:**
# $$|\gamma, \beta\rangle = U_M(\beta_p) U_C(\gamma_p) \cdots U_M(\beta_1) U_C(\gamma_1) |+\rangle^n$$
#
# where:
# - $U_C(\gamma) = e^{-i\gamma H_C}$ (cost unitary)
# - $U_M(\beta) = e^{-i\beta H_M}$ (mixer unitary)
#
# ### Feature Selection as Optimization
#
# Feature selection is naturally a combinatorial problem:
# - **n features** → binary vector x ∈ {0,1}^n
# - x_i = 1 → feature i is selected
# - Objective: maximize relevance, minimize redundancy
# - Search space: 2^n possible subsets → perfect for quantum!

# %% [markdown]
# ## Step 2: Mathematical Formulation
#
# ### Binary Feature Selection Vector
# $$\mathbf{x} \in \{0, 1\}^n \quad \text{where } x_i = 1 \text{ means feature } i \text{ is selected}$$
#
# ### Objective Function
# $$\max_{\mathbf{x}} \sum_i r_i x_i - \lambda \sum_{i<j} c_{ij} x_i x_j$$
#
# where:
# - $r_i = I(X_i; Y)$ — mutual information (relevance)
# - $c_{ij} = |\text{corr}(X_i, X_j)|$ — absolute correlation (redundancy)
# - $\lambda$ — trade-off parameter
#
# ### QUBO Form
# Convert to minimization: $\min_{\mathbf{x}} \mathbf{x}^T Q \mathbf{x}$
#
# $$Q_{ii} = -r_i, \quad Q_{ij} = \frac{\lambda \cdot c_{ij}}{2} \text{ for } i \neq j$$
#
# ### Ising Hamiltonian
# Substitute $x_i = (1 - Z_i)/2$:
# $$H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

# %%
# =====================================================================
# STEP 2: Mathematical Formulation — Implementation
# =====================================================================

def compute_relevance_scores(X, y, random_state=42):
    """
    Compute feature relevance using mutual information I(X_i; Y).
    Higher MI → more informative feature.
    """
    relevance = mutual_info_classif(X, y, random_state=random_state)
    if relevance.max() > 0:
        relevance = relevance / relevance.max()
    return relevance


def compute_redundancy_matrix(X):
    """
    Compute pairwise redundancy using |Pearson correlation|.
    High |corr| → redundant features.
    """
    corr = np.corrcoef(X.T)
    redundancy = np.abs(corr)
    np.fill_diagonal(redundancy, 0.0)
    return redundancy


def build_qubo_matrix(relevance, redundancy, lambda_param=0.5):
    """
    Build QUBO matrix Q.
    Q_ii = -r_i (maximize relevance)
    Q_ij = λ·c_ij/2 (minimize redundancy)
    """
    n = len(relevance)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -relevance[i]
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = lambda_param * redundancy[i, j] / 2.0
            Q[j, i] = lambda_param * redundancy[i, j] / 2.0
    return Q


def qubo_to_ising(Q):
    """
    Convert QUBO to Ising Hamiltonian coefficients.
    x_i = (1 - Z_i)/2 → H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j + offset
    """
    n = Q.shape[0]
    W = (Q + Q.T) / 2.0
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
    """Evaluate QUBO cost: f(x) = x^T Q x"""
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)


print("✅ Step 2: Mathematical formulation functions defined")
print("   • compute_relevance_scores(X, y)")
print("   • compute_redundancy_matrix(X)")
print("   • build_qubo_matrix(relevance, redundancy, λ)")
print("   • qubo_to_ising(Q)")
print("   • qubo_objective(x, Q)")

# %% [markdown]
# ## Step 3: Build QAOA Circuit
#
# ### Circuit Structure
#
# ```
# |0⟩ ─ H ─┤ Cost Layer (γ) ├─┤ Mixer Layer (β) ├─ Measure
# |0⟩ ─ H ─┤               ├─┤                ├─ Measure
# ...       └───────────────┘ └────────────────┘
# ```
#
# **Cost Layer** $U_C(\gamma) = e^{-i\gamma H_C}$:
# - Z terms: $e^{-i\gamma h_i Z_i}$ → `Rz(2γh_i)`
# - ZZ terms: $e^{-i\gamma J_{ij} Z_i Z_j}$ → `CNOT, Rz(2γJ_ij), CNOT`
#
# **Mixer Layer** $U_M(\beta) = e^{-i\beta H_M}$:
# - X terms: $e^{-i\beta X_i}$ → `Rx(2β)`

# %%
# =====================================================================
# STEP 3: Build QAOA Circuit
# =====================================================================

def build_qaoa_circuit(num_qubits, p, h, J):
    """
    Build the full parameterized QAOA ansatz.

    1. |+⟩^n initialization (Hadamard on all qubits)
    2. p layers of:
       a. Cost unitary Uc(γ_k) — Rz and CNOT-Rz-CNOT gates
       b. Mixer unitary Um(β_k) — Rx gates
    3. Measurement

    Parameters
    ----------
    num_qubits : number of qubits (= candidate features)
    p : QAOA depth (number of layers)
    h : linear Ising coefficients
    J : quadratic Ising coupling matrix
    """
    gamma_params = [Parameter(f"γ_{k}") for k in range(p)]
    beta_params = [Parameter(f"β_{k}") for k in range(p)]

    qc = QuantumCircuit(num_qubits, num_qubits)

    # Initial superposition: |+⟩^n
    for i in range(num_qubits):
        qc.h(i)
    qc.barrier()

    # Alternating layers
    for k in range(p):
        # ── Cost Layer ──
        # ZZ interactions
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2.0 * gamma_params[k] * J[i, j], j)
                    qc.cx(i, j)
        # Z fields
        for i in range(num_qubits):
            if abs(h[i]) > 1e-10:
                qc.rz(2.0 * gamma_params[k] * h[i], i)
        qc.barrier()

        # ── Mixer Layer ──
        for i in range(num_qubits):
            qc.rx(2.0 * beta_params[k], i)
        qc.barrier()

    # Measurement
    qc.measure(range(num_qubits), range(num_qubits))

    return qc, gamma_params, beta_params


print("✅ Step 3: QAOA circuit builder defined")
print("   • build_qaoa_circuit(num_qubits, p, h, J)")

# %% [markdown]
# ## Step 4: Classical Optimization Loop
#
# The optimization loop:
# 1. Initialize parameters (γ, β) randomly
# 2. Build & run QAOA circuit → bitstring samples
# 3. Compute expected cost: $\langle H_C \rangle = \sum_x P(x|\gamma,\beta) \cdot f(x)$
# 4. Classical optimizer (COBYLA) proposes new parameters
# 5. Repeat until convergence

# %%
# =====================================================================
# STEP 4: Classical Optimization Loop
# =====================================================================

class QAOAOptimizer:
    """QAOA parameter optimization using classical optimizer."""

    def __init__(self, Q, h, J, p=1, shots=1024, maxiter=200, random_state=42):
        self.Q = Q
        self.h = h
        self.J = J
        self.num_qubits = Q.shape[0]
        self.p = p
        self.shots = shots
        self.maxiter = maxiter
        self.random_state = random_state

        # Build circuit
        self.circuit, self.gamma_params, self.beta_params = \
            build_qaoa_circuit(self.num_qubits, p, h, J)

        # Simulator
        self.backend = AerSimulator()

        # Tracking
        self.cost_history = []
        self.iteration_count = 0
        self.optimal_params = None

    def _evaluate_cost(self, params):
        """Evaluate ⟨H_C⟩ for given parameters."""
        gammas = params[:self.p]
        betas = params[self.p:]

        # Bind parameters
        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        bound_circuit = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        counts = job.result().get_counts()

        # Expected cost from samples
        expected_cost = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            x = np.array([int(b) for b in reversed(bitstring)])
            cost = qubo_objective(x, self.Q)
            expected_cost += (count / total) * cost

        self.cost_history.append(expected_cost)
        self.iteration_count += 1

        if self.iteration_count % 25 == 0:
            print(f"    Iter {self.iteration_count}: Cost = {expected_cost:.6f}")

        return expected_cost

    def optimize(self):
        """Run COBYLA optimization."""
        print(f"\n  ⚛ QAOA Optimization (p={self.p}, {self.num_qubits} qubits)")
        print(f"    Shots: {self.shots}, Max iterations: {self.maxiter}")

        self.cost_history = []
        self.iteration_count = 0

        np.random.seed(self.random_state)
        init_params = np.concatenate([
            np.random.uniform(0, 2*np.pi, self.p),  # γ
            np.random.uniform(0, np.pi, self.p),     # β
        ])

        result = minimize(
            self._evaluate_cost, init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.5},
        )

        self.optimal_params = result.x
        print(f"    ✓ Converged after {self.iteration_count} iterations")
        print(f"    ✓ Final cost: {self.cost_history[-1]:.6f}")
        return result

    def get_optimal_counts(self, shots=4096):
        """Run optimized circuit with many shots."""
        gammas = self.optimal_params[:self.p]
        betas = self.optimal_params[self.p:]
        param_dict = {}
        for k in range(self.p):
            param_dict[self.gamma_params[k]] = gammas[k]
            param_dict[self.beta_params[k]] = betas[k]

        bound = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()


print("✅ Step 4: QAOA Optimizer class defined")

# %% [markdown]
# ## Step 5: Extract Optimal Bitstring
#
# After optimization, we:
# 1. Run the circuit with many shots (4096)
# 2. Find the most probable bitstring
# 3. Interpret: bit i = 1 → feature i is **selected**

# %%
# =====================================================================
# STEP 5: Extract Optimal Bitstring (Function)
# =====================================================================

def extract_optimal_solution(optimizer, Q):
    """
    Extract the best feature selection from QAOA results.

    1. Sample from optimized circuit
    2. Evaluate QUBO cost for top bitstrings
    3. Return the bitstring with minimum cost
    """
    counts = optimizer.get_optimal_counts(shots=4096)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  📊 Top measurement outcomes:")
    print(f"    {'Bitstring':<15} {'Count':>6} {'Prob':>8} {'QUBO Cost':>12}")
    print(f"    {'-'*43}")

    best_bitstring = None
    best_cost = float('inf')

    for bitstring, count in sorted_counts[:10]:
        x = np.array([int(b) for b in reversed(bitstring)])
        cost = qubo_objective(x, Q)
        prob = count / 4096
        marker = ""
        if cost < best_cost:
            best_cost = cost
            best_bitstring = bitstring
            marker = " ← best"
        if count > 50:  # Show significant outcomes
            print(f"    {bitstring:<15} {count:>6} {prob:>8.4f} {cost:>12.6f}{marker}")

    selected_mask = np.array([int(b) for b in reversed(best_bitstring)])
    return selected_mask, best_bitstring, best_cost


print("✅ Step 5: Bitstring extraction function defined")

# %% [markdown]
# ---
# # 🧬 PHASE 2 — DATA & ML INTEGRATION
# ---

# %% [markdown]
# ## Step 6: Dataset Handling
#
# We use the **Breast Cancer Wisconsin (Diagnostic)** dataset:
# - 569 samples, 30 features, 2 classes
# - Features: radius, texture, perimeter, area, smoothness, etc.
# - Task: classify tumors as malignant or benign

# %%
# =====================================================================
# STEP 6: Load and preprocess the medical dataset
# =====================================================================

data = load_breast_cancer()
X_raw = data.data
y = data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)

print(f"📋 DATASET: Breast Cancer Wisconsin (Diagnostic)")
print(f"   Samples:  {X_raw.shape[0]}")
print(f"   Features: {X_raw.shape[1]}")
print(f"   Classes:  {target_names}")
print(f"   Balance:  {dict(zip(*np.unique(y, return_counts=True)))}")

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"\n   ✓ Normalized with StandardScaler (μ=0, σ=1)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Split: {X_train.shape[0]} train, {X_test.shape[0]} test")

# Compute full relevance for later visualization
full_relevance = compute_relevance_scores(X, y)
print(f"   ✓ Mutual information scores computed")
print(f"   Top 5 features by MI: {[feature_names[i] for i in np.argsort(full_relevance)[::-1][:5]]}")

# %% [markdown]
# ## Step 7: Apply QAOA Feature Selection
#
# We pre-filter to the top 10 features by mutual information to keep
# qubits manageable on a simulator, then run QAOA on this subset.

# %%
# =====================================================================
# STEP 7: QAOA Feature Selection
# =====================================================================

print("=" * 65)
print("  QAOA FEATURE SELECTION PIPELINE")
print("=" * 65)

# Pre-filter to top candidates
N_CANDIDATES = 10
LAMBDA = 0.5
P_DEPTH = 1

print(f"\n  Configuration:")
print(f"    Candidate features: {N_CANDIDATES}")
print(f"    λ (trade-off):      {LAMBDA}")
print(f"    QAOA depth (p):     {P_DEPTH}")

# Pre-filter by MI
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top_indices = np.argsort(mi_scores)[::-1][:N_CANDIDATES]
top_indices = np.sort(top_indices)

X_train_filtered = X_train[:, top_indices]
X_test_filtered = X_test[:, top_indices]
candidate_names = [feature_names[i] for i in top_indices]
candidate_mi = mi_scores[top_indices]
candidate_mi_norm = candidate_mi / candidate_mi.max() if candidate_mi.max() > 0 else candidate_mi

print(f"\n  Candidates ({N_CANDIDATES} features):")
for i, (idx, name, mi) in enumerate(zip(top_indices, candidate_names, candidate_mi_norm)):
    print(f"    [{i}] Feature {idx}: {name} (MI={mi:.4f})")

# Build QUBO
print(f"\n  Building QUBO matrix...")
relevance = candidate_mi_norm
redundancy = compute_redundancy_matrix(X_train_filtered)
Q = build_qubo_matrix(relevance, redundancy, LAMBDA)
print(f"    QUBO shape: {Q.shape}")
print(f"    Diagonal (neg. relevance): {np.round(np.diag(Q), 3)}")

# Convert to Ising
h, J, offset = qubo_to_ising(Q)
print(f"\n  Ising Hamiltonian:")
print(f"    Linear coefficients: {np.round(h, 4)}")
print(f"    Non-zero ZZ couplings: {np.sum(np.abs(J) > 1e-10)}")

# Run QAOA
qaoa_start = time.time()
optimizer = QAOAOptimizer(Q, h, J, p=P_DEPTH, shots=1024, maxiter=150, random_state=42)
optimizer.optimize()
convergence_history = optimizer.cost_history.copy()

# Extract solution
selected_mask, best_bitstring, best_cost = extract_optimal_solution(optimizer, Q)
qaoa_time = time.time() - qaoa_start

qaoa_selected_in_candidates = np.where(selected_mask == 1)[0]
qaoa_selected_original = top_indices[qaoa_selected_in_candidates]

print(f"\n  ═══════════════════════════════════════")
print(f"  ⚛ QAOA RESULT")
print(f"  ═══════════════════════════════════════")
print(f"  Best bitstring:    {best_bitstring}")
print(f"  QUBO cost:         {best_cost:.6f}")
print(f"  Features selected: {len(qaoa_selected_original)} / {X.shape[1]}")
print(f"  Selected features: {[feature_names[i] for i in qaoa_selected_original]}")
print(f"  Time:              {qaoa_time:.1f}s")

# %% [markdown]
# ## Step 8 & 9: Train ML Models and Evaluate
#
# We compare classifier performance using:
# - **All 30 features** (baseline)
# - **QAOA-selected features**
# - **Classical baseline features** (next section)

# %%
# =====================================================================
# STEPS 8-9: Train ML models and evaluate
# =====================================================================

def evaluate_model(X_tr, X_te, y_tr, y_te, model_name="logistic_regression"):
    """Train and evaluate a single model."""
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=10000, random_state=42)
        name = "Logistic Regression"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        name = "Random Forest"

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    y_pred = model.predict(X_te)

    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy")

    return {
        "model_name": name,
        "n_features": X_tr.shape[1],
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted"),
        "recall": recall_score(y_te, y_pred, average="weighted"),
        "f1_score": f1_score(y_te, y_pred, average="weighted"),
        "cv_mean": cv.mean(),
        "cv_std": cv.std(),
        "train_time": train_time,
        "model": model,
        "y_pred": y_pred,
    }


# Evaluate on all features
print("=" * 70)
print("  ML PERFORMANCE EVALUATION")
print("=" * 70)

all_results = {}

# All features
print("\n  ── All Features (30) ──")
all_results["All Features"] = {}
for m in ["logistic_regression", "random_forest"]:
    res = evaluate_model(X_train, X_test, y_train, y_test, m)
    all_results["All Features"][m] = res
    print(f"    {res['model_name']:25s} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}  CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}")

# QAOA features
print(f"\n  ── QAOA Features ({len(qaoa_selected_original)}) ──")
all_results["QAOA"] = {}
for m in ["logistic_regression", "random_forest"]:
    res = evaluate_model(
        X_train[:, qaoa_selected_original], X_test[:, qaoa_selected_original],
        y_train, y_test, m
    )
    all_results["QAOA"][m] = res
    print(f"    {res['model_name']:25s} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}  CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}")

# %% [markdown]
# ---
# # ⚙️ PHASE 3 — CLASSICAL BASELINE COMPARISON
# ---

# %% [markdown]
# ## Step 10: Classical Feature Selection Methods
#
# We implement three classical baselines:
# 1. **PCA** — Dimensionality reduction via variance
# 2. **LASSO** — L1 regularized feature selection
# 3. **Genetic Algorithm** — Evolutionary feature subset optimization

# %%
# =====================================================================
# STEP 10: Classical Baselines — PCA
# =====================================================================

print("\n  ── PCA ──")
pca_start = time.time()
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pca_time = time.time() - pca_start

n_pca = X_train_pca.shape[1]
print(f"    Components:    {n_pca}")
print(f"    Explained var: {sum(pca.explained_variance_ratio_):.4f}")
print(f"    Time:          {pca_time:.3f}s")

all_results["PCA"] = {}
for m in ["logistic_regression", "random_forest"]:
    res = evaluate_model(X_train_pca, X_test_pca, y_train, y_test, m)
    all_results["PCA"][m] = res
    print(f"    {res['model_name']:25s} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}")

# %%
# =====================================================================
# STEP 10: Classical Baselines — LASSO
# =====================================================================

print("\n  ── LASSO (L1 Regularization) ──")
lasso_start = time.time()

best_C, best_score = 1.0, 0
for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    lr = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=10000, random_state=42)
    sc = cross_val_score(lr, X_train, y_train, cv=3).mean()
    if sc > best_score:
        best_score, best_C = sc, C

lr_lasso = LogisticRegression(penalty="l1", C=best_C, solver="saga", max_iter=10000, random_state=42)
lr_lasso.fit(X_train, y_train)
coef = np.abs(lr_lasso.coef_).mean(axis=0)
lasso_indices = np.where(coef > 1e-6)[0]
if len(lasso_indices) < 2:
    lasso_indices = np.argsort(coef)[::-1][:5]

lasso_time = time.time() - lasso_start
print(f"    Optimal C:     {best_C}")
print(f"    Features:      {len(lasso_indices)}")
print(f"    Time:          {lasso_time:.3f}s")

all_results["LASSO"] = {}
for m in ["logistic_regression", "random_forest"]:
    res = evaluate_model(
        X_train[:, lasso_indices], X_test[:, lasso_indices],
        y_train, y_test, m
    )
    all_results["LASSO"][m] = res
    print(f"    {res['model_name']:25s} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}")

# %%
# =====================================================================
# STEP 10: Classical Baselines — Genetic Algorithm
# =====================================================================

print("\n  ── Genetic Algorithm ──")
ga_start = time.time()

np.random.seed(42)
POP_SIZE = 20
N_GENS = 30
MUT_RATE = 0.1
n_features_total = X_train.shape[1]

def ga_fitness(mask):
    if mask.sum() == 0:
        return 0.0
    idx = np.where(mask)[0]
    lr = LogisticRegression(max_iter=5000, random_state=42)
    sc = cross_val_score(lr, X_train[:, idx], y_train, cv=3).mean()
    penalty = 0.01 * max(0, mask.sum() - 10)
    return sc - penalty

# Initialize population
population = np.random.randint(0, 2, (POP_SIZE, n_features_total))
for i in range(POP_SIZE):
    if population[i].sum() == 0:
        population[i][np.random.randint(n_features_total)] = 1

ga_fitness_history = []

for gen in range(N_GENS):
    fitnesses = np.array([ga_fitness(ind) for ind in population])
    best_idx = np.argmax(fitnesses)
    ga_fitness_history.append(fitnesses[best_idx])

    if (gen + 1) % 10 == 0:
        print(f"    Gen {gen+1:3d}: fitness={fitnesses[best_idx]:.4f}, feats={population[best_idx].sum()}")

    new_pop = [population[best_idx].copy()]
    while len(new_pop) < POP_SIZE:
        # Tournament selection
        t = np.random.choice(POP_SIZE, 3, replace=False)
        p1 = population[t[np.argmax(fitnesses[t])]].copy()
        t = np.random.choice(POP_SIZE, 3, replace=False)
        p2 = population[t[np.argmax(fitnesses[t])]].copy()
        # Crossover
        pt = np.random.randint(1, n_features_total)
        child = np.concatenate([p1[:pt], p2[pt:]])
        # Mutation
        flip = np.random.random(n_features_total) < MUT_RATE
        child[flip] = 1 - child[flip]
        if child.sum() == 0:
            child[np.random.randint(n_features_total)] = 1
        new_pop.append(child)
    population = np.array(new_pop)

fitnesses = np.array([ga_fitness(ind) for ind in population])
best_ga = population[np.argmax(fitnesses)]
ga_indices = np.where(best_ga)[0]
ga_time = time.time() - ga_start

print(f"    Features:      {len(ga_indices)}")
print(f"    Time:          {ga_time:.1f}s")

all_results["Genetic Algorithm"] = {}
for m in ["logistic_regression", "random_forest"]:
    res = evaluate_model(
        X_train[:, ga_indices], X_test[:, ga_indices],
        y_train, y_test, m
    )
    all_results["Genetic Algorithm"][m] = res
    print(f"    {res['model_name']:25s} Acc={res['accuracy']:.4f}  F1={res['f1_score']:.4f}")

# %% [markdown]
# ## Step 11: Benchmarking — Comprehensive Comparison

# %%
# =====================================================================
# STEP 11: Comprehensive Benchmarking Table
# =====================================================================

print("\n" + "=" * 90)
print("  COMPREHENSIVE BENCHMARKING RESULTS")
print("=" * 90)
header = f"  {'Method':<22s} {'Model':<22s} {'#Feat':>5s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'CV':>12s}"
print(header)
print("  " + "-" * 88)

for method, models in all_results.items():
    for model_name, res in models.items():
        cv_str = f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}"
        print(f"  {method:<22s} {res['model_name']:<22s} "
              f"{res['n_features']:>5d} {res['accuracy']:>7.4f} "
              f"{res['precision']:>7.4f} {res['recall']:>7.4f} "
              f"{res['f1_score']:>7.4f} {cv_str:>12s}")

print("=" * 90)

# Feature counts & times
feature_counts = {
    "All Features": 30,
    "QAOA": len(qaoa_selected_original),
    "PCA": n_pca,
    "LASSO": len(lasso_indices),
    "GA": len(ga_indices),
}
time_dict = {
    "QAOA": qaoa_time,
    "PCA": pca_time,
    "LASSO": lasso_time,
    "GA": ga_time,
}

print(f"\n  Feature Counts: {feature_counts}")
print(f"  Computation Times: { {k: f'{v:.2f}s' for k, v in time_dict.items()} }")

# %% [markdown]
# ---
# # 📊 PHASE 4 — VISUALIZATION & ANALYSIS
# ---

# %% [markdown]
# ## Step 12: Visualizations

# %%
# =====================================================================
# STEP 12: Visualization — Feature Importance
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 8))

sort_idx = np.argsort(full_relevance)[::-1]
sorted_scores = full_relevance[sort_idx]
sorted_names = [feature_names[i] for i in sort_idx]

colors = ["#2ecc71" if sort_idx[i] in qaoa_selected_original else "#3498db"
          for i in range(len(sort_idx))]

bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=colors,
               edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sorted_scores)))
ax.set_yticklabels(sorted_names, fontsize=8)
ax.set_xlabel("Mutual Information Score (Normalized)")
ax.set_title("Feature Importance — Mutual Information with Target\n"
             "(Green = QAOA Selected, Blue = Not Selected)", fontsize=13)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2ecc71", label="QAOA Selected"),
    Patch(facecolor="#3498db", label="Not Selected"),
], loc="lower right")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/feature_importance.png")

# %%
# =====================================================================
# STEP 12: Visualization — QAOA Convergence
# =====================================================================

fig, ax = plt.subplots(figsize=(10, 5))

iterations = range(1, len(convergence_history) + 1)
ax.plot(iterations, convergence_history, "b-", alpha=0.4, linewidth=0.8,
        label="Cost per iteration")

running_min = np.minimum.accumulate(convergence_history)
ax.plot(iterations, running_min, "r-", linewidth=2, label="Running minimum")

if len(convergence_history) > 10:
    w = min(10, len(convergence_history) // 5)
    ma = np.convolve(convergence_history, np.ones(w)/w, mode="valid")
    ax.plot(range(w, len(convergence_history)+1), ma, "g--", linewidth=1.5,
            alpha=0.7, label=f"Moving avg (w={w})")

ax.set_xlabel("Optimization Iteration")
ax.set_ylabel("Expected Cost ⟨H_C⟩")
ax.set_title("QAOA Optimization Convergence")
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

min_idx = np.argmin(convergence_history)
ax.annotate(f"Min: {convergence_history[min_idx]:.4f}\n(iter {min_idx+1})",
            xy=(min_idx+1, convergence_history[min_idx]),
            xytext=(min_idx+1 + len(convergence_history)*0.1,
                    convergence_history[min_idx]),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red")

plt.tight_layout()
plt.savefig("outputs/qaoa_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/qaoa_convergence.png")

# %%
# =====================================================================
# STEP 12: Visualization — Accuracy Comparison
# =====================================================================

methods = list(all_results.keys())
model_types = ["logistic_regression", "random_forest"]
metrics = ["accuracy", "precision", "recall", "f1_score"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[ax_idx]
    x = np.arange(len(methods))
    width = 0.35

    for i, mt in enumerate(model_types):
        vals = [all_results[m][mt][metric] for m in methods]
        name = all_results[methods[0]][mt]["model_name"]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("ML Performance Across Feature Selection Methods",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/accuracy_comparison.png")

# %%
# =====================================================================
# STEP 12: Visualization — Feature Count & Time Comparison
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Feature counts
methods_fc = list(feature_counts.keys())
counts_fc = list(feature_counts.values())
colors_fc = sns.color_palette("viridis", len(methods_fc))
bars = ax1.bar(methods_fc, counts_fc, color=colors_fc, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts_fc):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             str(v), ha="center", va="bottom", fontweight="bold")
ax1.set_ylabel("Number of Features")
ax1.set_title("Features Selected by Each Method")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.3)

# Computation time
methods_t = list(time_dict.keys())
times_t = list(time_dict.values())
colors_t = sns.color_palette("magma", len(methods_t))
bars = ax2.bar(methods_t, times_t, color=colors_t, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, times_t):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f"{v:.2f}s", ha="center", va="bottom", fontweight="bold")
ax2.set_ylabel("Time (seconds)")
ax2.set_title("Computational Cost Comparison")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/benchmark_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/benchmark_comparison.png")

# %%
# =====================================================================
# STEP 12: Visualization — Correlation Heatmap
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 12))
corr = np.corrcoef(X[:, :15].T)
short_names = [fn[:15] for fn in feature_names[:15]]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, xticklabels=short_names,
            yticklabels=short_names, ax=ax, vmin=-1, vmax=1,
            annot_kws={"fontsize": 7})
ax.set_title("Feature Correlation Matrix (First 15 Features)\n"
             "High |corr| = Redundancy → QAOA Penalizes", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: outputs/correlation_heatmap.png")

# %% [markdown]
# ---
# # 📄 PHASE 6 — RESEARCH OUTPUT
# ---

# %% [markdown]
# ## Step 15: Key Research Sections
#
# The following sections form the basis of an IEEE-style research paper.

# %%
# =====================================================================
# STEP 15: Generate Research Report Summary
# =====================================================================

print("=" * 70)
print("  RESEARCH PAPER — KEY SECTIONS")
print("=" * 70)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABSTRACT
--------
We present a hybrid quantum-classical approach to medical feature selection
using the Quantum Approximate Optimization Algorithm (QAOA). Feature selection
is formulated as a QUBO problem that maximizes mutual information (relevance)
while minimizing pairwise correlation (redundancy). Applied to the Breast Cancer
Wisconsin dataset (30 features, 569 samples), QAOA selects {n_qaoa} features
achieving competitive classification accuracy compared to all features and
classical baselines (PCA, LASSO, Genetic Algorithm).

PROBLEM STATEMENT
-----------------
Medical datasets often contain high-dimensional, redundant feature spaces.
Feature selection is NP-hard (2^n subsets for n features). QAOA encodes this
combinatorial search naturally in quantum superposition, offering potential
quantum advantage for large-scale problems.

METHODOLOGY
-----------
1. Formulate feature selection as QUBO: min x^T Q x
   - Q_ii = -MI(Xi, Y)  (relevance)
   - Q_ij = λ|corr(Xi,Xj)|/2  (redundancy)
2. Map QUBO to Ising Hamiltonian via x_i = (1-Z_i)/2
3. Build QAOA ansatz with p={p_depth} layers
4. Optimize (γ, β) with COBYLA minimizing ⟨H_C⟩
5. Extract optimal bitstring → feature subset

RESULTS
-------""".format(n_qaoa=len(qaoa_selected_original), p_depth=P_DEPTH))

for method, models in all_results.items():
    for mn, res in models.items():
        print(f"  {method:22s} | {res['model_name']:22s} | "
              f"Acc={res['accuracy']:.4f} | F1={res['f1_score']:.4f} | "
              f"Features={res['n_features']}")

print(f"""
FUTURE WORK
-----------
1. Execute on real quantum hardware (IBM Quantum, IonQ)
2. Explore deeper QAOA circuits (p > 1) for better approximation
3. Apply to larger genomics datasets with hierarchical decomposition
4. Investigate warm-starting QAOA with classical solutions
5. Multi-objective QAOA balancing accuracy, features, and interpretability
6. Clinical validation of selected biomarkers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# %% [markdown]
# ## 🎯 Summary
#
# This notebook implemented a **complete end-to-end pipeline** for
# quantum-classical hybrid feature selection:
#
# | Phase | Steps | Description |
# |-------|-------|-------------|
# | Phase 1 | 1-5 | QAOA theory, QUBO formulation, circuit, optimization |
# | Phase 2 | 6-9 | Dataset handling, QAOA selection, ML training |
# | Phase 3 | 10-11 | PCA, LASSO, GA baselines and benchmarking |
# | Phase 4 | 12 | Feature importance, accuracy, convergence plots |
# | Phase 5 | 13-14 | See `docs/architecture.md` for diagrams |
# | Phase 6 | 15 | IEEE-style research paper sections |
#
# All results are saved in the `outputs/` directory.

# %%
print("\n🎉 PIPELINE COMPLETE!")
print(f"   QAOA selected {len(qaoa_selected_original)} features: "
      f"{[feature_names[i] for i in qaoa_selected_original]}")
print(f"   All plots saved to outputs/")
