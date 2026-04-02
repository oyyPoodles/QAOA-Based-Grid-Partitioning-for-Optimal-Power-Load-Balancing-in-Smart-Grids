# %% [markdown]
# # ⚡ QAOA-Based Grid Partitioning for Optimal Power Load Balancing
# # in Smart Grids
#
# **Complete End-to-End Pipeline — Google Colab Ready**
#
# ---
# ## Pipeline: Dataset → Graph → QUBO → QAOA → Partition → Evaluation

# %% [markdown]
# ## 📦 Setup

# %%
# Uncomment for Google Colab:
# !pip install qiskit qiskit-aer networkx matplotlib seaborn pandas numpy scipy

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, time
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

from scipy.optimize import minimize

print("✅ All imports successful!")

# %% [markdown]
# ---
# # 🧠 PHASE 1 — PROBLEM FORMULATION
# ---
#
# ## Step 1: Smart Grid Optimization Problem
#
# **Grid partitioning** divides a power network into balanced sub-grids:
# - **Nodes** → power regions (PJM interconnection zones)
# - **Edges** → transmission interconnections
# - **Edge weights** → similarity between consumption patterns
#
# **Objective:**
# - Minimize transmission losses across partition boundaries
# - Balance total load evenly across partitions
#
# This is an NP-hard graph bisection problem (2^n possible partitions).
# QAOA provides a quantum-native approach.

# %% [markdown]
# ## Step 2: Load Dataset & Construct Graph
#
# We use the **PJM Interconnection Hourly Energy Consumption** dataset.
# Each CSV represents a regional power zone with hourly MW data.

# %%
# =====================================================================
# STEP 2: Load Dataset → Construct Grid Graph
# =====================================================================

DATASET_DIR = "dataset"

region_files = {
    "AEP": "AEP_hourly.csv", "COMED": "COMED_hourly.csv",
    "DAYTON": "DAYTON_hourly.csv", "DEOK": "DEOK_hourly.csv",
    "DOM": "DOM_hourly.csv", "DUQ": "DUQ_hourly.csv",
    "EKPC": "EKPC_hourly.csv", "FE": "FE_hourly.csv",
    "NI": "NI_hourly.csv", "PJME": "PJME_hourly.csv",
    "PJMW": "PJMW_hourly.csv",
}

# Load all regions
regions = {}
for name, filename in region_files.items():
    filepath = os.path.join(DATASET_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df.columns = ["Datetime", "MW"]
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        regions[name] = df

print(f"📋 Loaded {len(regions)} PJM regions:")
for name, df in regions.items():
    print(f"  {name:8s}  mean={df['MW'].mean():>10,.0f} MW  rows={len(df)}")

# Align to common date range (daily aggregation)
# Iteratively remove regions with non-overlapping dates
valid_regions = dict(regions)
while len(valid_regions) > 2:
    starts = {n: df.index.min() for n, df in valid_regions.items()}
    ends = {n: df.index.max() for n, df in valid_regions.items()}
    start = max(starts.values())
    end = min(ends.values())
    if start < end:
        break
    worst = min(ends, key=ends.get)
    print(f"  Excluding '{worst}' (no date overlap)")
    del valid_regions[worst]

start = max(df.index.min() for df in valid_regions.values())
end = min(df.index.max() for df in valid_regions.values())
print(f"\nCommon range: {start.date()} to {end.date()} ({len(valid_regions)} regions)")

aligned = pd.DataFrame()
for name, df in valid_regions.items():
    aligned[name] = df.loc[start:end].resample("D").mean()["MW"]
aligned = aligned.dropna()

mean_loads = aligned.mean()
print(f"Aligned: {aligned.shape[0]} days x {aligned.shape[1]} regions")

# Compute correlation matrix
corr_matrix = aligned.corr()

# Build graph
# Select 8 representative nodes for QAOA (from valid regions only)
nodes_sorted = sorted(valid_regions.keys(), key=lambda n: mean_loads[n])
n_target = min(8, len(nodes_sorted))
indices = np.linspace(0, len(nodes_sorted) - 1, n_target, dtype=int)
selected_nodes = [nodes_sorted[i] for i in indices]
print(f"\nSelected nodes: {selected_nodes}")

G = nx.Graph()
for node in selected_nodes:
    G.add_node(node, load=float(mean_loads[node]))

for i, u in enumerate(selected_nodes):
    for j, v in enumerate(selected_nodes):
        if i < j:
            corr = abs(corr_matrix.loc[u, v])
            if corr > 0.3:
                G.add_edge(u, v, weight=round(1.0 - corr, 4))

print(f"\n🔗 Grid Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"\nNode loads:")
for node in G.nodes():
    print(f"  {node:8s}  {G.nodes[node]['load']:>10,.0f} MW")

# %%
# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr_sub = corr_matrix.loc[selected_nodes, selected_nodes]
sns.heatmap(corr_sub, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Inter-Region Load Correlation\n(High corr → similar patterns → keep together)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Visualize original grid
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42, k=2)
loads = [G.nodes[n]["load"] for n in G.nodes()]
max_load = max(loads)
node_sizes = [800 + 2000 * (l / max_load) for l in loads]
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_w = max(edge_weights) if edge_weights else 1
edge_widths = [3.0 * (1 - w / max_w) + 0.5 for w in edge_weights]

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="#95a5a6", ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#3498db",
                       alpha=0.85, edgecolors="#2c3e50", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
ax.set_title("Original Power Grid Network\n(Node size ∝ load)", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/original_grid.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 3: Define Cost Function
#
# Two components:
#
# **1. Edge Cut (MAX-CUT):**
# $$C_{cut} = \sum_{(i,j) \in E} w_{ij} (x_i + x_j - 2x_i x_j)$$
# Maximize this → separate dissimilar regions.
#
# **2. Load Balancing:**
# $$C_{balance} = \alpha \cdot \left(\sum_i l_i x_i - \frac{L}{2}\right)^2$$
# Minimize this → equal load per partition.
#
# **Combined:** minimize $-C_{cut} + \alpha \cdot C_{balance}$

# %% [markdown]
# ## Step 4: QUBO Formulation
#
# QUBO: minimize $\mathbf{x}^T Q \mathbf{x}$
#
# For MAX-CUT minimization:
# - $Q_{ii} = -\sum_{j:(i,j)\in E} w_{ij}$
# - $Q_{ij} = w_{ij}$
#
# For load balance:
# - $Q_{ii} += \alpha(l_i^2 - L \cdot l_i)$
# - $Q_{ij} += 2\alpha \cdot l_i \cdot l_j$

# %%
# =====================================================================
# STEPS 3-4: Build QUBO Matrix
# =====================================================================

nodes = sorted(G.nodes())
n = len(nodes)
node_idx = {node: i for i, node in enumerate(nodes)}
ALPHA = 2.0  # Balance penalty weight

# MAX-CUT component
Q_cut = np.zeros((n, n))
for u, v, data in G.edges(data=True):
    w = data["weight"]
    i, j = node_idx[u], node_idx[v]
    Q_cut[i, i] -= w
    Q_cut[j, j] -= w
    Q_cut[i, j] += w
    Q_cut[j, i] += w

# Load balance component
loads_arr = np.array([G.nodes[node]["load"] for node in nodes])
L = loads_arr.sum()
loads_norm = loads_arr / L  # Normalize

Q_bal = np.zeros((n, n))
for i in range(n):
    Q_bal[i, i] = loads_norm[i] ** 2 - loads_norm[i]
for i in range(n):
    for j in range(i + 1, n):
        Q_bal[i, j] = 2.0 * loads_norm[i] * loads_norm[j]
        Q_bal[j, i] = Q_bal[i, j]

# Combined (normalize scales)
cut_scale = max(np.abs(Q_cut).max(), 1e-10)
bal_scale = max(np.abs(Q_bal).max(), 1e-10)
Q = Q_cut / cut_scale + ALPHA * Q_bal / bal_scale

print(f"✅ QUBO Matrix ({n}×{n}):")
print(f"   Cut scale:     {cut_scale:.6f}")
print(f"   Balance scale: {bal_scale:.6f}")
print(f"   Q range:       [{Q.min():.4f}, {Q.max():.4f}]")

def qubo_objective(x, Q):
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)

# %% [markdown]
# ## Step 5: Map QUBO → Ising Hamiltonian
#
# Substitute $x_i = (1 - Z_i)/2$:
# $$H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{offset}$$

# %%
# =====================================================================
# STEP 5: QUBO → Ising Hamiltonian
# =====================================================================

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

print(f"✅ Ising Hamiltonian:")
print(f"   h = {np.round(h, 4)}")
print(f"   Non-zero J: {np.sum(np.abs(J) > 1e-10)}")
print(f"   Offset: {offset:.4f}")

# %% [markdown]
# ---
# # ⚛️ PHASE 2 — QAOA IMPLEMENTATION
# ---
#
# ## Step 6: Build QAOA Circuit
#
# ```
# |0⟩ ─ H ─┤ Cost Layer (γ) ├─┤ Mixer Layer (β) ├─ Measure
# |0⟩ ─ H ─┤               ├─┤                ├─ Measure
# ...
# ```
#
# **Cost:** Rz(2γh) for Z terms, CNOT-Rz-CNOT for ZZ terms
# **Mixer:** Rx(2β) on each qubit

# %%
# =====================================================================
# STEP 6: Build QAOA Circuit
# =====================================================================

P_DEPTH = 1  # QAOA depth

gamma_params = [Parameter(f"gamma_{k}") for k in range(P_DEPTH)]
beta_params = [Parameter(f"beta_{k}") for k in range(P_DEPTH)]

qc = QuantumCircuit(n, n)

# Initial superposition
for i in range(n):
    qc.h(i)
qc.barrier()

for k in range(P_DEPTH):
    # Cost layer
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                qc.cx(i, j)
                qc.rz(2.0 * gamma_params[k] * J[i, j], j)
                qc.cx(i, j)
    for i in range(n):
        if abs(h[i]) > 1e-10:
            qc.rz(2.0 * gamma_params[k] * h[i], i)
    qc.barrier()

    # Mixer layer
    for i in range(n):
        qc.rx(2.0 * beta_params[k], i)
    qc.barrier()

qc.measure(range(n), range(n))

print(f"✅ QAOA Circuit built:")
print(f"   Qubits: {n}")
print(f"   Depth:  {qc.depth()}")
print(f"   Gates:  {qc.size()}")
print(f"   Ops:    {dict(qc.count_ops())}")

# Draw circuit (simplified)
print("\n" + str(qc.draw(output="text", fold=100)))

# %% [markdown]
# ## Step 7: Classical Optimization Loop
#
# COBYLA minimizes $\langle H_C \rangle = \sum_x P(x|\gamma,\beta) \cdot f(x)$

# %%
# =====================================================================
# STEP 7: Classical Optimization Loop
# =====================================================================

backend = AerSimulator()
SHOTS = 1024
MAXITER = 200

cost_history = []
iter_count = [0]

def evaluate_cost(params):
    gammas = params[:P_DEPTH]
    betas = params[P_DEPTH:]
    param_dict = {}
    for k in range(P_DEPTH):
        param_dict[gamma_params[k]] = gammas[k]
        param_dict[beta_params[k]] = betas[k]

    bound = qc.assign_parameters(param_dict)
    transpiled = transpile(bound, backend)
    job = backend.run(transpiled, shots=SHOTS)
    counts = job.result().get_counts()

    exp_cost = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in reversed(bitstring)])
        exp_cost += (count / total) * qubo_objective(x, Q)

    cost_history.append(exp_cost)
    iter_count[0] += 1
    if iter_count[0] % 25 == 0:
        print(f"    Iter {iter_count[0]}: ⟨Cost⟩ = {exp_cost:.6f}")
    return exp_cost

print(f"⚛ QAOA Optimization (p={P_DEPTH}, {n} qubits, {SHOTS} shots)")
print(f"  Optimizing with COBYLA (max {MAXITER} iters)...\n")

np.random.seed(42)
init_params = np.concatenate([
    np.random.uniform(0, 2 * np.pi, P_DEPTH),
    np.random.uniform(0, np.pi, P_DEPTH),
])

qaoa_start = time.time()
result = minimize(evaluate_cost, init_params, method="COBYLA",
                  options={"maxiter": MAXITER, "rhobeg": 0.5})
qaoa_time = time.time() - qaoa_start

print(f"\n  ✓ Converged: {iter_count[0]} iterations, {qaoa_time:.1f}s")
print(f"  ✓ Final cost: {cost_history[-1]:.6f}")

# %% [markdown]
# ## Step 8: Extract Optimal Partition

# %%
# =====================================================================
# STEP 8: Extract Optimal Partition
# =====================================================================

opt_params = result.x
gammas = opt_params[:P_DEPTH]
betas = opt_params[P_DEPTH:]
param_dict = {}
for k in range(P_DEPTH):
    param_dict[gamma_params[k]] = gammas[k]
    param_dict[beta_params[k]] = betas[k]

bound = qc.assign_parameters(param_dict)
transpiled = transpile(bound, backend)
job = backend.run(transpiled, shots=8192)
final_counts = job.result().get_counts()

sorted_counts = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)

print("📊 Top 10 measurement outcomes:")
print(f"  {'Bitstring':<12} {'Count':>6} {'Prob':>8} {'QUBO Cost':>12}")
print(f"  {'-'*40}")

best_bs, best_cost = None, float("inf")
for bs, count in sorted_counts[:10]:
    x = np.array([int(b) for b in reversed(bs)])
    cost = qubo_objective(x, Q)
    prob = count / 8192
    mk = " ← best" if cost < best_cost else ""
    if cost < best_cost:
        best_cost, best_bs = cost, bs
    print(f"  {bs:<12} {count:>6} {prob:>8.4f} {cost:>12.6f}{mk}")

partition_bits = np.array([int(b) for b in reversed(best_bs)])
part_A = [nodes[i] for i in range(n) if partition_bits[i] == 0]
part_B = [nodes[i] for i in range(n) if partition_bits[i] == 1]
load_A = sum(G.nodes[nd]["load"] for nd in part_A)
load_B = sum(G.nodes[nd]["load"] for nd in part_B)
total_load = load_A + load_B
imbalance = abs(load_A - load_B) / total_load

print(f"\n{'='*55}")
print(f"  ⚡ OPTIMAL GRID PARTITION")
print(f"{'='*55}")
print(f"  Partition A: {part_A}")
print(f"  Partition B: {part_B}")
print(f"  Load A:      {load_A:,.0f} MW")
print(f"  Load B:      {load_B:,.0f} MW")
print(f"  Imbalance:   {imbalance*100:.2f}%")

# %% [markdown]
# ---
# # ⚙️ PHASE 3 — PRACTICAL INTERPRETATION
# ---
#
# ## Steps 9-10: Map Results & Evaluate Performance

# %%
# =====================================================================
# STEPS 9-10: Evaluation — QAOA vs Random Partition
# =====================================================================

# Random baseline
np.random.seed(0)
rand_bits = np.random.randint(0, 2, n)
if rand_bits.sum() == 0: rand_bits[0] = 1
elif rand_bits.sum() == n: rand_bits[0] = 0

rand_A = [nodes[i] for i in range(n) if rand_bits[i] == 0]
rand_B = [nodes[i] for i in range(n) if rand_bits[i] == 1]
rand_load_A = sum(G.nodes[nd]["load"] for nd in rand_A)
rand_load_B = sum(G.nodes[nd]["load"] for nd in rand_B)
rand_imbalance = abs(rand_load_A - rand_load_B) / (rand_load_A + rand_load_B)

rand_var = np.var([rand_load_A, rand_load_B])
qaoa_var = np.var([load_A, load_B])
improvement = (1 - qaoa_var / rand_var) * 100 if rand_var > 0 else 0

print(f"{'='*55}")
print(f"  PERFORMANCE COMPARISON")
print(f"{'='*55}")
print(f"  {'Metric':<18} {'Random':>12} {'QAOA':>12}")
print(f"  {'-'*44}")
print(f"  {'Load A':<18} {rand_load_A:>10,.0f}MW {load_A:>10,.0f}MW")
print(f"  {'Load B':<18} {rand_load_B:>10,.0f}MW {load_B:>10,.0f}MW")
print(f"  {'Imbalance':<18} {rand_imbalance*100:>10.2f}%  {imbalance*100:>10.2f}%")
print(f"  {'Load Variance':<18} {rand_var:>12,.0f} {qaoa_var:>12,.0f}")
print(f"  Variance improvement: {improvement:.1f}%")

# Transmission loss proxy
rand_cut_w = sum(G[u][v]["weight"] for u, v in G.edges()
                 if rand_bits[node_idx[u]] != rand_bits[node_idx[v]])
qaoa_cut_w = sum(G[u][v]["weight"] for u, v in G.edges()
                 if partition_bits[node_idx[u]] != partition_bits[node_idx[v]])
rand_loss = rand_cut_w * abs(rand_load_A - rand_load_B)
qaoa_loss = qaoa_cut_w * abs(load_A - load_B)
print(f"\n  Transmission loss proxy:")
print(f"    Random: {rand_loss:,.0f}")
print(f"    QAOA:   {qaoa_loss:,.0f}")
if rand_loss > 0:
    print(f"    Reduction: {(1-qaoa_loss/rand_loss)*100:.1f}%")

# %% [markdown]
# ---
# # 📊 PHASE 4 — VISUALIZATION
# ---

# %%
# =====================================================================
# STEP 11: Partitioned Grid Visualization
# =====================================================================

fig, ax = plt.subplots(figsize=(10, 8))
node_sizes = [800 + 2000 * (G.nodes[nd]["load"] / max_load) for nd in G.nodes()]
colors = ["#e74c3c" if partition_bits[node_idx[nd]] == 0 else "#2ecc71"
          for nd in G.nodes()]

for u, v, d in G.edges(data=True):
    is_cut = partition_bits[node_idx[u]] != partition_bits[node_idx[v]]
    style, color, alpha, lw = ("--", "#e74c3c", 0.8, 2.5) if is_cut else ("-", "#2ecc71", 0.4, 1.5)
    ax.annotate("", xy=pos[v], xytext=pos[u],
                arrowprops=dict(arrowstyle="-", color=color, alpha=alpha, lw=lw, linestyle=style))

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors,
                       alpha=0.85, edgecolors="#2c3e50", linewidths=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
ax.legend(handles=[
    mpatches.Patch(facecolor="#e74c3c", label=f"Partition A: {load_A:,.0f} MW"),
    mpatches.Patch(facecolor="#2ecc71", label=f"Partition B: {load_B:,.0f} MW"),
    plt.Line2D([0], [0], color="#e74c3c", ls="--", lw=2, label="Cut Edge"),
], loc="upper left")
ax.set_title("QAOA-Optimized Grid Partition", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/partitioned_grid.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# =====================================================================
# STEP 12: Load Distribution & Convergence
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Load distribution
ax1 = axes[0]
bar_loads = [G.nodes[nd]["load"] for nd in nodes]
bar_colors = ["#e74c3c" if partition_bits[i] == 0 else "#2ecc71" for i in range(n)]
bars = ax1.bar(range(n), bar_loads, color=bar_colors, edgecolor="white", linewidth=1.5)
ax1.set_xticks(range(n))
ax1.set_xticklabels(nodes, rotation=30, ha="right")
ax1.set_ylabel("Mean Load (MW)")
ax1.set_title("Load Distribution by Partition")
for bar, ld in zip(bars, bar_loads):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
             f"{ld:,.0f}", ha="center", fontsize=7, fontweight="bold")
ax1.legend(handles=[
    mpatches.Patch(facecolor="#e74c3c", label=f"A: {load_A:,.0f} MW"),
    mpatches.Patch(facecolor="#2ecc71", label=f"B: {load_B:,.0f} MW"),
])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Convergence
ax2 = axes[1]
ax2.plot(range(1, len(cost_history)+1), cost_history, "b-", alpha=0.4, lw=0.8, label="Cost")
rm = np.minimum.accumulate(cost_history)
ax2.plot(range(1, len(cost_history)+1), rm, "r-", lw=2, label="Running min")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("⟨H_C⟩")
ax2.set_title("QAOA Convergence")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/load_and_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Load balance comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(2)
w = 0.35
ax1.bar(x - w/2, [rand_load_A, rand_load_B], w, label="Random", color="#e74c3c", alpha=0.8)
ax1.bar(x + w/2, [load_A, load_B], w, label="QAOA", color="#2ecc71", alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(["Partition A", "Partition B"])
ax1.set_ylabel("Load (MW)")
ax1.set_title("Partition Load: Random vs QAOA")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

bars = ax2.bar(["Random", "QAOA"], [rand_var, qaoa_var], color=["#e74c3c", "#2ecc71"])
ax2.set_ylabel("Load Variance")
ax2.set_title(f"Load Variance (Improvement: {improvement:.1f}%)")
ax2.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, [rand_var, qaoa_var]):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+100,
             f"{v:,.0f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/load_balance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# # 📄 PHASE 6 — RESEARCH OUTPUT
# ---

# %%
print("=" * 60)
print("  RESEARCH PAPER SECTIONS")
print("=" * 60)
print(f"""
ABSTRACT
--------
We present a quantum computing approach to smart grid partitioning using
QAOA. The problem is formulated as a QUBO combining MAX-CUT with a load
balancing penalty. Applied to PJM Interconnection data ({n} regions),
QAOA achieves {imbalance*100:.2f}% load imbalance, a {improvement:.1f}%
variance reduction vs random partitioning.

PROBLEM STATEMENT
-----------------
Grid partitioning divides power networks into balanced sub-grids.
With n nodes, 2^n partitions exist → NP-hard. QAOA provides quantum
speedup potential.

METHODOLOGY
-----------
1. PJM hourly data → weighted graph (corr-based edges)
2. QUBO: MAX-CUT + α·(balance penalty)
3. Ising: x_i = (1-Z_i)/2
4. QAOA circuit (p={P_DEPTH}) optimized via COBYLA
5. Bitstring → partition assignment

RESULTS
-------
  Partition A: {part_A} ({load_A:,.0f} MW)
  Partition B: {part_B} ({load_B:,.0f} MW)
  Imbalance: {imbalance*100:.2f}%
  Variance improvement: {improvement:.1f}%

FUTURE SCOPE
------------
1. Multi-way partitioning (k > 2 sub-grids)
2. Real quantum hardware (IBM Quantum)
3. Dynamic repartitioning for real-time load changes
4. Integration with renewable energy sources
5. Larger graph via recursive QAOA / coarsening
""")

print("🎉 PIPELINE COMPLETE!")
