# System Architecture — QAOA Smart Grid Partitioning

## Steps 13–14: Architecture & End-to-End Workflow

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Layer"
        A["PJM Energy Dataset<br/>(11 Regional CSVs)"] --> B["Data Processing<br/>• Load hourly data<br/>• Align time series<br/>• Daily aggregation"]
        B --> C["Graph Construction<br/>• Nodes = regions<br/>• Edges = correlations<br/>• Weights = 1-|corr|"]
    end

    subgraph "QAOA Optimization Layer"
        C --> D["QUBO Formulation<br/>• MAX-CUT (edge cost)<br/>• Load balance penalty<br/>• Combined Q matrix"]
        D --> E["Ising Mapping<br/>x_i = (1-Z_i)/2<br/>H_C = ΣhZ + ΣJZZ"]
        E --> F["QAOA Circuit<br/>|γ,β⟩ = U_M U_C |+⟩^n"]
        F --> G["Quantum Simulation<br/>(Qiskit Aer)"]
        G --> H["Measurement<br/>Sample bitstrings"]
        H --> I{"COBYLA Optimizer"}
        I -->|"Update γ, β"| F
        I -->|"Converged"| J["Optimal Bitstring<br/>→ Partition Assignment"]
    end

    subgraph "Output Layer"
        J --> K["Grid Partitioning<br/>• Partition A nodes<br/>• Partition B nodes"]
        K --> L["Evaluation<br/>• Load balance<br/>• Edge cut cost<br/>• Transmission loss"]
        L --> M["Visualization<br/>• Grid plots<br/>• Convergence<br/>• Load charts"]
        L --> N["Research Report<br/>(IEEE-style)"]
    end

    style A fill:#3498db,stroke:#2980b9,color:#fff
    style D fill:#e67e22,stroke:#d35400,color:#fff
    style F fill:#9b59b6,stroke:#8e44ad,color:#fff
    style G fill:#9b59b6,stroke:#8e44ad,color:#fff
    style I fill:#e67e22,stroke:#d35400,color:#fff
    style J fill:#2ecc71,stroke:#27ae60,color:#fff
    style K fill:#2ecc71,stroke:#27ae60,color:#fff
    style M fill:#1abc9c,stroke:#16a085,color:#fff
```

### End-to-End Pipeline Flow

```mermaid
sequenceDiagram
    participant Data as Data Layer
    participant Graph as Graph Construction
    participant QUBO as QUBO Engine
    participant QAOA as QAOA Circuit
    participant Eval as Evaluation

    Data->>Graph: Load 11 regional CSVs
    Graph->>Graph: Align time series
    Graph->>Graph: Compute correlations
    Graph->>QUBO: Weighted graph G(V,E)

    QUBO->>QUBO: MAX-CUT + Balance penalty
    QUBO->>QUBO: Build Q matrix
    QUBO->>QAOA: Ising coefficients (h, J)

    loop COBYLA Optimization
        QAOA->>QAOA: Build circuit |γ,β⟩
        QAOA->>QAOA: Simulate & measure
        QAOA->>QAOA: Compute ⟨H_C⟩
        QAOA->>QAOA: Update (γ, β)
    end

    QAOA->>Eval: Optimal bitstring
    Eval->>Eval: Map to partition
    Eval->>Eval: Compute load balance
    Eval->>Eval: Compare vs random
    Eval->>Eval: Generate plots & report
```

### Pipeline Summary

```
Dataset (PJM CSVs)
    ↓
Graph Construction (NetworkX)
    ↓
QUBO Formulation (MAX-CUT + Balance)
    ↓
Ising Mapping (x_i → Z_i)
    ↓
QAOA Circuit (Qiskit)
    ↓
Classical Optimization (COBYLA)
    ↓
Optimal Partition
    ↓
Evaluation & Visualization
```
