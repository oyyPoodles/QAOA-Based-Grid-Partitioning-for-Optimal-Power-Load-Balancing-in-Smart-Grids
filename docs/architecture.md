# System Architecture — QAOA Medical Feature Selection

## Steps 13–14: Architecture & End-to-End Workflow

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Layer"
        A["Medical Dataset<br/>(Breast Cancer Wisconsin)"] --> B["Preprocessing<br/>• Cleaning<br/>• Normalization<br/>• Train/Test Split"]
        B --> C["Feature Analysis<br/>• Mutual Information<br/>• Correlation Matrix"]
    end

    subgraph "Quantum Layer (QAOA)"
        C --> D["QUBO Formulation<br/>Q_ii = -r_i<br/>Q_ij = λc_ij/2"]
        D --> E["Ising Mapping<br/>H_C = Σh_iZ_i + ΣJ_ijZ_iZ_j"]
        E --> F["QAOA Circuit<br/>|γ,β⟩ = U_M U_C |+⟩^n"]
        F --> G["Quantum Simulation<br/>(Qiskit Aer)"]
        G --> H["Measurement<br/>Sample bitstrings"]
        H --> I{"Classical Optimizer<br/>(COBYLA/SPSA)"}
        I -->|"Update γ, β"| F
        I -->|"Converged"| J["Optimal Bitstring<br/>→ Feature Subset"]
    end

    subgraph "Classical ML Layer"
        J --> K["Selected Features"]
        B --> L["All Features"]
        K --> M["ML Models<br/>• Logistic Regression<br/>• Random Forest"]
        L --> M
        N["Classical Baselines<br/>• PCA<br/>• LASSO<br/>• Genetic Algorithm"] --> M
        B --> N
    end

    subgraph "Evaluation Layer"
        M --> O["Metrics<br/>• Accuracy<br/>• Precision<br/>• Recall<br/>• F1-Score"]
        O --> P["Comparative Analysis<br/>• QAOA vs All Features<br/>• QAOA vs Baselines"]
        P --> Q["Visualization<br/>• Feature Importance<br/>• Accuracy Charts<br/>• Convergence Plot"]
        P --> R["Research Report<br/>(IEEE-style)"]
    end

    style A fill:#3498db,stroke:#2980b9,color:#fff
    style F fill:#9b59b6,stroke:#8e44ad,color:#fff
    style G fill:#9b59b6,stroke:#8e44ad,color:#fff
    style I fill:#e67e22,stroke:#d35400,color:#fff
    style J fill:#2ecc71,stroke:#27ae60,color:#fff
    style M fill:#e74c3c,stroke:#c0392b,color:#fff
    style Q fill:#1abc9c,stroke:#16a085,color:#fff
    style R fill:#1abc9c,stroke:#16a085,color:#fff
```

### End-to-End Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant Data as Data Layer
    participant Quantum as Quantum Layer
    participant ML as ML Layer
    participant Eval as Evaluation Layer

    User->>Data: Input medical dataset
    Data->>Data: Clean & normalize
    Data->>Data: Compute MI & correlations
    Data->>Quantum: Feature statistics (r_i, c_ij)
    
    Note over Quantum: QUBO Formulation
    Quantum->>Quantum: Build Q matrix
    Quantum->>Quantum: Convert to Ising H_C
    
    loop QAOA Optimization
        Quantum->>Quantum: Build circuit |γ,β⟩
        Quantum->>Quantum: Simulate & measure
        Quantum->>Quantum: Compute ⟨H_C⟩
        Quantum->>Quantum: Update γ, β (COBYLA)
    end
    
    Quantum->>ML: Selected feature indices
    Data->>ML: Train/test data
    
    ML->>ML: Train LogReg & RF (QAOA features)
    ML->>ML: Train LogReg & RF (all features)
    ML->>ML: Run PCA, LASSO, GA baselines
    ML->>ML: Train LogReg & RF (baseline features)
    
    ML->>Eval: All predictions & metrics
    Eval->>Eval: Compute Acc, Prec, Rec, F1
    Eval->>Eval: Generate comparison tables
    Eval->>Eval: Create visualizations
    Eval->>Eval: Generate research report
    
    Eval->>User: Results, plots, paper
```

### Component Dependency Map

```
main.py
├── src/data/dataset.py          # Data loading & preprocessing
├── src/qaoa/
│   ├── hamiltonian.py           # QUBO & Ising formulation
│   ├── circuit.py               # QAOA circuit construction
│   ├── optimizer.py             # Classical optimization loop
│   └── feature_selector.py      # High-level QAOA selector
├── src/ml/
│   ├── classifier.py            # ML training & evaluation
│   └── baselines.py             # PCA, LASSO, GA
├── src/visualization/plots.py   # All visualizations
└── src/research/report.py       # Research paper generation
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Quantum Circuit | Qiskit | Build parameterized QAOA ansatz |
| Simulation | Qiskit Aer | Simulate quantum circuits |
| Optimization | SciPy (COBYLA) | Classical parameter optimization |
| Data Processing | NumPy, Pandas, Scikit-learn | Dataset handling |
| Machine Learning | Scikit-learn | Classification models |
| Visualization | Matplotlib, Seaborn | Publication-quality plots |
| Research Output | Markdown / LaTeX | IEEE-style paper generation |
