"""
================================================================================
Classical Feature Selection Baselines
================================================================================

STEP 10: Implement Classical Methods
--------------------------------------
For fair comparison, we implement three classical feature selection methods:

1. PCA (Principal Component Analysis):
   - Unsupervised dimensionality reduction
   - Projects data onto orthogonal directions of maximum variance
   - Selects top-k components by explained variance ratio
   - Note: PCA creates NEW features (linear combinations), not a subset

2. LASSO (L1 Regularization):
   - Supervised feature selection via regularized logistic regression
   - L1 penalty drives coefficients of irrelevant features to zero
   - Features with non-zero coefficients are selected
   - α parameter controls sparsity

3. Genetic Algorithm (GA):
   - Evolutionary optimization for feature subset selection
   - Mimics natural selection: population of solutions evolves
   - Uses fitness (model accuracy), crossover, and mutation
   - Stochastic, may find different solutions each run

STEP 11: Benchmarking
-----------------------
Compare all methods on:
  • Classification performance (accuracy, F1)
  • Number of features selected
  • Computational cost (wall-clock time)
================================================================================
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def pca_feature_selection(X_train, X_test, n_components=None,
                          variance_threshold=0.95):
    """
    Dimensionality reduction using PCA.

    Selects the minimum number of components that explain at least
    `variance_threshold` fraction of total variance.

    Note: PCA transforms features into principal components (linear
    combinations of original features). Unlike other methods, it does
    NOT select a subset of original features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test : np.ndarray, shape (n_test, n_features)
    n_components : int, optional
        Fixed number of components. If None, uses variance_threshold.
    variance_threshold : float
        Minimum cumulative explained variance.

    Returns
    -------
    result : dict
        Contains transformed data, PCA object, and metadata.
    """
    start_time = time.time()

    if n_components is None:
        # Find minimum components for threshold
        pca_full = PCA().fit(X_train)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, X_train.shape[1])

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    elapsed = time.time() - start_time

    explained_var = sum(pca.explained_variance_ratio_)

    print(f"\n  PCA Feature Selection:")
    print(f"    Components:       {n_components}")
    print(f"    Explained var:    {explained_var:.4f} ({explained_var*100:.1f}%)")
    print(f"    Time:             {elapsed:.3f}s")

    return {
        "method": "PCA",
        "X_train": X_train_pca,
        "X_test": X_test_pca,
        "n_features": n_components,
        "pca": pca,
        "explained_variance": explained_var,
        "time": elapsed,
        "indices": None,  # PCA creates new features, not a subset
    }


def lasso_feature_selection(X_train, X_test, y_train, alpha=None):
    """
    Feature selection using LASSO (L1 regularized logistic regression).

    L1 regularization adds a penalty term λ·||w||₁ to the loss function,
    which drives irrelevant feature weights to exactly zero.

    If α is not specified, uses cross-validation (LassoCV) to find
    the optimal regularization strength.

    Parameters
    ----------
    X_train : np.ndarray
    X_test : np.ndarray
    y_train : np.ndarray
    alpha : float, optional
        L1 regularization strength. If None, found by CV.

    Returns
    -------
    result : dict
    """
    start_time = time.time()

    # Use L1 regularized logistic regression
    if alpha is None:
        # Search over a range of C values (C = 1/α)
        best_score = 0
        best_C = 1.0
        for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            model = LogisticRegression(
                penalty="l1", C=C, solver="saga", max_iter=10000,
                random_state=42
            )
            scores = cross_val_score(model, X_train, y_train, cv=3,
                                     scoring="accuracy")
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_C = C
        C = best_C
    else:
        C = 1.0 / alpha

    model = LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=10000, random_state=42
    )
    model.fit(X_train, y_train)

    # Features with non-zero coefficients
    coef = np.abs(model.coef_).mean(axis=0)  # Average across classes if multi-class
    selected_indices = np.where(coef > 1e-6)[0]

    # If too few selected, take top features by coefficient magnitude
    if len(selected_indices) < 2:
        selected_indices = np.argsort(coef)[::-1][:5]

    elapsed = time.time() - start_time

    print(f"\n  LASSO Feature Selection:")
    print(f"    Regularization C: {C}")
    print(f"    Features selected: {len(selected_indices)}")
    print(f"    Selected indices:  {selected_indices.tolist()}")
    print(f"    Time:              {elapsed:.3f}s")

    return {
        "method": "LASSO",
        "X_train": X_train[:, selected_indices],
        "X_test": X_test[:, selected_indices],
        "n_features": len(selected_indices),
        "indices": selected_indices,
        "coefficients": coef,
        "C": C,
        "time": elapsed,
    }


def genetic_algorithm_selection(X_train, X_test, y_train, y_test=None,
                                n_features_target=None, pop_size=30,
                                n_generations=50, mutation_rate=0.1,
                                random_state=42):
    """
    Feature selection using a Genetic Algorithm (GA).

    The GA evolves a population of binary feature masks:
      1. Initialize random population of binary vectors
      2. Evaluate fitness (5-fold CV accuracy)
      3. Select parents via tournament selection
      4. Create offspring via crossover
      5. Apply random bit-flip mutation
      6. Repeat for n_generations

    Parameters
    ----------
    X_train : np.ndarray
    X_test : np.ndarray
    y_train : np.ndarray
    y_test : np.ndarray, optional
    n_features_target : int, optional
        Soft target for number of features (via penalty).
    pop_size : int
        Population size.
    n_generations : int
        Number of generations.
    mutation_rate : float
        Probability of flipping each bit during mutation.
    random_state : int

    Returns
    -------
    result : dict
    """
    start_time = time.time()
    np.random.seed(random_state)
    n_total_features = X_train.shape[1]

    if n_features_target is None:
        n_features_target = n_total_features // 3

    print(f"\n  Genetic Algorithm Feature Selection:")
    print(f"    Population:   {pop_size}")
    print(f"    Generations:  {n_generations}")
    print(f"    Mutation rate: {mutation_rate}")

    def fitness(mask):
        """Evaluate fitness of a feature mask."""
        if mask.sum() == 0:
            return 0.0

        indices = np.where(mask)[0]
        X_sub = X_train[:, indices]

        model = LogisticRegression(max_iter=5000, random_state=42, solver="lbfgs")
        scores = cross_val_score(model, X_sub, y_train, cv=3, scoring="accuracy")
        acc = scores.mean()

        # Penalty for too many features
        n_selected = mask.sum()
        penalty = 0.01 * max(0, n_selected - n_features_target)

        return acc - penalty

    def tournament_select(population, fitnesses, k=3):
        """Select parent via tournament selection."""
        indices = np.random.choice(len(population), size=k, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def crossover(parent1, parent2):
        """Single-point crossover."""
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def mutate(individual, rate):
        """Bit-flip mutation."""
        mask = np.random.random(len(individual)) < rate
        individual[mask] = 1 - individual[mask]
        return individual

    # Initialize population
    population = np.random.randint(0, 2, size=(pop_size, n_total_features))
    # Ensure at least one feature is selected in each individual
    for i in range(pop_size):
        if population[i].sum() == 0:
            population[i][np.random.randint(n_total_features)] = 1

    # Evolution loop
    best_fitness_history = []

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = np.array([fitness(ind) for ind in population])

        best_idx = np.argmax(fitnesses)
        best_fitness_history.append(fitnesses[best_idx])

        if (gen + 1) % 10 == 0:
            print(f"    Gen {gen+1:3d}: Best fitness = {fitnesses[best_idx]:.4f}, "
                  f"Features = {population[best_idx].sum()}")

        # Create next generation
        new_population = [population[best_idx].copy()]  # Elitism

        while len(new_population) < pop_size:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            if child.sum() == 0:
                child[np.random.randint(n_total_features)] = 1
            new_population.append(child)

        population = np.array(new_population)

    # Final evaluation
    fitnesses = np.array([fitness(ind) for ind in population])
    best_idx = np.argmax(fitnesses)
    best_mask = population[best_idx]
    selected_indices = np.where(best_mask)[0]

    elapsed = time.time() - start_time

    print(f"    Final: {len(selected_indices)} features selected")
    print(f"    Selected indices: {selected_indices.tolist()}")
    print(f"    Time: {elapsed:.1f}s")

    return {
        "method": "Genetic Algorithm",
        "X_train": X_train[:, selected_indices],
        "X_test": X_test[:, selected_indices],
        "n_features": len(selected_indices),
        "indices": selected_indices,
        "best_mask": best_mask,
        "fitness_history": best_fitness_history,
        "time": elapsed,
    }


def run_all_baselines(X_train, X_test, y_train, y_test):
    """
    Run all classical feature selection baselines.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : np.ndarray

    Returns
    -------
    baselines : dict
        {method_name: result_dict}
    """
    print(f"\n{'='*60}")
    print(f"  CLASSICAL BASELINE FEATURE SELECTION")
    print(f"{'='*60}")

    baselines = {}

    # PCA
    pca_result = pca_feature_selection(X_train, X_test)
    baselines["PCA"] = pca_result

    # LASSO
    lasso_result = lasso_feature_selection(X_train, X_test, y_train)
    baselines["LASSO"] = lasso_result

    # Genetic Algorithm
    ga_result = genetic_algorithm_selection(
        X_train, X_test, y_train, y_test,
        n_generations=30, pop_size=20
    )
    baselines["Genetic Algorithm"] = ga_result

    print(f"\n{'='*60}")
    print(f"  BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'#Features':>10s} {'Time (s)':>10s}")
    print(f"  {'-'*42}")
    for name, result in baselines.items():
        print(f"  {name:<20s} {result['n_features']:>10d} "
              f"{result['time']:>10.3f}")
    print(f"{'='*60}\n")

    return baselines
