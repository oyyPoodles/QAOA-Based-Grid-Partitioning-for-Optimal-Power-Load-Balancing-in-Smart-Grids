"""
================================================================================
ML Classifier Training and Evaluation
================================================================================

STEP 8: Train ML Model
------------------------
We train classifiers using:
  1. All original features (baseline)
  2. QAOA-selected features (our method)
  3. Features from classical methods (comparison)

Models:
  • Logistic Regression: Linear classifier, interpretable, works well
    with feature selection since fewer features = less overfitting
  • Random Forest: Ensemble of decision trees, captures non-linear
    relationships, provides feature importance

STEP 9: Evaluation
--------------------
Metrics:
  • Accuracy:  (TP + TN) / (TP + TN + FP + FN)
  • Precision: TP / (TP + FP) — "of predicted positives, how many correct?"
  • Recall:    TP / (TP + FN) — "of actual positives, how many found?"
  • F1-Score:  2·P·R / (P + R) — harmonic mean of precision and recall

We compare all feature selection methods across these metrics.
================================================================================
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="logistic_regression"):
    """
    Train a classifier and evaluate on the test set.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test : np.ndarray, shape (n_test, n_features)
    y_train : np.ndarray, shape (n_train,)
    y_test : np.ndarray, shape (n_test,)
    model_name : str
        'logistic_regression' or 'random_forest'

    Returns
    -------
    results : dict
        Dictionary containing metrics, model, predictions, and timing.
    """
    # Select model
    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=10000,
            random_state=42,
            solver="lbfgs",
        )
        display_name = "Logistic Regression"
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        display_name = "Random Forest"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_test)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    # Compute metrics
    results = {
        "model_name": display_name,
        "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "train_time": train_time,
        "model": model,
        "predictions": y_pred,
    }

    return results


def compare_feature_sets(X_train, X_test, y_train, y_test,
                         feature_selections, feature_names=None):
    """
    Compare ML performance across different feature selection methods.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_all_features)
    X_test : np.ndarray, shape (n_test, n_all_features)
    y_train : np.ndarray, shape (n_train,)
    y_test : np.ndarray, shape (n_test,)
    feature_selections : dict
        {method_name: selected_feature_indices} or
        {method_name: None} for all features.
    feature_names : list of str, optional

    Returns
    -------
    all_results : dict
        {method_name: {model_name: results_dict}}
    """
    all_results = {}
    models = ["logistic_regression", "random_forest"]

    print(f"\n{'='*70}")
    print(f"  ML PERFORMANCE COMPARISON")
    print(f"{'='*70}")

    for method_name, indices in feature_selections.items():
        all_results[method_name] = {}

        if indices is None:
            # Use all features
            X_tr = X_train
            X_te = X_test
            n_feat = X_train.shape[1]
        else:
            X_tr = X_train[:, indices]
            X_te = X_test[:, indices]
            n_feat = len(indices)

        print(f"\n  ┌─ {method_name} ({n_feat} features)")

        for model_name in models:
            results = train_and_evaluate(X_tr, X_te, y_train, y_test, model_name)
            all_results[method_name][model_name] = results

            print(f"  │  {results['model_name']:25s}  "
                  f"Acc={results['accuracy']:.4f}  "
                  f"F1={results['f1_score']:.4f}  "
                  f"CV={results['cv_accuracy_mean']:.4f}±{results['cv_accuracy_std']:.4f}")

        print(f"  └─")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    header = f"  {'Method':<25s} {'Model':<22s} {'#Feat':>5s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for method_name, method_results in all_results.items():
        for model_name, results in method_results.items():
            print(f"  {method_name:<25s} {results['model_name']:<22s} "
                  f"{results['n_features']:>5d} "
                  f"{results['accuracy']:>7.4f} "
                  f"{results['precision']:>7.4f} "
                  f"{results['recall']:>7.4f} "
                  f"{results['f1_score']:>7.4f}")

    print(f"{'='*70}\n")

    return all_results
