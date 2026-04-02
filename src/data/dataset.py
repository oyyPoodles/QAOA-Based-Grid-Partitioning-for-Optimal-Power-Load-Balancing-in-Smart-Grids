"""
================================================================================
Medical Dataset Handling
================================================================================

STEP 6: Dataset Handling
-------------------------
We use the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn:

  • 569 samples
  • 30 numerical features (computed from digitized images of breast masses)
  • 2 classes: malignant (0) and benign (1)
  • Features include: mean radius, texture, perimeter, area, smoothness,
    compactness, concavity, concave points, symmetry, fractal dimension
    (each with mean, standard error, and worst-case variants)

Pipeline:
  1. Load dataset
  2. Handle missing values (none expected, but defensive)
  3. Normalize using StandardScaler (zero mean, unit variance)
  4. Split into train (80%) and test (20%) sets
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_medical_dataset(as_dataframe=False):
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    This is a well-known medical classification dataset suitable for
    demonstrating feature selection because:
      - 30 features → many are correlated (redundant)
      - Clear clinical relevance
      - Well-studied benchmark

    Parameters
    ----------
    as_dataframe : bool
        If True, return pandas DataFrame instead of numpy arrays.

    Returns
    -------
    X : np.ndarray, shape (569, 30) or pd.DataFrame
        Feature matrix.
    y : np.ndarray, shape (569,)
        Target labels (0 = malignant, 1 = benign).
    feature_names : list of str
        Feature names.
    target_names : list of str
        Target class names.
    """
    data = load_breast_cancer()

    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    print(f"\n{'='*60}")
    print(f"DATASET: Breast Cancer Wisconsin (Diagnostic)")
    print(f"{'='*60}")
    print(f"  Samples:       {X.shape[0]}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Classes:       {target_names}")
    print(f"  Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Feature names: {feature_names[:5]}... (+ {len(feature_names)-5} more)")

    if as_dataframe:
        X = pd.DataFrame(X, columns=feature_names)

    return X, y, feature_names, target_names


def preprocess_data(X, feature_names=None):
    """
    Preprocess the feature matrix.

    Steps:
      1. Check for and handle missing values
      2. Normalize using StandardScaler (z-score normalization):
         z = (x - μ) / σ
         This ensures all features are on the same scale, which is
         important for both QUBO formulation and ML models.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix.
    feature_names : list of str, optional

    Returns
    -------
    X_normalized : np.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        Fitted scaler (for transforming test data).
    """
    print(f"\n▸ Preprocessing data...")

    # Check for missing values
    if isinstance(X, pd.DataFrame):
        n_missing = X.isnull().sum().sum()
        X_array = X.values
    else:
        n_missing = np.sum(np.isnan(X))
        X_array = X.copy()

    print(f"  Missing values: {n_missing}")

    if n_missing > 0:
        # Impute with column mean
        col_means = np.nanmean(X_array, axis=0)
        for i in range(X_array.shape[1]):
            mask = np.isnan(X_array[:, i])
            X_array[i, mask] = col_means[i]
        print(f"  → Imputed with column means")

    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_array)

    print(f"  Normalization: StandardScaler (μ=0, σ=1)")
    print(f"  Shape: {X_normalized.shape}")

    return X_normalized, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    test_size : float
        Fraction of data for testing.
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n▸ Train-test split ({int((1-test_size)*100)}/{int(test_size*100)}):")
    print(f"  Training:  {X_train.shape[0]} samples")
    print(f"  Testing:   {X_test.shape[0]} samples")
    print(f"  Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test


def get_feature_statistics(X, feature_names):
    """
    Compute descriptive statistics for the dataset.

    Parameters
    ----------
    X : np.ndarray
    feature_names : list of str

    Returns
    -------
    stats : pd.DataFrame
        Summary statistics.
    """
    df = pd.DataFrame(X, columns=feature_names)
    stats = df.describe().T
    stats["range"] = stats["max"] - stats["min"]
    return stats
