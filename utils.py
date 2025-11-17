"""Utility functions for diabetes risk prediction ML project."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def load_data(path: str) -> pd.DataFrame:
    """
    Load diabetes dataset from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame containing the diabetes data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file at {path} is empty")


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Preprocess the diabetes dataset.
    
    Handles zero-as-missing for Glucose, BloodPressure, SkinThickness, Insulin, BMI
    by replacing zeros with NaN and imputing with median.
    
    Args:
        df: Raw DataFrame with diabetes data
        
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature DataFrame
        - y: Target Series
        - feature_names: List of feature column names
    """
    df = df.copy()
    
    # Identify target column
    if "Outcome" not in df.columns:
        raise ValueError("'Outcome' column not found in dataset")
    
    # Separate features and target
    feature_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    
    # Check all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    X = df[feature_cols].copy()
    y = df["Outcome"].copy()
    
    # Handle zero-as-missing for specific columns
    zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    for col in zero_as_missing_cols:
        if col in X.columns:
            # Replace zeros with NaN
            X[col] = X[col].replace(0, np.nan)
            # Impute with median
            median_value = X[col].median()
            X[col] = X[col].fillna(median_value)
    
    return X, y, feature_cols


def make_pipeline() -> Pipeline:
    """
    Create a scikit-learn pipeline with StandardScaler and RandomForestClassifier.
    
    Returns:
        Pipeline object with preprocessing and classifier
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
        ]
    )
    return pipeline


def evaluate(clf: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate classifier performance on given data.
    
    Args:
        clf: Trained classifier (can be a Pipeline)
        X: Feature DataFrame
        y: True target labels
        
    Returns:
        Dictionary with evaluation metrics: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_pred_proba)),
    }
    
    return metrics

