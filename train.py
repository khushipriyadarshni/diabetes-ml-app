"""Training script for diabetes risk prediction model."""

import json
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_data, preprocess, make_pipeline, evaluate

DATA_PATH = Path("data/diabetes.csv")          # fix: no double 'data'
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def print_eda_stats(df: pd.DataFrame) -> None:
    """
    Print simple EDA statistics for the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")  # Exclude target
    print(f"Number of samples: {df.shape[0]}")
    
    print("\n" + "-" * 60)
    print("Target Distribution:")
    print("-" * 60)
    if "Outcome" in df.columns:
        outcome_counts = df["Outcome"].value_counts()
        print(f"Class 0 (No Diabetes): {outcome_counts.get(0, 0)} ({outcome_counts.get(0, 0) / len(df) * 100:.2f}%)")
        print(f"Class 1 (Diabetes): {outcome_counts.get(1, 0)} ({outcome_counts.get(1, 0) / len(df) * 100:.2f}%)")
    
    print("\n" + "-" * 60)
    print("Feature Statistics:")
    print("-" * 60)
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
    
    for col in feature_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
            print(f"  Zeros: {(df[col] == 0).sum()}")
            print(f"  Missing: {df[col].isna().sum()}")
    
    print("\n" + "=" * 60)


def save_model_metadata(
    metrics: Dict[str, float],
    feature_names: list,
    feature_ranges: Dict[str, Dict[str, float]],
) -> None:
    """
    Save model metadata including metrics and feature ranges to JSON.
    
    Args:
        metrics: Dictionary of evaluation metrics
        feature_names: List of feature names
        feature_ranges: Dictionary with min/max for each feature
    """
    metadata = {
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel metadata saved to: {METRICS_PATH}")


def main() -> None:
    """Main training function."""
    # Create directories if they don't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data(str(DATA_PATH))
    
    # EDA
    print_eda_stats(df)
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y, feature_names = preprocess(df)
    
    # Calculate feature ranges for validation
    feature_ranges = {}
    for col in feature_names:
        feature_ranges[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
        }
    
    # Split data with stratification
    print("\nSplitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Build pipeline
    print("\nBuilding pipeline...")
    pipeline = make_pipeline()
    
    # Train model
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = evaluate(pipeline, X_val, y_val)
    
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    for metric_name, metric_value in val_metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    # Save model
    print(f"\nSaving model to: {MODEL_PATH}")
    import joblib
    joblib.dump(pipeline, MODEL_PATH)
    
    # Save metadata
    save_model_metadata(
        val_metrics,
        feature_names,
        feature_ranges,
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

