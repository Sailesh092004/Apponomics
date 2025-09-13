"""Train basic machine learning models for the Apponomics dataset."""

from __future__ import annotations

import argparse
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_dataset(
    path: str,
    target: str | None,
    feature_cols: Iterable[str] | None,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Load features and optional target column from CSV."""
    df = pd.read_csv(path)

    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()
        if target and target in feature_cols:
            feature_cols.remove(target)
    X = df[list(feature_cols)]
    y = df[target] if target else None
    return X, y


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a machine learning model")
    parser.add_argument("--data", required=True, help="Path to CSV training data")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression", "clustering"],
        help="Type of model to train",
    )
    parser.add_argument("--target", help="Target column for supervised tasks")
    parser.add_argument("--model", required=True, help="Where to save the trained model")
    parser.add_argument("--features", nargs="+", help="Optional list of feature columns to use")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for clustering")
    args = parser.parse_args(argv)

    X, y = load_dataset(args.data, args.target, args.features)

    if args.task == "classification":
        if y is None:
            raise ValueError("Target column required for classification")
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
    elif args.task == "regression":
        if y is None:
            raise ValueError("Target column required for regression")
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
    else:  # clustering
        model = KMeans(n_clusters=args.clusters, random_state=42)
        model.fit(X)

    joblib.dump(model, args.model)
    print(f"Trained {args.task} model saved to {args.model}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
