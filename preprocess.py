from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **read_csv_kwargs)


def fill_and_normalize_numeric(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    numeric_cols = df.select_dtypes(include="number").columns if columns is None else list(columns)
    processed = df.copy()
    for col in numeric_cols:
        processed[col] = processed[col].fillna(processed[col].mean())

    if scaler is None:
        scaler = StandardScaler()
        processed[numeric_cols] = scaler.fit_transform(processed[numeric_cols])
    else:
        processed[numeric_cols] = scaler.transform(processed[numeric_cols])

    return processed, scaler


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
