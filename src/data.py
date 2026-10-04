"""Load the cleaned table and split it spatially.

Each of the ~2,000 points per city is measured on many dates. The split keeps
ALL dates of a point on the same side and keeps neighbouring points (~1 km
blocks) together, so the test score measures prediction for unseen places.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config as C


def load(path=C.CLEAN_DATA, max_rows: int | None = C.MAX_ROWS, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    if max_rows and n > max_rows:
        df = df.sample(max_rows, random_state=seed)
    print(f"Loaded {n:,} rows from {path}; using {len(df):,}")
    return prepare(df.reset_index(drop=True))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in C.CATEGORICAL_CANDIDATES:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing").astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def split_features(df: pd.DataFrame, cols) -> tuple[list[str], list[str]]:
    present = [c for c in cols if c in df.columns and df[c].nunique(dropna=True) > 1]
    categorical = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in present if c not in categorical], categorical


def spatial_blocks(df: pd.DataFrame, size: float = C.BLOCK_DEG) -> pd.Series:
    bx = np.floor(df[C.COORDS[0]] / size).astype(int).astype(str)
    by = np.floor(df[C.COORDS[1]] / size).astype(int).astype(str)
    return df.get("city", "x").astype(str) + "_" + bx + "_" + by


def spatial_split(df: pd.DataFrame, test_size=C.TEST_SIZE, seed=C.RANDOM_STATE):
    groups = spatial_blocks(df)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(df, groups=groups))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True), groups
