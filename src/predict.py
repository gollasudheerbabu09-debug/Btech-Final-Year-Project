"""Load saved models and predict LST.

    python -m src.predict points.csv --model RF --out predictions.csv
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src import config as C
from src.suhi import classify


def models_available() -> bool:
    return (C.MODELS_DIR / "metadata.json").exists()


def load_models():
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    models = {k: joblib.load(C.MODELS_DIR / f"{k}.joblib") for k in meta["model_names"]
              if (C.MODELS_DIR / f"{k}.joblib").exists()}
    return models, meta


def input_columns(meta):
    return meta["numeric_features"] + meta["categorical_features"]


def missing_columns(df, meta):
    return [c for c in input_columns(meta) if c not in df.columns]


def predict(df: pd.DataFrame, models: dict, meta: dict, keys=None) -> pd.DataFrame:
    X = df.copy()
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("string").fillna("missing").astype(str).str.replace(r"\.0$", "", regex=True)
    for c in meta["numeric_features"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").where(lambda s: s != C.NODATA)
    out = df.copy()
    for k in keys or models:
        out[f"pred_{k}"] = models[k].predict(X[input_columns(meta)]).round(2)
    return out


def add_suhi(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.assign(suhi=classify(df, col).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    models, meta = load_models()
    df = pd.read_csv(a.csv)
    if miss := missing_columns(df, meta):
        raise SystemExit(f"Missing columns: {miss}")
    key = a.model or meta["best_model"]
    res = add_suhi(predict(df, models, meta, [key]), f"pred_{key}")
    res.to_csv(a.out, index=False)
    print(f"Saved {len(res):,} predictions ({key}) -> {a.out}; {res.suhi.mean():.1%} heat island")


if __name__ == "__main__":
    main()
