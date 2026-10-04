"""SUHI rule (Furuya et al. 2023, Eq. 7–8), applied per scene (city + date):

    SUHI  if  LST > mean + 0.5 * SD   (mean, SD of that city on that date)
"""
import numpy as np
import pandas as pd

from src.config import GROUP, SUHI_K


def threshold(values, k=SUHI_K) -> float:
    v = np.asarray(values, float)
    return float(np.nanmean(v) + k * np.nanstd(v, ddof=1))


def classify(df: pd.DataFrame, col: str, k=SUHI_K) -> pd.Series:
    keys = [g for g in GROUP if g in df.columns]
    if keys:
        thr = df.groupby(keys)[col].transform(lambda s: s.mean() + k * s.std())
    else:
        thr = threshold(df[col], k)
    return df[col] > thr


def evaluate(df: pd.DataFrame, ref: str, pred: str) -> tuple[dict, pd.DataFrame]:
    """Overall agreement + paper-style table (Table 4): mean LST of SUHI vs non-SUHI per season."""
    d = df.copy()
    d["suhi_ref"], d["suhi_pred"] = classify(d, ref), classify(d, pred)
    tp = (d.suhi_ref & d.suhi_pred).sum()
    overall = {
        "agreement_pct": 100 * (d.suhi_ref == d.suhi_pred).mean(),
        "suhi_precision_pct": 100 * tp / max(d.suhi_pred.sum(), 1),
        "suhi_recall_pct": 100 * tp / max(d.suhi_ref.sum(), 1),
        "ref_suhi_pct": 100 * d.suhi_ref.mean(), "pred_suhi_pct": 100 * d.suhi_pred.mean(),
    }
    rows = []
    for season, g in d.groupby("season"):
        rows.append({
            "season": season, "scenes": g[GROUP].drop_duplicates().shape[0],
            "ref_mean_suhi": g.loc[g.suhi_ref, ref].mean(), "ref_mean_non_suhi": g.loc[~g.suhi_ref, ref].mean(),
            "pred_mean_suhi": g.loc[g.suhi_pred, pred].mean(), "pred_mean_non_suhi": g.loc[~g.suhi_pred, pred].mean(),
            "agreement_pct": 100 * (g.suhi_ref == g.suhi_pred).mean(),
        })
    t = pd.DataFrame(rows)
    t["ref_difference"] = t.ref_mean_suhi - t.ref_mean_non_suhi
    t["pred_difference"] = t.pred_mean_suhi - t.pred_mean_non_suhi
    return {k: round(float(v), 2) for k, v in overall.items()}, t.round(2)
