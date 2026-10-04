"""Future-year test: train on 2019–2023, predict every place in 2024.

Complements the spatial test (unseen places) and leave-one-city-out (unseen city)
by checking whether the models still work on dates they have never seen.

    python -m src.temporal_test
"""
import json

import matplotlib
import pandas as pd

from src import config as C
from src.data import prepare, split_features
from src.models import MODEL_NAMES, build_pipeline, get_estimators
from src.suhi import evaluate
from src.train import cap, metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(rows=C.LOCO_ROWS, test_year="2024"):
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    df = prepare(pd.read_csv(C.CLEAN_DATA, low_memory=False))
    train = cap(df[df.date < test_year], None, rows)
    test = df[df.date >= test_year]
    test = cap(test, None, rows // 2)
    num, cat = split_features(df, C.FEATURE_SETS[meta["feature_set"]]["cols"])
    print(f"Train {len(train):,} rows (2019–{int(test_year) - 1}), test {len(test):,} rows ({test_year}), "
          f"{meta['feature_set']}")
    rows_out, best = [], (None, 1e9, None)
    for key, est in get_estimators().items():
        t = cap(train, key)
        pipe = build_pipeline(est, num, cat).fit(t[num + cat], t[C.TARGET])
        p = pipe.predict(test[num + cat])
        m = metrics(test[C.TARGET], p)
        rows_out.append({"algorithm": key, **m})
        if m["RMSE"] < best[1]:
            best = (key, m["RMSE"], p)
        print(f"  {key:4s} R²={m['R2']:.3f}  MAE={m['MAE']:.2f} °C  RMSE={m['RMSE']:.2f} °C")
    res = pd.DataFrame(rows_out)
    res.round(4).to_csv(C.RESULTS_DIR / "future_year_test.csv", index=False)
    overall, _ = evaluate(test.assign(pred=best[2]), C.TARGET, "pred")
    print(f"SUHI agreement in {test_year} with {best[0]}: {overall['agreement_pct']}%")
    pd.DataFrame([{"model": best[0], **overall}]).to_csv(C.RESULTS_DIR / "future_year_suhi.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(res.algorithm, res.R2, color="#3C7FB1")
    for i, v in enumerate(res.R2):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Future-year test: trained on 2019–{int(test_year) - 1}, R² on {test_year}", loc="left")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "future_year_test.png", dpi=130)


if __name__ == "__main__":
    main()
