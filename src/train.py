"""Train LR, KNN (k=5), DT, RF, MLP and SVM to predict LST for 5 Andhra Pradesh
cities, evaluate on unseen places and unseen cities, map SUHI, save for the app.

    python -m src.train                      # uses data/andhra_pradesh_suhi.csv
    python -m src.train --max-rows 400000    # more rows (slower)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split

from src import config as C
from src.data import load, prepare, spatial_blocks, spatial_split, split_features
from src.models import MODEL_NAMES, build_pipeline, get_estimators
from src.suhi import classify, evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
HEAT, COOL, BLUE = "#B4232A", "#4E9A6B", "#3C7FB1"
SET_COLORS = [BLUE, "#5BA88F", "#E8833A", HEAT]
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-monsoon"]


def _r(y, p):
    return float(pearsonr(y, p)[0]) if np.std(p) > 0 else 0.0


def metrics(y, p):
    return {"r": _r(y, p), "R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


SCORING = {"r": make_scorer(_r), "R2": "r2", "MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error"}


def cap(df, key, n=None):
    n = n or C.ROW_CAPS.get(key)
    return df.sample(n, random_state=C.RANDOM_STATE) if n and len(df) > n else df


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ experiments
def leakage_check(df):
    num, cat = split_features(df, C.FEATURE_SETS["Set 4"]["cols"])
    rows = []
    for label, split, coords in [("Random split + coordinates", "random", True),
                                 ("Spatial split + coordinates", "spatial", True),
                                 ("Spatial split, no coordinates (used)", "spatial", False)]:
        n = num + (C.COORDS if coords else [])
        tr, te = (train_test_split(df, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE) if split == "random"
                  else spatial_split(df)[:2])
        pipe = build_pipeline(get_estimators()["DT"], n, cat).fit(tr[n + cat], tr[C.TARGET])
        rows.append({"setup": label, **metrics(te[C.TARGET], pipe.predict(te[n + cat]))})
    return pd.DataFrame(rows)


CACHE = C.ROOT.parent / "suhi_cache"


def cached(name, fn):
    """Run fn() once; reuse the saved result if the run is restarted."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    out = fn()
    joblib.dump(out, path)
    return out


def run_sets(df, train, test, cv_folds):
    blocks = spatial_blocks(train)
    cv_idx = train.sample(min(C.CV_ROWS, len(train)), random_state=C.RANDOM_STATE).index
    test_rows, cv_rows, fitted = [], [], {}
    for set_name, spec in C.FEATURE_SETS.items():
        num, cat = split_features(df, spec["cols"])
        cols = num + cat
        log(f"\n{set_name} ({spec['desc']}): {len(cols)} inputs -> {cols}")
        for key, est in get_estimators().items():
          def one(key=key, est=est):
            t0 = time.time()
            cv_out = []
            pipe = build_pipeline(est, num, cat)
            if cv_folds > 1:
                cvd = cap(train.loc[cv_idx], key)
                res = cross_validate(pipe, cvd[cols], cvd[C.TARGET], groups=blocks.loc[cvd.index],
                                     cv=GroupKFold(n_splits=cv_folds), scoring=SCORING, n_jobs=-1)
                for i in range(cv_folds):
                    cv_out.append({"set": set_name, "algorithm": key, "fold": i, "r": res["test_r"][i],
                                    "R2": res["test_R2"][i], "MAE": -res["test_MAE"][i],
                                    "RMSE": -res["test_RMSE"][i]})
            tr = cap(train, key)
            pipe.fit(tr[cols], tr[C.TARGET])
            pred = pipe.predict(test[cols])
            m = metrics(test[C.TARGET], pred)
            row = {"set": set_name, "algorithm": key, "train_rows": len(tr), **m, "seconds": round(time.time() - t0)}
            return row, cv_out, pipe, pred
          row, cv_out, pipe, pred = cached(f"{set_name}_{key}", one)
          test_rows.append(row)
          cv_rows.extend(cv_out)
          fitted[(set_name, key)] = (pipe, pred, num, cat)
          log(f"  {key:4s} r={row['r']:.3f}  R²={row['R2']:.3f}  MAE={row['MAE']:.2f} °C  RMSE={row['RMSE']:.2f} °C"
              f"  ({row['seconds']}s)")
    return pd.DataFrame(test_rows), pd.DataFrame(cv_rows), fitted


def leave_one_city_out(df, set_name):
    """Train on 4 cities, test on the 5th – can the models transfer to a new city?"""
    d = cap(df, None, C.LOCO_ROWS)
    num, cat = split_features(df, C.FEATURE_SETS[set_name]["cols"])
    rows = []
    for city in sorted(d.city.unique()):
        tr, te = d[d.city != city], d[d.city == city]

        def one(tr=tr, te=te, city=city):
            out = []
            for key, est in get_estimators().items():
                t = cap(tr, key)
                pipe = build_pipeline(est, num, cat).fit(t[num + cat], t[C.TARGET])
                out.append({"test_city": city, "algorithm": key, **metrics(te[C.TARGET], pipe.predict(te[num + cat]))})
            return out
        rows.extend(cached(f"loco_{set_name}_{city}", one))
        log(f"  unseen city {city}: " + ", ".join(f"{r['algorithm']} R²={r['R2']:.2f}" for r in rows[-6:]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def save(fig, name, dpi=130):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=dpi)
    plt.close(fig)


def fig_eda(full):
    cities = sorted(full.city.unique())
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(SEASON_ORDER)
    colors = [BLUE, HEAT, COOL, "#E8833A"]
    for k, s in enumerate(SEASON_ORDER):
        data = [full[(full.city == c) & (full.season == s)].LST.values for c in cities]
        pos = np.arange(len(cities)) + (k - 1.5) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=colors[k], alpha=0.8)
        ax.plot([], [], color=colors[k], lw=8, label=s)
    ax.set_xticks(range(len(cities)), cities)
    ax.set_ylabel("LST (°C)")
    ax.set_title("Land surface temperature by city and season (2019–2024)", loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "lst_by_city_season.png")


def fig_cv(cv_df):
    if cv_df.empty:
        return
    algos, sets = list(MODEL_NAMES), list(cv_df["set"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    w = 0.8 / len(sets)
    for ax, met in zip(axes, ["R2", "MAE", "RMSE"]):
        for k, s in enumerate(sets):
            data = [cv_df[(cv_df["set"] == s) & (cv_df.algorithm == a)][met].values for a in algos]
            bp = ax.boxplot(data, positions=np.arange(len(algos)) + (k - (len(sets) - 1) / 2) * w,
                            widths=w * 0.85, patch_artist=True, flierprops={"markersize": 3})
            for b in bp["boxes"]:
                b.set(facecolor=SET_COLORS[k % 4], alpha=0.8)
            ax.plot([], [], color=SET_COLORS[k % 4], lw=8, label=f"{s}: {C.FEATURE_SETS[s]['desc']}")
        ax.set_xticks(range(len(algos)), algos)
        ax.set_ylabel("R²" if met == "R2" else f"{met} (°C)")
        ax.set_title(f"Spatial cross-validation – {'R²' if met == 'R2' else met}", loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "cv_boxplots.png")


def fig_test(test_df):
    sets, algos = list(test_df["set"].unique()), list(MODEL_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.8 / len(sets)
    for ax, met in zip(axes, ["R2", "RMSE"]):
        for k, s in enumerate(sets):
            t = test_df[test_df["set"] == s].set_index("algorithm").reindex(algos)
            ax.bar(np.arange(len(algos)) + (k - (len(sets) - 1) / 2) * w, t[met], w, color=SET_COLORS[k % 4], label=s)
        ax.set_xticks(range(len(algos)), algos)
        ax.set_title(f"Test (unseen places) – {'R²' if met == 'R2' else 'RMSE (°C)'}", loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "test_metrics.png")


def fig_leakage(leak):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.barh(leak["setup"], leak["R2"], color=["#999999", "#E8833A", HEAT])
    for i, (r2, mae) in enumerate(zip(leak["R2"], leak["MAE"])):
        ax.text(max(r2, 0) + 0.01, i, f"R² {r2:.2f}, MAE {mae:.2f} °C", va="center", fontsize=9)
    ax.set_xlim(0, 1.35)
    ax.invert_yaxis()
    ax.set_title("Decision Tree under different evaluation setups", loc="left")
    save(fig, "leakage_check.png")


def fig_loco(loco):
    p = loco.pivot(index="test_city", columns="algorithm", values="R2")[list(MODEL_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(p.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(p.shape[1]), p.columns)
    ax.set_yticks(range(p.shape[0]), p.index)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="R²")
    ax.set_title("Unseen city: R² when trained on the other four cities", loc="left")
    save(fig, "leave_one_city_out.png")


def fig_pred(test, pred, best):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    hb = ax.hexbin(test[C.TARGET], pred, gridsize=70, cmap="YlOrRd", mincnt=1, bins="log")
    lo, hi = np.percentile(np.r_[test[C.TARGET], pred], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Satellite LST (°C)", ylabel="Predicted LST (°C)",
           title=f"{MODEL_NAMES[best[1]]}, {best[0]}")
    fig.colorbar(hb, ax=ax, label="points (log)")
    save(fig, "pred_vs_reference.png")


def fig_suhi(test, pred, best):
    d = test.assign(pred=pred)
    d["suhi_ref"], d["suhi_pred"] = classify(d, C.TARGET), classify(d, "pred")
    cities = sorted(d.city.unique())
    fig, axes = plt.subplots(2, len(cities), figsize=(3.6 * len(cities), 7.4), squeeze=False)
    for j, city in enumerate(cities):
        g = d[d.city == city]
        date = g.groupby("date").size().idxmax()
        g = g[g.date == date]
        for i, (flag, lab) in enumerate([("suhi_ref", "Satellite"), ("suhi_pred", f"Predicted ({best[1]})")]):
            ax = axes[i, j]
            ax.scatter(g.Longitude, g.Latitude, s=9, c=np.where(g[flag], HEAT, COOL), edgecolor="none")
            ax.set_title(f"{city}\n{date} – {lab}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect(1 / np.cos(np.radians(g.Latitude.mean())))
    fig.suptitle("Heat islands (red) at test locations, busiest date per city", x=0.01, ha="left")
    save(fig, "suhi_maps.png", dpi=120)


def fig_importance(imp, corr):
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.33 * max(len(imp), len(corr)) + 1.5))
    imp = imp.sort_values()
    axes[0].barh(imp.index, imp.values, color=HEAT)
    axes[0].set_title("Permutation importance (increase in RMSE, °C)", loc="left", fontsize=10)
    corr = corr.sort_values()
    axes[1].barh(corr.index, corr.values, color=[HEAT if v > 0 else BLUE for v in corr.values])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("Correlation with LST (season: correlation ratio)", loc="left", fontsize=10)
    save(fig, "feature_importance.png")


def associations(df):
    num, _ = split_features(df, C.FEATURE_SETS["Set 4"]["cols"])
    corr = df[num].corrwith(df[C.TARGET]).dropna()
    grand = df[C.TARGET].mean()
    eta = np.sqrt(sum(len(g) * (g.mean() - grand) ** 2 for _, g in df.groupby("season")[C.TARGET])
                  / ((df[C.TARGET] - grand) ** 2).sum())
    return pd.concat([corr, pd.Series({"season": eta})])


# ------------------------------------------------------------------ saving
def save_app_sample(full):
    """Complete scenes (all points of a date) for the app map: up to N dates per city, spread over seasons."""
    picks = []
    for city, g in full.groupby("city"):
        sizes = g.groupby(["season", "date"]).size().reset_index(name="n").sort_values("n", ascending=False)
        chosen = sizes.groupby("season").head(1).head(C.APP_DATES_PER_CITY)
        picks.append(g[g.date.isin(chosen.date)])
    s = pd.concat(picks)
    cols = [c for c in s.columns if c not in ("geometry", "UHI", "UTFVI")]
    s[cols].round(6).to_csv(C.APP_SAMPLE, index=False)
    return len(s)


def save_models(train, fitted, set_name, best_key, extra):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sizes, num, cat = {}, None, None
    for key in MODEL_NAMES:
        pipe, _, num, cat = fitted[(set_name, key)]
        path = C.MODELS_DIR / f"{key}.joblib"
        joblib.dump(pipe, path, compress=3)
        sizes[key] = round(path.stat().st_size / 1e6, 1)
    per_scene = train.groupby(C.GROUP + ["season"])[C.TARGET].agg(["mean", "std"]).reset_index()
    per_scene["thr"] = per_scene["mean"] + C.SUHI_K * per_scene["std"]
    meta = {
        "sklearn_version": sklearn.__version__, "target": C.TARGET,
        "feature_set": set_name, "feature_set_desc": C.FEATURE_SETS[set_name]["desc"],
        "numeric_features": num, "categorical_features": cat,
        "categories": {c: sorted(train[c].unique().tolist()) for c in cat},
        "feature_stats": {c: {"min": float(train[c].quantile(0.005)), "max": float(train[c].quantile(0.995)),
                              "median": float(train[c].median())} for c in num},
        "suhi_threshold_by_city_season": {f"{r.city}|{r.season}": round(r.thr, 2) for r in
                                          per_scene.groupby(["city", "season"]).thr.mean().reset_index().itertuples()},
        "suhi_threshold_by_season": per_scene.groupby("season").thr.mean().round(2).to_dict(),
        "cities": sorted(train.city.unique().tolist()),
        "best_model": best_key, "model_names": MODEL_NAMES, "model_size_mb": sizes, **extra,
    }
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    return meta


def quick_fit():
    """Retrain the six models on the app sample if saved models can't be loaded (e.g. version mismatch)."""
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    df = prepare(pd.read_csv(C.APP_SAMPLE))
    num, cat = meta["numeric_features"], meta["categorical_features"]
    for key, est in get_estimators().items():
        t = cap(df, key)
        joblib.dump(build_pipeline(est, num, cat).fit(t[num + cat], t[C.TARGET]),
                    C.MODELS_DIR / f"{key}.joblib", compress=3)
    meta["sklearn_version"] = sklearn.__version__
    meta["retrained_on_app_sample"] = True
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.CLEAN_DATA))
    ap.add_argument("--max-rows", type=int, default=C.MAX_ROWS)
    ap.add_argument("--cv-folds", type=int, default=C.CV_FOLDS)
    ap.add_argument("--skip-loco", action="store_true")
    a = ap.parse_args()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    full = prepare(pd.read_csv(a.data, low_memory=False))
    log(f"Full dataset: {len(full):,} rows, {full.city.nunique()} cities, "
        f"{full[C.GROUP].drop_duplicates().shape[0]} scenes")
    full.groupby("city").agg(rows=("LST", "size"), scenes=("date", "nunique"),
                             points=("Longitude", "nunique"), mean_LST=("LST", "mean")).round(2) \
        .to_csv(C.RESULTS_DIR / "dataset_summary.csv")
    fig_eda(full)
    assoc = associations(full)
    n_app = save_app_sample(full)

    df = full.sample(min(a.max_rows, len(full)), random_state=C.RANDOM_STATE).reset_index(drop=True)
    del full
    train, test, groups = spatial_split(df)
    log(f"Sample {len(df):,} rows -> spatial split over {groups.nunique()} blocks: "
        f"train {len(train):,}, test {len(test):,} (test places never seen in training)")

    log("\nLeakage check (Decision Tree):")
    leak = leakage_check(df)
    log(leak.round(3).to_string(index=False))
    leak.round(4).to_csv(C.RESULTS_DIR / "leakage_check.csv", index=False)

    test_df, cv_df, fitted = run_sets(df, train, test, a.cv_folds)
    test_df.round(4).to_csv(C.RESULTS_DIR / "test_metrics.csv", index=False)
    if not cv_df.empty:
        cv_df.round(4).to_csv(C.RESULTS_DIR / "cv_scores.csv", index=False)
        cv_df.groupby(["set", "algorithm"])[["r", "R2", "MAE", "RMSE"]].agg(["mean", "std"]).round(3) \
            .to_csv(C.RESULTS_DIR / "cv_summary.csv")

    best_row = test_df.sort_values(["RMSE", "MAE"]).iloc[0]
    best = (best_row["set"], best_row["algorithm"])
    pipe, pred, num, cat = fitted[best]
    log(f"\nBest: {MODEL_NAMES[best[1]]} with {best[0]} – R² {best_row['R2']:.3f}, "
        f"MAE {best_row['MAE']:.2f} °C, RMSE {best_row['RMSE']:.2f} °C")

    suhi_overall, suhi_season = evaluate(test.assign(pred=pred), C.TARGET, "pred")
    suhi_season.to_csv(C.RESULTS_DIR / "suhi_by_season.csv", index=False)
    pd.DataFrame([suhi_overall]).to_csv(C.RESULTS_DIR / "suhi_comparison.csv", index=False)
    per_city = pd.DataFrame([{"city": c, **metrics(g[C.TARGET], pred[g.index])} for c, g in test.groupby("city")])
    per_city.round(3).to_csv(C.RESULTS_DIR / "best_model_by_city.csv", index=False)

    sample = test.sample(min(5000, len(test)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(pipe, sample[num + cat], sample[C.TARGET], n_repeats=5,
                                scoring="neg_root_mean_squared_error", random_state=C.RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=num + cat)
    pd.DataFrame({"permutation_importance": imp, "association_with_LST": assoc}).round(4) \
        .to_csv(C.RESULTS_DIR / "feature_importance.csv")

    loco = pd.DataFrame()
    if not a.skip_loco:
        log(f"\nLeave-one-city-out ({best[0]}):")
        loco = leave_one_city_out(df, best[0])
        loco.round(4).to_csv(C.RESULTS_DIR / "leave_one_city_out.csv", index=False)
        fig_loco(loco)

    fig_cv(cv_df); fig_test(test_df); fig_leakage(leak); fig_pred(test, pred, best)
    fig_suhi(test, pred, best); fig_importance(imp, assoc)

    meta = save_models(train, fitted, best[0], best[1], {
        "rows_used": len(df), "train_rows": len(train), "test_rows": len(test), "app_sample_rows": n_app,
        "test_metrics": test_df[test_df["set"] == best[0]].set_index("algorithm")[["r", "R2", "MAE", "RMSE"]]
        .round(3).to_dict(orient="index"),
        "leakage_check": leak.round(3).to_dict(orient="records"), "suhi_comparison": suhi_overall,
    })
    log("\n=== Test R² (unseen places) ===")
    log(test_df.pivot_table(index="algorithm", columns="set", values="R2").round(3).to_string())
    log("\nSUHI by season:\n" + suhi_season.to_string(index=False))
    log(f"SUHI agreement: {suhi_overall['agreement_pct']}%  | model sizes MB: {meta['model_size_mb']}")
    log(f"scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
