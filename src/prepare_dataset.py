"""Combine the Earth Engine exports (one CSV per city, .csv or .csv.gz) and clean them.

Cleaning (every step is counted in results/cleaning_report.csv):
  1. -9999 placeholders -> missing (filled later with the training median)
  2. same point, same date seen in two overlapping Landsat scenes -> averaged
  3. NDVI / NDBI / NDWI outside [-1, 1] (division by ~0 over dark pixels) -> removed
  4. LST outside 10–70 °C -> removed
  5. city-dates with < 400 clear points (mostly cloudy scenes) -> removed
  6. LST more than 4 robust SDs from its scene's median (cloud edges) -> removed
Optional: Amenity = nearest OpenStreetMap amenity within 250 m (needs internet).

    python -m src.prepare_dataset "exports/*.csv*" [--osm]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from src import config as C

ORIGINAL_ORDER = ["soil_moisture", "NDBI", "BU", "Roughness", "Slope", "NDVI", "LST", "UHI", "UTFVI", "NDWI",
                  "SAVI", "lulc_classes", "Amenity", "LandUse", "GHI", "CH4", "CO", "HCHO", "NO2", "O3", "SO2",
                  "Longitude", "Latitude", "Zone", "geometry"]
EXTRA = ["city", "date", "season"]
KEY = ["city", "date", "Longitude", "Latitude"]


def osm_amenity(points: pd.DataFrame, radius: float = 250) -> pd.Series:
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import box

    gp = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.Longitude, points.Latitude), crs=4326)
    area = box(gp.Longitude.min() - .01, gp.Latitude.min() - .01, gp.Longitude.max() + .01, gp.Latitude.max() + .01)
    amen = ox.features_from_polygon(area, tags={"amenity": True})
    if amen.empty:
        return pd.Series("none", index=points.index)
    utm = gp.estimate_utm_crs()
    amen = amen[["amenity", "geometry"]].reset_index(drop=True).to_crs(utm)
    amen["geometry"] = amen.geometry.centroid
    j = gpd.sjoin_nearest(gp.to_crs(utm), amen, how="left", max_distance=radius)
    return j[~j.index.duplicated()]["amenity"].fillna("none").astype(str).reindex(points.index)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [("raw rows", len(df))]
    df = df.drop(columns=[c for c in ["system:index", ".geo"] if c in df.columns]).replace(C.NODATA, np.nan)
    df["Longitude"], df["Latitude"] = df.Longitude.round(7), df.Latitude.round(7)

    num = [c for c in df.select_dtypes("number").columns if c not in KEY]
    other = [c for c in df.columns if c not in num and c not in KEY]
    n0 = len(df)
    df = df.groupby(KEY, as_index=False, sort=False).agg({**{c: "mean" for c in num}, **{c: "first" for c in other}})
    log.append(("merged duplicate scenes (same point, same date)", n0 - len(df)))

    bad = (df[["NDVI", "NDBI", "NDWI"]].abs() > 1).any(axis=1)
    df = df[~bad]
    log.append(("removed: NDVI/NDBI/NDWI outside [-1, 1]", int(bad.sum())))

    lo, hi = C.LST_RANGE
    bad = ~df.LST.between(lo, hi)
    df = df[~bad]
    log.append((f"removed: LST outside {lo:.0f}–{hi:.0f} °C", int(bad.sum())))

    size = df.groupby(C.GROUP).LST.transform("size")
    bad = size < C.MIN_POINTS_PER_SCENE
    log.append((f"removed: scenes with < {C.MIN_POINTS_PER_SCENE} clear points "
                f"({df.loc[bad, C.GROUP].drop_duplicates().shape[0]} city-dates)", int(bad.sum())))
    df = df[~bad]

    g = df.groupby(C.GROUP).LST
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    bad = ((df.LST - med) / mad.replace(0, np.nan)).abs() > C.ROBUST_Z
    df = df[~bad]
    log.append((f"removed: LST > {C.ROBUST_Z:g} robust SD from its scene (cloud edges)", int(bad.sum())))

    # recompute UHI / UTFVI on the cleaned scenes (same formulas as the export)
    g = df.groupby(C.GROUP).LST
    mu, sd = g.transform("mean"), g.transform("std")
    df["UHI"] = (df.LST - mu) / sd
    df["UTFVI"] = (df.LST - mu) / df.LST
    log.append(("final rows", len(df)))
    return df.reset_index(drop=True), pd.DataFrame(log, columns=["step", "rows"])


def build(files, use_osm=False):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Read {len(files)} file(s), {len(df):,} rows, cities: {sorted(df.city.unique())}")
    df, report = clean(df)
    if use_osm:
        pts = df[["city", "Longitude", "Latitude"]].drop_duplicates().reset_index(drop=True)
        pts["Amenity"] = pd.concat([osm_amenity(g) for _, g in pts.groupby("city")])
        df = df.merge(pts, on=["city", "Longitude", "Latitude"], how="left")
    df["geometry"] = "POINT (" + df.Longitude.astype(str) + " " + df.Latitude.astype(str) + ")"
    cols = [c for c in ORIGINAL_ORDER if c in df.columns] + EXTRA
    return df[cols].sort_values(["city", "date"]).reset_index(drop=True), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(C.CLEAN_DATA))
    ap.add_argument("--osm", action="store_true", help="add Amenity from OpenStreetMap (needs internet)")
    a = ap.parse_args()
    files = sorted({f for p in a.files for f in glob.glob(p)})
    df, report = build(files, a.osm)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(C.RESULTS_DIR / "cleaning_report.csv", index=False)
    df.to_csv(a.out, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns -> {a.out}")
    print(df.groupby("city").agg(rows=("LST", "size"), dates=("date", "nunique")).to_string())


if __name__ == "__main__":
    main()
