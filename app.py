"""Andhra Pradesh urban heat islands – predict land surface temperature with six
machine-learning models and map surface urban heat islands.

Run locally:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.predict import add_suhi, load_models, missing_columns, models_available, predict

st.set_page_config(page_title="Andhra Pradesh Heat Islands", page_icon="🌡️", layout="wide")

LABELS = {
    "NDVI": "NDVI (vegetation)", "NDBI": "NDBI (built-up)", "NDWI": "NDWI (water)",
    "SAVI": "SAVI (soil-adjusted vegetation)", "BU": "BU (NDBI − NDVI)",
    "soil_moisture": "Soil moisture (m³/m³)", "Roughness": "Roughness (m)", "Slope": "Slope (°)",
    "GHI": "Solar radiation (kWh/m²/day)", "CH4": "Methane CH₄ (ppb)", "CO": "Carbon monoxide CO (mol/m²)",
    "HCHO": "Formaldehyde HCHO (mol/m²)", "NO2": "Nitrogen dioxide NO₂ (mol/m²)", "O3": "Ozone O₃ (mol/m²)",
    "SO2": "Sulphur dioxide SO₂ (mol/m²)", "lulc_classes": "Land cover (ESA WorldCover code)",
    "LandUse": "Land use (Dynamic World)", "season": "Season", "Amenity": "Amenity",
}
LULC = {"10": "10 – Trees", "20": "20 – Shrubland", "30": "30 – Grassland", "40": "40 – Cropland",
        "50": "50 – Built-up", "60": "60 – Bare", "80": "80 – Water", "90": "90 – Wetland", "95": "95 – Mangroves"}
GROUPS = [("Satellite indices", C.SPECTRAL), ("Surface, terrain and climate", C.SURFACE),
          ("Land cover and land use", C.LAND_USE), ("Air quality (Sentinel-5P)", C.AIR)]
HEAT, COOL = "#B4232A", "#4E9A6B"


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    try:
        return load_models()
    except Exception:
        from src.train import quick_fit   # version mismatch on the server: retrain on the app sample
        quick_fit()
        return load_models()


@st.cache_data
def get_sample():
    return pd.read_csv(C.APP_SAMPLE) if C.APP_SAMPLE.exists() else None


if not models_available():
    st.error("No trained models found. Run `python -m src.train` and commit the models/ and data/ folders.")
    st.stop()

models, meta = get_models()
sample = get_sample()
names, best = meta["model_names"], meta.get("best_model", "RF")
num, cat = meta["numeric_features"], meta["categorical_features"]

st.title("Where do Andhra Pradesh's cities overheat?")
st.write(
    "Land surface temperature (LST) for Vijayawada, Visakhapatnam, Guntur, Tirupati and Nellore, predicted from "
    "satellite indices, terrain, climate, land use and air quality. A spot is a **surface urban heat island** when "
    "it is hotter than its city's mean + 0.5 × standard deviation on that day. "
    "Method after Furuya et al. (2023), *Environmental Earth Sciences* 82:325."
)
tab_map, tab_one, tab_models, tab_about = st.tabs(
    ["Heat-island map", "Predict one location", "Compare models", "About the data"])

# ------------------------------------------------------------------ map
with tab_map:
    up = st.file_uploader("Use your own points (CSV with the same columns) or the built-in satellite scenes",
                          type="csv")
    src = None
    if up is not None:
        src = pd.read_csv(up)
        if miss := missing_columns(src, meta):
            st.error("This file is missing: " + ", ".join(miss))
            src = None
    elif sample is not None:
        src = sample
    if src is not None:
        c1, c2, c3 = st.columns(3)
        key = c1.selectbox("Model", list(models), index=list(models).index(best), format_func=lambda k: names[k])
        if "city" in src:
            city = c2.selectbox("City", sorted(src.city.unique()))
            src = src[src.city == city]
        if "date" in src:
            opts = (src[["date", "season"]].drop_duplicates().sort_values("date")
                    if "season" in src else src[["date"]].drop_duplicates())
            date = c3.selectbox("Date", opts.date.tolist(),
                                format_func=lambda d: f"{d} ({opts.set_index('date').season.get(d, '')})"
                                if "season" in opts else d)
            src = src[src.date == date]
        out = add_suhi(predict(src, models, meta, [key]), f"pred_{key}")
        has_ref = C.TARGET in out
        m = st.columns(4)
        m[0].metric("Points", f"{len(out):,}")
        m[1].metric("Predicted heat-island points", f"{out.suhi.mean():.0%}")
        if has_ref:
            ref = add_suhi(out, C.TARGET)
            m[2].metric("Agreement with satellite", f"{(ref.suhi == out.suhi).mean():.0%}")
            m[3].metric("Mean error", f"{(out[f'pred_{key}'] - out[C.TARGET]).abs().mean():.2f} °C")
            st.caption("These built-in points include places the models were trained on, so agreement here is "
                       "optimistic. Scores on unseen places are in 'Compare models'.")
        if set(C.COORDS) <= set(out.columns):
            def draw(d, flag, title):
                d = d.assign(Status=d[flag].map({True: "Heat island", False: "Not a heat island"}))
                fig = px.scatter_map(d, lat="Latitude", lon="Longitude", color="Status", zoom=11, height=480,
                                     map_style="carto-positron",
                                     color_discrete_map={"Heat island": HEAT, "Not a heat island": COOL},
                                     hover_data={f"pred_{key}": ":.1f", "Latitude": False, "Longitude": False,
                                                 **({C.TARGET: ":.1f"} if has_ref else {})})
                fig.update_traces(marker={"size": 7, "opacity": 0.8})
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title=title, legend_title_text="")
                return fig
            if has_ref:
                a, b = st.columns(2)
                a.plotly_chart(draw(ref.assign(**{f"pred_{key}": out[f"pred_{key}"].values}), "suhi",
                                    "From satellite temperature"), use_container_width=True)
                b.plotly_chart(draw(out, "suhi", f"From {names[key]}"), use_container_width=True)
            else:
                st.plotly_chart(draw(out, "suhi", f"From {names[key]}"), use_container_width=True)
        st.download_button("Download predictions", out.to_csv(index=False), "lst_predictions.csv", "text/csv")

# ------------------------------------------------------------------ one location
with tab_one:
    left, right = st.columns([3, 2], gap="large")
    inputs = {}
    with left:
        st.subheader("Describe the location")
        st.caption("Values start at the typical (median) value. Change any of them.")
        c1, c2 = st.columns(2)
        city = c1.selectbox("City (for the heat-island threshold)", meta["cities"])
        if "season" in cat:
            inputs["season"] = c2.selectbox("Season", meta["categories"]["season"])
        for title, cols in GROUPS:
            feats = [c for c in cols if c in num + cat]
            if not feats:
                continue
            with st.expander(title, expanded=title == "Satellite indices"):
                grid = st.columns(2)
                for i, f in enumerate(feats):
                    if f in cat:
                        opts = meta["categories"][f]
                        inputs[f] = grid[i % 2].selectbox(LABELS.get(f, f), opts,
                                                          format_func=lambda v, f=f: LULC.get(v, v) if f == "lulc_classes" else v)
                    else:
                        s = meta["feature_stats"][f]
                        inputs[f] = grid[i % 2].number_input(LABELS.get(f, f), value=float(s["median"]),
                                                             format="%.6g",
                                                             help=f"Typical range: {s['min']:.4g} to {s['max']:.4g}")
    with right:
        st.subheader("Predicted surface temperature")
        res = predict(pd.DataFrame([inputs]), models, meta)
        val = res[f"pred_{best}"].iloc[0]
        st.metric(f"{names[best]} (best model)", f"{val:.1f} °C")
        thr = meta["suhi_threshold_by_city_season"].get(f"{city}|{inputs.get('season')}")
        if thr is not None:
            if val > thr:
                st.error(f"Likely a heat island: above the typical {inputs['season'].lower()} threshold "
                         f"for {city} ({thr:.1f} °C).")
            else:
                st.success(f"Not a heat island: below the typical {inputs['season'].lower()} threshold "
                           f"for {city} ({thr:.1f} °C).")
        st.dataframe(pd.DataFrame({"Model": [names[k] for k in models],
                                   "LST (°C)": [res[f"pred_{k}"].iloc[0] for k in models]}),
                     hide_index=True, use_container_width=True)
        odd = [LABELS.get(f, f) for f in num
               if not meta["feature_stats"][f]["min"] <= inputs[f] <= meta["feature_stats"][f]["max"]]
        if odd:
            st.warning("Unusual values, so treat with caution: " + ", ".join(odd))

# ------------------------------------------------------------------ models
with tab_models:
    st.subheader("How well each model predicts places it has never seen")
    st.write("Locations were grouped into ~1 km blocks; 20% of blocks (all their dates) were held out for testing.")
    tm = C.RESULTS_DIR / "test_metrics.csv"
    if tm.exists():
        t = pd.read_csv(tm)
        sets = t["set"].unique().tolist()
        chosen = st.radio("Inputs", sets, index=sets.index(meta["feature_set"]), horizontal=True,
                          captions=[C.FEATURE_SETS[s]["desc"] for s in sets])
        tt = t[t["set"] == chosen].assign(Model=lambda d: d.algorithm.map(names))
        st.dataframe(tt[["Model", "r", "R2", "MAE", "RMSE"]].round(3)
                     .rename(columns={"R2": "R²", "MAE": "MAE (°C)", "RMSE": "RMSE (°C)"}),
                     hide_index=True, use_container_width=True)
    for f, title in [("leave_one_city_out.csv", "A city the models never saw (trained on the other four)"),
                     ("suhi_by_season.csv", "Heat islands by season: mean LST of heat-island vs other places (°C)"),
                     ("leakage_check.csv", "Why the split matters")]:
        p = C.RESULTS_DIR / f
        if p.exists():
            st.subheader(title)
            d = pd.read_csv(p)
            if f == "leave_one_city_out.csv":
                d = d.pivot(index="test_city", columns="algorithm", values="R2")[list(names)].round(3)
                st.caption("R² per unseen city")
            st.dataframe(d.round(3), use_container_width=True)
    for img, cap in [("test_metrics.png", "Test scores by feature set"),
                     ("cv_boxplots.png", "Spatial cross-validation"),
                     ("feature_importance.png", "What drives surface temperature"),
                     ("suhi_maps.png", "Heat islands at unseen test locations: satellite vs predicted"),
                     ("leave_one_city_out.png", "Transfer to an unseen city"),
                     ("future_year_test.png", "Unseen year: trained on 2019–2023, tested on 2024"),
                     ("lst_by_city_season.png", "Temperature by city and season"),
                     ("pred_vs_reference.png", "Predicted vs satellite LST"),
                     ("leakage_check.png", "Evaluation setups compared")]:
        p = C.FIG_DIR / img
        if p.exists():
            st.image(str(p), caption=cap)

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("Data")
    st.markdown(f"""
Five Andhra Pradesh cities (Vijayawada, Visakhapatnam, Guntur, Tirupati, Nellore). About 2,000 fixed points per
city, measured on every clear Landsat 8/9 date from 2019 to 2024, extracted with Google Earth Engine.

| Variable | Source |
|---|---|
| LST (target) | Landsat 8/9 thermal band |
| NDVI, NDBI, NDWI, SAVI, BU | Landsat surface reflectance, same scene |
| soil moisture, solar radiation (GHI) | ERA5-Land |
| slope, roughness | SRTM elevation |
| land cover / land use | ESA WorldCover, Google Dynamic World |
| CH₄, CO, HCHO, NO₂, O₃, SO₂ | Sentinel-5P |
| season | Winter (Jan–Feb), Summer (Mar–May), Monsoon (Jun–Sep), Post-monsoon (Oct–Dec) |

**Not used as inputs:** UHI and UTFVI (calculated from LST), coordinates (used only for the split and maps).

**Models** ({meta["feature_set"]}: {meta["feature_set_desc"].lower()}): linear regression, k-nearest neighbours
(k = 5), decision tree, random forest, multilayer perceptron, support vector machine. Trained on
{meta.get("train_rows", 0):,} points; inputs scaled to 0–1 inside each saved model.
""")
