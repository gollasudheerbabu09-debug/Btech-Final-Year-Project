[README.md](https://github.com/user-attachments/files/31774762/README.md)
# Surface Urban Heat Island (SUHI) Prediction for Smart-City Planning

An end-to-end machine learning framework that predicts **Land Surface Temperature (LST)** from satellite-derived spectral indices, then flags Surface Urban Heat Island zones and renders them on an interactive map. Six regression models are benchmarked on a shared split over **1.97 million geospatial pixels**, and the best performer is deployed behind a working prediction and mapping pipeline.

---

## Highlights

- Built an **end-to-end ML framework for smart-city planning**, predicting Land Surface Temperature from Landsat 8-derived spectral indices and geospatial coordinates across **1,973,361 pixel records**
- **Benchmarked six regressors on a single shared 80/20 split** — Linear Regression, KNN (k=5, distance-weighted), MLP neural network, SVR, Decision Tree, and Random Forest — evaluated on a common MAE / RMSE / R² protocol
- **Decision Tree delivered the best performance** (MAE 0.00 °C, RMSE 0.07 °C), with Random Forest and KNN close behind, while Linear Regression confirmed the LST surface is strongly non-linear
- Extended prediction into a **SUHI detection layer** — statistical thresholding on predicted LST plus a **Folium interactive map** marking heat-island hotspots by coordinate
- Serialized all six trained models for reuse and built a **single-point and batch inference pipeline** for planning scenarios

---

## Problem

Urban surfaces — asphalt, concrete, roofing — absorb and re-radiate far more solar energy than vegetation or water. The result is the **Surface Urban Heat Island** effect: built-up zones running measurably hotter than their surroundings, driving up cooling demand, worsening air quality, and raising heat-stress mortality.

City planners need to answer a specific question: *if we develop this parcel, how hot does it get?* That requires predicting LST from land-surface characteristics rather than waiting for the next satellite pass — which is exactly what this framework does.

---

## Dataset

| Property | Value |
|---|---|
| Records | **1,973,361** pixels |
| Columns | 25 |
| Target | `LST` (Land Surface Temperature, °C) |
| Spatial coverage | Milan region (≈ 9.24° E, 45.42° N) |
| Missing values | None |

The full table carries 25 columns spanning four families:

**Spectral indices** — `NDVI` (vegetation), `NDBI` (built-up), `NDWI` (water), `SAVI` (soil-adjusted vegetation), `BU` (built-up index)

**Surface and terrain** — `soil_moisture`, `Roughness`, `Slope`, `lulc_classes`, `LandUse`, `Amenity`

**Thermal and derived** — `LST`, `UHI`, `UTFVI` (Urban Thermal Field Variance Index)

**Atmospheric (Sentinel-5P)** — `CH4`, `CO`, `HCHO`, `NO2`, `O3`, `SO2`, `GHI`

**Geospatial** — `Longitude`, `Latitude`, `Zone`, `geometry`

### Feature selection

Seven predictors were used:

```python
features = ['soil_moisture', 'NDBI', 'NDVI', 'NDWI', 'SAVI', 'Longitude', 'Latitude']
X = data[features]
y = data['LST']
```

`UHI` and `UTFVI` were deliberately excluded — both are computed *from* LST, so including them would be direct target leakage. The four spectral indices are the physically meaningful drivers: NDBI rises with impervious surface (hotter), NDVI rises with vegetation (cooler through evapotranspiration), NDWI marks water bodies (coolest), and SAVI corrects NDVI for soil background in sparsely vegetated areas.

---

## Exploratory Analysis

Before modeling, the feature space was profiled from several angles:

- **Histograms and KDE plots** — per-feature distributions and skew
- **Correlation heatmap** — inter-feature relationships across all 7 predictors plus LST
- **Pairplot** — pairwise scatter across the full feature set
- **Boxplots and violin plots** — outlier detection and density shape
- **Correlation bar chart** — each feature's direct correlation with LST, ranked
- **Hexbin plot** — Longitude vs LST density, exposing the spatial temperature gradient

The hexbin and correlation views are the ones that matter most here: they show LST varies systematically with location, which sets up both the strength and the central caveat of the results below.

---

## Modeling

All models are trained and evaluated on the **same split**, so the comparison is like-for-like:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

That yields ~1.58M training and ~394K test pixels.

| Model | Configuration | Scaling |
|---|---|---|
| Decision Tree | `DecisionTreeRegressor(random_state=42)` | None (scale-invariant) |
| Random Forest | `n_estimators=100, random_state=42` | MinMax |
| KNN | `n_neighbors=5, weights='distance'` | MinMax (required) |
| Linear Regression | default OLS | MinMax |
| MLP | `Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1)`, Adam, MSE, 100 epochs, batch 32, 10% val split | MinMax |
| SVR | `kernel='rbf', C=100, gamma='scale', epsilon=0.1` | MinMax |

KNN uses `weights='distance'` rather than uniform voting, so nearer neighbors dominate the prediction — appropriate when the underlying surface is spatially smooth. SVR was trained on a **150,000-sample subset**, since RBF-kernel SVR scales roughly quadratically and does not run on 1.58M rows in reasonable time.

---

## Results

Evaluated on the held-out ~394K-pixel test set:

| Rank | Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|
| 1 | **Decision Tree** | **0.02** | **0.07** | **1.00** |
| 2 | Random Forest | 0.05 | 0.04 | 1.00 |
| 3 | KNN (k=5) | 0.04 | 0.23 | 0.99 |
| 4 | MLP | 0.92 | 1.21 | 0.75 |
| 5 | SVR | 1.08 | 1.52 | 0.61 |
| 6 | Linear Regression | 1.46 | 1.99 | 0.34 |

Results were assembled into a ranking table sorted by R², then MAE, then RMSE, and visualized as grouped bar charts across all three metrics. Every model also has an **actual-vs-predicted scatter plot** against the 45° perfect-prediction line.

### Reading the results

**Linear Regression at R² = 0.34 is the most informative number in the table.** A linear model explains only a third of the variance in LST, which establishes that the relationship between spectral indices and surface temperature is strongly **non-linear** — exactly what the physics predicts, since evapotranspiration, thermal admittance, and surface albedo all interact rather than add.

**The tree-based models and KNN cluster at the top,** which is consistent with a target surface that is locally smooth but globally non-linear. Trees partition the feature space into regions and fit a constant within each; KNN averages nearby observations. Both suit this structure far better than a global linear fit or an RBF kernel with a single bandwidth.

**The MLP at R² = 0.75 underperforms the trees** despite 100 epochs on 1.58M samples. With only 7 input features and a target that is close to a spatial lookup, the network's advantage — learning distributed representations of high-dimensional input — has little to work with here.

### An important caveat on R² ≈ 1.00

Decision Tree and Random Forest reaching MAE ≈ 0.00 °C and R² = 1.00 should be read carefully rather than taken at face value.

`Longitude` and `Latitude` are among the seven features, and the dataset contains many pixels sharing identical or near-identical index values (adjacent pixels in a raster are highly autocorrelated — visible in the raw data, where consecutive rows repeat the same LST and NDVI values). With a **random pixel-level split**, a neighboring pixel of nearly every test point ends up in the training set. An unconstrained tree can then learn something close to a coordinate lookup table rather than a physical relationship between land cover and temperature.

This does not mean the models are broken — they are highly accurate *at interpolating within the mapped region*, which is genuinely useful for filling gaps and for the scenario tool below. But it means the reported R² should not be read as the model's ability to generalize to an unmapped city.

A more demanding protocol would:

- **Split spatially** — hold out contiguous blocks or grid tiles rather than random pixels
- **Drop `Longitude` / `Latitude`** and force the model to predict from land-surface characteristics alone
- **Hold out a separate zone** using the existing `Zone` column, and report cross-zone performance

Reporting both the interpolation score and a spatial-holdout score would make the framework substantially stronger. This is the clearest next step for the project.

---

## SUHI Detection

Prediction is the input to the actual planning output: identifying which locations qualify as heat islands.

Predicted LST is thresholded statistically, so the definition adapts to the local temperature distribution rather than depending on a fixed absolute cutoff:

```python
threshold = mean_lst + 0.5 * std_lst
df['Is_SUHI'] = df['Predicted_LST'] > threshold
```

Any pixel more than half a standard deviation above the regional mean is flagged as a Surface Urban Heat Island.

Flagged locations are then rendered on an interactive **Folium** map with `MarkerCluster`, color-coded — red for SUHI, green for normal — with popups showing predicted temperature and classification:

```python
color = 'red' if row['Is_SUHI'] else 'green'
folium.CircleMarker(
    location=[row['Latitude'], row['Longitude']],
    radius=7, color=color, fill=True, fill_color=color,
    fill_opacity=0.7,
    popup=folium.Popup(f"LST: {row['Predicted_LST']:.2f} °C<br>SUHI: {row['Is_SUHI']}")
).add_to(marker_cluster)
```

This is what makes the project a planning tool rather than a benchmark: a planner supplies candidate parcels, gets predicted temperatures, and sees the heat-island risk rendered geographically.

---

## Inference Pipeline

All six models are serialized (`pickle` for the scikit-learn models, `.keras` for the MLP) and reloaded for prediction:

```
decision_tree_model.pkl      random_forest_model.pkl  +  rf_scaler.pkl
knn_model.pkl                lr_model.pkl
svr_model_limited.pkl  +  svr_scaler_limited.pkl      mlp_model.keras  +  mlp_scaler.pkl
```

Three inference modes are implemented:

1. **Multi-model comparison** — one input vector scored by all six models side by side
2. **Interactive input** — prompts for the seven feature values and returns a Decision Tree prediction
3. **Batch scenario scoring** — an array of candidate sites scored at once, thresholded for SUHI, and mapped

```python
input_values = [soil_moisture, ndbi, ndvi, ndwi, savi, longitude, latitude]
pred = dt_model.predict(np.array(input_values).reshape(1, -1))[0]
print(f"Decision Tree Prediction: {pred:.2f} °C")
```

The Decision Tree is the deployed model — it is the top performer and needs no scaler at inference time, since trees are invariant to monotonic feature scaling.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | pandas, NumPy |
| Classical ML | scikit-learn (DecisionTree, RandomForest, KNN, LinearRegression, SVR) |
| Deep Learning | TensorFlow / Keras (MLP) |
| Preprocessing | MinMaxScaler |
| Visualization | Matplotlib, Seaborn |
| Geospatial | Folium, MarkerCluster |
| Persistence | pickle, joblib |
| Environment | Kaggle Notebooks |

---

## Running It

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn folium joblib
```

Open `finalyearprojectsuhi.ipynb` and run top to bottom. The notebook covers EDA → model training and benchmarking → serialization → inference → SUHI thresholding → interactive map.

Point the `pd.read_csv(...)` path at your local copy of `combined_v2.csv` and update the model-loading paths in the inference cells.

---

## Known Limitations & Next Steps

- **Random rather than spatial split.** The single most important fix; see the caveat section above. Adding a spatial-holdout evaluation alongside the current numbers would make the reported performance defensible for unseen regions.
- **Scaler handling at inference.** Several inference cells fit a fresh `MinMaxScaler` on the single input sample being predicted. Fitting a scaler on one row maps every feature to the same value, so scaled models receive degenerate input — which is why the per-model predictions for one identical input diverge so widely. The saved `rf_scaler.pkl`, `mlp_scaler.pkl`, and `svr_scaler_limited.pkl` should be loaded and applied instead. The Decision Tree path is unaffected, since it needs no scaling.
- **Seven of twenty-five columns used.** The atmospheric variables (`NO2`, `CO`, `O3`, `SO2`), terrain (`Slope`, `Roughness`), and land-use classes are all available and unused. Adding them — with feature-importance analysis — is a natural extension.
- **Single region, single time point.** Trained on one metropolitan area. Cross-city validation would test whether the learned relationships transfer.
- **No hyperparameter tuning.** All models use near-default settings. Given the tree models already saturate on this split, tuning matters most once a spatial split is in place.

---

## What This Project Demonstrates

- **Large-scale tabular ML** — a full pipeline over ~2 million geospatial records
- **Domain-informed feature selection** — recognizing `UHI` / `UTFVI` as LST-derived and excluding them to prevent target leakage
- **Systematic model benchmarking** on a shared split with a common metric protocol, spanning linear, instance-based, kernel, tree, ensemble, and neural approaches
- **Interpreting results physically** — reading Linear Regression's R² = 0.34 as evidence of non-linearity rather than as a failed run
- **Critical evaluation** — identifying spatial autocorrelation as the likely source of a near-perfect score, and specifying the protocol that would test it properly
- **Geospatial visualization and deployment** — statistical SUHI thresholding and interactive Folium mapping
- **End-to-end delivery** — from raw satellite-derived data through to a working scenario-scoring tool
