# Surface Urban Heat Islands in Andhra Pradesh with Machine Learning

Predicting **land surface temperature (LST)** for five Andhra Pradesh cities from satellite, climate, terrain,
land-use and air-quality data with six machine-learning models, and mapping **surface urban heat islands (SUHI)**.

Method based on Furuya et al. (2023), *A machine learning approach for mapping surface urban heat island using
environmental and socioeconomic variables*, Environmental Earth Sciences 82:325,
https://doi.org/10.1007/s12665-023-11017-8

**Live app:** `https://<your-app>.streamlit.app` (add after deploying)

## Data

Extracted with Google Earth Engine (`gee/extract_suhi_features.js`): about 2,000 fixed points per city, measured on
every clear Landsat 8/9 date from 2019 to 2024.

| City | Rows (clean) | Scenes | Mean LST (°C) |
|---|---|---|---|
| Guntur | 200,408 | 170 | 40.7 |
| Nellore | 241,499 | 177 | 39.3 |
| Tirupati | 174,628 | 102 | 36.5 |
| Vijayawada | 142,491 | 91 | 37.9 |
| Visakhapatnam | 171,551 | 104 | 34.3 |
| **Total** | **930,577** | **644** | |

| Column | Source |
|---|---|
| LST (target) | Landsat 8/9 Collection 2 Level 2, band ST_B10 (°C) |
| NDVI, NDBI, NDWI, SAVI, BU | Landsat surface reflectance, same scene (BU = NDBI − NDVI) |
| soil_moisture, GHI | ERA5-Land daily (m³/m³; kWh/m²/day) |
| Slope, Roughness | SRTM 30 m elevation (degrees; m) |
| lulc_classes, LandUse | ESA WorldCover 2021 code; Google Dynamic World class (±15 days) |
| CH4, CO, HCHO, NO2, O3, SO2 | Sentinel-5P (±15 days; CH4 ±30 days) |
| season | IMD: Winter (Jan–Feb), Summer (Mar–May), Monsoon (Jun–Sep), Post-monsoon (Oct–Dec) |
| UHI, UTFVI | (LST − mean)/SD and (LST − mean)/LST per city and date. **Not used as inputs** (derived from LST) |
| Longitude, Latitude, Zone | point location (used only for splitting and maps) |

**Cleaning** (`src/prepare_dataset.py`, `results/cleaning_report.csv`): from 1,140,677 raw rows, 181,386 duplicates
from overlapping Landsat scenes were averaged; 5,000 rows with NDVI/NDBI/NDWI outside [−1, 1], 304 with LST outside
10–70 °C, 85 mostly cloudy scenes (17,474 rows) and 5,936 cloud-edge outliers were removed. Missing values
(mainly CH4, 43%) are filled with the training median inside each model.

## Method

- **Models:** Linear Regression, K-Nearest Neighbors (k = 5), Decision Tree, Random Forest (100 trees), Multilayer
  Perceptron (64-32), Support Vector Machine (RBF SVR, trained on 15,000 rows). Inputs are Min-Max scaled to [0, 1]
  inside each saved model (as in the paper); categories are one-hot encoded.
- **Feature sets** (each adds one group): Set 1 spectral indices + season → Set 2 + surface, terrain, climate →
  Set 3 + land cover/use → Set 4 + air quality.
- **Evaluation:** 200,000 sampled rows. Points were grouped into ~1 km blocks and 20% of blocks (with all their
  dates) were held out, so the test measures **places never seen in training**. 5-fold spatial cross-validation
  on the training part. Extra tests: an **unseen city** and an **unseen year**.
- **SUHI:** a point is a heat island when LST > mean + 0.5 × SD of its city on that date (paper Eq. 7–8).

## Results

**Test R² on unseen places**

| Model | Set 1 | Set 2 | Set 3 | Set 4 |
|---|---|---|---|---|
| Linear Regression | 0.53 | 0.56 | 0.56 | 0.62 |
| KNN (k = 5) | 0.52 | 0.72 | 0.70 | 0.81 |
| Decision Tree | 0.47 | 0.81 | 0.80 | 0.82 |
| **Random Forest** | 0.59 | 0.82 | 0.82 | **0.86** |
| MLP | 0.58 | 0.66 | 0.65 | 0.75 |
| SVM | 0.56 | 0.61 | 0.61 | 0.68 |

Best: **Random Forest, all variables: R² 0.86, MAE 1.84 °C, RMSE 2.56 °C**. Tree-based models perform best, as in
the paper. Climate variables (Set 2) give the largest gain because they describe the weather of each day.

**Heat islands:** maps from predicted LST agree with satellite-based maps for **77.8%** of test points. Heat islands
are hotter than other places in every season:

| Season | Satellite: SUHI − non-SUHI | Predicted |
|---|---|---|
| Monsoon | +6.1 °C | +5.6 °C |
| Summer | +5.6 °C | +4.5 °C |
| Winter | +4.2 °C | +3.6 °C |
| Post-monsoon | +3.8 °C | +3.3 °C |

**Unseen city** (trained on the other four): best R² 0.68 (Vijayawada, RF) and 0.61 (Guntur, RF); Tirupati is the
hardest (≤ 0.30), likely because of its hilly terrain and different climate.

**Unseen year** (trained 2019–2023, tested on 2024): SVM R² 0.64, Linear Regression 0.63, Random Forest 0.62,
MLP 0.55, KNN 0.47, Decision Tree 0.41; heat-island agreement 76.8%. Simpler models generalise better over time,
while single trees and KNN partly memorise date-specific weather.

**Most influential variables** (permutation importance): O₃ (follows season and weather), NDBI (built-up
surfaces) and solar radiation (GHI).

**Leakage check:** a random split gives R² 0.88 against 0.82 with the spatial split (Decision Tree), so the honest
spatial evaluation is used throughout.

Figures in `results/figures/`; all numbers in `results/*.csv`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py                         # web app, http://localhost:8501
```

Rebuild everything from the Earth Engine exports (keep the large CSVs on Google Drive, not GitHub):

```bash
python -m src.prepare_dataset "exports/suhi_features_*.csv"   # -> data/andhra_pradesh_suhi.csv
python -m src.train                                           # models, results, figures (~1 h)
python -m src.temporal_test                                   # unseen-year test
python -m src.predict my_points.csv --model RF                # batch predictions + SUHI labels
```

Optional: `pip install osmnx geopandas` and `python -m src.prepare_dataset ... --osm` adds `Amenity` (nearest
OpenStreetMap amenity) for a column-for-column match with the original Kaggle dataset.

## Deploy (free, public)

1. Upload this repository to GitHub (all files are under the 25 MB browser-upload limit).
2. https://share.streamlit.io → Create app → this repository, branch `main`, file `app.py`.
3. Advanced settings → Python 3.12 → Deploy.

If the saved models cannot be loaded on the server, the app retrains them automatically from `data/app_sample.csv`.

## Repository

```
├── app.py                          Streamlit web app
├── gee/extract_suhi_features.js    Google Earth Engine data extraction
├── src/
│   ├── prepare_dataset.py          merge + clean Earth Engine exports
│   ├── train.py                    experiments, figures, saved models
│   ├── temporal_test.py            unseen-year test
│   ├── config.py · data.py · models.py · suhi.py · predict.py
├── models/                         six trained models + metadata.json
├── data/app_sample.csv             complete scenes per city for the app map
└── results/                        metrics, cleaning report, figures
```

## Limitations

- Daytime only (Landsat passes ~10:30 local time).
- Sentinel-5P and ERA5 layers are coarse (km scale) and partly act as date/location signals.
- Amenity (OpenStreetMap) was not included in the trained models.
- Transfer to a new city works moderately (R² 0.24–0.68); retrain before applying elsewhere.

## License

MIT. Please cite Furuya et al. (2023) for the method.
