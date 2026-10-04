"""Configuration – Andhra Pradesh SUHI dataset (5 cities, 2019–2024)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

CLEAN_DATA = DATA_DIR / "andhra_pradesh_suhi.csv"   # full cleaned table (large, not on GitHub)
APP_SAMPLE = DATA_DIR / "app_sample.csv"            # a few complete dates per city, for the app

RANDOM_STATE = 42

# ------------------------------------------------------------------ columns
TARGET = "LST"
COORDS = ["Longitude", "Latitude"]
LEAKY = ["UHI", "UTFVI"]                    # calculated from LST
IGNORE = ["geometry", "Zone", "city", "date"]
CATEGORICAL_CANDIDATES = ["lulc_classes", "LandUse", "Amenity", "season"]
GROUP = ["city", "date"]                    # one satellite scene of one city

# ------------------------------------------------------------------ feature sets
SPECTRAL = ["NDVI", "NDBI", "NDWI", "SAVI", "BU"]
SURFACE = ["soil_moisture", "Roughness", "Slope", "GHI"]
LAND_USE = ["lulc_classes", "LandUse", "Amenity"]
AIR = ["CH4", "CO", "HCHO", "NO2", "O3", "SO2"]
SEASON = ["season"]

FEATURE_SETS = {
    "Set 1": {"desc": "Spectral indices + season", "cols": SPECTRAL + SEASON},
    "Set 2": {"desc": "+ surface, terrain and climate", "cols": SPECTRAL + SEASON + SURFACE},
    "Set 3": {"desc": "+ land cover / land use", "cols": SPECTRAL + SEASON + SURFACE + LAND_USE},
    "Set 4": {"desc": "+ air quality (all variables)", "cols": SPECTRAL + SEASON + SURFACE + LAND_USE + AIR},
}

# ------------------------------------------------------------------ cleaning
NODATA = -9999
MIN_POINTS_PER_SCENE = 400   # drop mostly cloudy city-dates
ROBUST_Z = 4.0               # drop LST outliers (cloud edges) beyond 4 robust SDs of the scene
LST_RANGE = (10.0, 70.0)     # physically plausible surface temperature in Andhra Pradesh (°C)

# ------------------------------------------------------------------ sizes
MAX_ROWS = 200_000           # rows sampled for training + testing
ROW_CAPS = {"SVM": 15_000, "MLP": 60_000}   # SVR scales badly with rows; MLP gains little beyond 60k
CV_ROWS = 40_000
CV_FOLDS = 5
LOCO_ROWS = 60_000           # rows for the leave-one-city-out test
APP_DATES_PER_CITY = 4       # complete scenes per city saved for the app map
BLOCK_DEG = 0.01             # ~1 km spatial blocks
TEST_SIZE = 0.2
MAX_MODEL_MB = 24            # GitHub's browser upload limit is 25 MB per file

SUHI_K = 0.5                 # SUHI: LST > mean + 0.5 SD of the same city and date
