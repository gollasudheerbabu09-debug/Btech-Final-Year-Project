"""The six algorithms. Each is wrapped in a Pipeline that also contains the
imputer, Min-Max scaler and one-hot encoder, so the SAME preprocessing fitted on
the training data is saved inside the model file and reused at prediction time.
"""
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE

MODEL_NAMES = {
    "LR": "Linear Regression",
    "KNN": "K-Nearest Neighbors (k=5)",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "MLP": "Multilayer Perceptron",
    "SVM": "Support Vector Machine (SVR)",
}


def get_estimators() -> dict:
    scaled = lambda est: TransformedTargetRegressor(regressor=est, transformer=StandardScaler())  # noqa: E731
    return {
        "LR": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DT": DecisionTreeRegressor(min_samples_leaf=10, random_state=RANDOM_STATE),
        "RF": RandomForestRegressor(n_estimators=100, min_samples_leaf=15, max_depth=25,
                                    max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE),
        "MLP": scaled(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                   random_state=RANDOM_STATE)),
        "SVM": scaled(SVR(kernel="rbf", C=3.0, epsilon=0.05, cache_size=1000)),
    }


def build_pipeline(estimator, numeric, categorical) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                              ("scale", MinMaxScaler())]), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.002,
                                                  max_categories=25, sparse_output=False), categorical))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", estimator)])
