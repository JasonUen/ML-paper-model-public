import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def build_estimator():
    return GradientBoostingRegressor(random_state=42)

param_grid = {
    "regressor__model__n_estimators": np.arange(100, 301, 100),
    "regressor__model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__model__subsample": [0.8, 1.0],
    "regressor__model__loss": ["squared_error", "absolute_error", "huber", "quantile"],

    # tree-related
    "regressor__model__max_depth": [3,5,7],
    "regressor__model__max_features": ["sqrt", "log2", None],
    "regressor__model__criterion": ["squared_error", "friedman_mse"],
    "regressor__model__min_samples_split": [2, 5, 10],
    "regressor__model__min_samples_leaf": [1, 5, 10],
}