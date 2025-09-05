import numpy as np
from sklearn.ensemble import RandomForestRegressor

def build_estimator():
    return RandomForestRegressor(n_jobs=-1, random_state=42)

param_grid = {
    "regressor__model__n_estimators": np.arange(100, 201, 25),
    "regressor__model__max_depth": [None, 10, 12, 15, 18, 20],
    "regressor__model__min_samples_split": [2, 4, 6, 8, 10],
    "regressor__model__min_samples_leaf": [1, 2, 4, 8],
    "regressor__model__max_features": ["sqrt", "log2"],
    "regressor__model__criterion": ["squared_error", "absolute_error", "poisson"],
}