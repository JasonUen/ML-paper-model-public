import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def build_estimator():
    return AdaBoostRegressor(estimator = DecisionTreeRegressor(random_state=42), 
                             random_state=42)

param_grid = {
    # tune trees
    "regressor__model__estimator__max_depth": [1, 2, 4, 6, 8],
    "regressor__model__estimator__min_samples_leaf": [1, 2, 4, 8],
    "regressor__model__estimator__max_features": [None, "sqrt", "log2"],
    
    "regressor__model__n_estimators": [40, 60, 80, 100],
    "regressor__model__learning_rate": np.logspace(-2,0,5),
    "regressor__model__loss": ["linear", "square", "exponential"],
}