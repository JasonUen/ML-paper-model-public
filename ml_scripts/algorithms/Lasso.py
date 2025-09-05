import numpy as np
from sklearn.linear_model import Lasso

def build_estimator():
    return Lasso(max_iter=10000)

param_grid = {
    'regressor__model__alpha': np.logspace(-4, 2, 20)
}