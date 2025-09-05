import numpy as np
from sklearn.linear_model import Ridge

def build_estimator():
    return Ridge()

param_grid = {
    'regressor__model__alpha': np.logspace(-4, 2, 20)
}