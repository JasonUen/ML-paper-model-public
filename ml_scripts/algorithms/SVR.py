import numpy as np
from sklearn.svm import SVR

def build_estimator():
    return SVR()

param_grid = {
    'regressor__model__kernel': ["linear","rbf", "sigmoid"], # ["linear", "poly", "rbf", "sigmoid"], 
    'regressor__model__C': [0.1, 1, 5, 10, 15, 20], # [15, 16, 17], [14, 15, 16], [13, 15, 17], [5, 10, 15, 20], [0.1, 0.5, 1, 10],]
    'regressor__model__gamma': ["scale", 'auto'], # ["scale", 'auto'], 
    'regressor__model__tol': [1e-4], 
    'regressor__model__epsilon': [0.01, 0.1, 0.5, 1],  # [0.01, 0.02], [0.02, 0.03, 0.04], [0.01, 0.03], [0.01, 0.1, 0.5, 1], 
}