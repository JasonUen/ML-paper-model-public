#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:50:51 2022

@author: tuen2
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from utils import get_data

random_search_controler = False # control whether to run RandomizedSearchCV
grid_search_controler = False # control whether to run GridSearchCV

#%%
df = get_data(transform=True)

# normalize data
scaler = MinMaxScaler()
scaler.fit(df)

data_trans = pd.DataFrame(scaler.transform(df), columns=df.columns)

# ML data
X_data = data_trans.loc[:,data_trans.columns!='FW']
y_data = data_trans['FW']

## hyperparameter tuning for RandomizedSearchCV
param_grid = {'n_estimators': [50, 100, 150], # 
              'max_features': ['sqrt', 'log2'],# 'sqrt', 'log2'
              'max_depth': [None, 10, 20, 30], # None, 10, 20, 30
              'criterion': ['squared_error', 'absolute_error', 'poisson'],
              'min_samples_leaf': [1, 10, 20],
              'min_samples_split': [2, 5, 10]
              }

#%% hyperparameter tuning
# Instantiate the regressor
rf = RandomForestRegressor(random_state=42)
if random_search_controler:
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=100, # samples from total grid combinations
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    random_search.fit(X_data, y_data)
    best_params = random_search.best_params_
    # Show top parameters
    print(f"Best parameters from RandomizedSearchCV: {best_params}")
# {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'criterion': 'poisson'}
#%%
## refined grid search for best parameters
if grid_search_controler:
    """The parameters below are refined based on the results from the RandomizedSearchCV above
    The commented parameters in brackets are the ones that were tested.
    """
    refined_grid = {
        'n_estimators': [160], # [155, 160, 165], [140, 150, 160], [125, 150, 175]
        'max_depth': [15], # [13, 15, 17], [10, 15, 20], [5,15,25], [15, None]
        'min_samples_split': [6], # [5,6,7], [2, 4, 6], [2,10,18]
        'min_samples_leaf': [1], # [1, 2, 3], [1,5,10]
        'max_features': ['sqrt'], # ['sqrt', 'log2']
        'criterion': ['poisson'] # ['squared_error', 'absolute_error', 'poisson']
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=refined_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    grid_search.fit(X_data, y_data)
    best_params_ = grid_search.best_params_
    df_results = pd.DataFrame(grid_search.cv_results_)
    print(f"Best parameters from GridSearchCV: {best_params_}")
# GridSearchCV: {'criterion': 'poisson', 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 160}
#%%
## feature selection
# tree importance
final_rf = RandomForestRegressor(
    n_estimators=160,
    max_depth=15,
    min_samples_split=6,
    min_samples_leaf=1,
    max_features='sqrt',
    criterion='poisson',
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X_data, y_data)

importances = final_rf.feature_importances_
feature_names = X_data.columns
tbi_df = pd.DataFrame({
    "Feature": feature_names,
    "Tree_based_importance": importances
}).sort_values("Tree_based_importance", ascending=False)
tbi_df.reset_index(drop=True, inplace=True)

# permutation importance
result = permutation_importance(
    estimator=final_rf,
    X=X_data,
    y=y_data,
    scoring="neg_mean_squared_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

pfi_df = pd.DataFrame({
    "Feature": X_data.columns,
    "PFI_mean": result.importances_mean,
    "PFI_std": result.importances_std
}).sort_values("PFI_mean", ascending=False)
pfi_df.reset_index(drop=True, inplace=True)
# compare results
df_importance_all = tbi_df.merge(pfi_df, on = "Feature", how="inner")
print(df_importance_all)

#%%
## plot results
# tree-based importance
df_tbi_vis = df_importance_all.sort_values("Tree_based_importance", ascending=False)
plt.rcParams.update({"font.size": 14}) 
plt.figure(figsize=(12, 7))
plt.barh(
    y=df_tbi_vis["Feature"],
    width=df_tbi_vis["Tree_based_importance"],
    color="skyblue",
    alpha=0.7,
    label="Tree-based Importance"
)
plt.title("Tree-based importance")
#plt.subplots_adjust(left=0.2, top=1, right=1)
plt.tight_layout()
plt.savefig(f'data_analysis/tree_based_FI_rf.png')

# PFI
df_pfi_vis = df_importance_all.sort_values("PFI_mean", ascending=False)
plt.figure(figsize=(12, 7))
plt.barh(
    y=df_pfi_vis["Feature"],
    width=df_pfi_vis["PFI_mean"],
    xerr=df_pfi_vis["PFI_std"],
    color="steelblue",
    alpha=0.8,
    ecolor="black",
    capsize=6,
    error_kw=dict(lw=1.7, capsize=5, capthick=2),
)
plt.xlabel("Permutation Importance (mean ± std)")
plt.tight_layout()
plt.savefig(f'data_analysis/PFI_rf.png')