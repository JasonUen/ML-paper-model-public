from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from utils import get_data

random_search_controler = False # control whether to run RandomizedSearchCV
grid_search_controler = True # control whether to run GridSearchCV
#%%
df = get_data(transform=True)

# normalize data
scaler = MinMaxScaler()
scaler.fit(df)

data_trans = pd.DataFrame(scaler.transform(df), columns=df.columns)

# ML data
X_data = data_trans.loc[:,data_trans.columns!='FW']
y_data = data_trans['FW']

## hyperparameter tuning for random search
# the commented parameters are the ones that were tested
param_grid = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"], # precomputed
    # "degree": [3], #  for poly kernel only
    "gamma": ["scale", "auto"],
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 0.5, 1],
    # "coef0": [0], only for poly and sigmoid
    "tol": [1e-3, 1e-4],
}
#%% hyperparameter tuning
svr= SVR() # Instantiate the algorithm
## random search
if random_search_controler:
    random_search = RandomizedSearchCV(
        estimator=svr,
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
# RandomizedSearchCV: {'tol': 0.0001, 'kernel': 'rbf', 'gamma': 'auto', 'epsilon': 0.01, 'C': 10}

## refined grid search for best parameters
if grid_search_controler:
    refined_grid = {
        'kernel': ["rbf"], # ["linear", "poly", "rbf", "sigmoid"], 
        'C': [16], # [15, 16, 17], [14, 15, 16], [13, 15, 17], [5, 10, 15, 20], [0.1, 0.5, 1, 10],]
        'gamma': ["auto"], # ["scale", 'auto'], 
        'tol': [1e-4], 
        'epsilon': [0.02],  # [0.01, 0.02], [0.02, 0.03, 0.04], [0.01, 0.03], [0.01, 0.1, 0.5, 1], 
    }

    grid_search = GridSearchCV(
        estimator=svr,
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

#%%
## feature selection
# permutation importance
final_svr= SVR(
    kernel="rbf",
    C=16,
    gamma="auto",
    tol=1e-4,
    epsilon=0.02
)
final_svr.fit(X_data, y_data)


result = permutation_importance(
    estimator=final_svr,
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
print(pfi_df)


#%%
## plot results
# PFI
df_pfi_vis = pfi_df.sort_values("PFI_mean", ascending=False)
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
plt.savefig(f'data_analysis/PFI_svr.png')