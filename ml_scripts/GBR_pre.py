#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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

## hyperparameter tuning for random search
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "max_features": ["sqrt", "log2", None],
    "subsample": [0.8, 1.0],
    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
    "criterion": ["friedman_mse", "squared_error"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    # alpha # only used if loss="huber"
}
#%% hyperparameter tuning
# Instantiate the algorithm
gbr= GradientBoostingRegressor(random_state=42)

## random search
if random_search_controler:
    random_search = RandomizedSearchCV(
        estimator=gbr,
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
# {'subsample': 1.0, 'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_features': None, 
# 'max_depth': 3, 'loss': 'huber', 'learning_rate': 0.05, 'criterion': 'friedman_mse'}

## refined grid search for best parameters
if grid_search_controler:
    refined_grid = {
        "n_estimators": [85], # 80,81,82,83,84,85], [75, 80, 85], [85, 90, 95], [90, 95, 100], [85, 90, 95], [95, 100, 105], [75, 100, 125], [50, 100, 150],
        "learning_rate": [0.07], # [0.06, 0.07, 0.08] [0.03, 0.05, 0.07], [0.01, 0.05, 0.1],
        "max_depth": [3], # [3, None], [2, 3, 4], [1, 3, 5],
        "max_features": [ None], # ["sqrt", "log2", None],
        "subsample": [0.9], #  [0.85, 0.9, 0.95], [0.9, 1.0], [0.6, 0.8, 1.0],
        "loss": ["huber"], # ["squared_error", "absolute_error", "huber", "quantile"],
        "criterion": ["friedman_mse"], # ["friedman_mse", "squared_error"],
        "min_samples_split": [2], # [2, 3, 4], [2, 5, 8], [2, 10, 15],
        "min_samples_leaf": [4], #  [1, 2, 3, 4, 5], [4,6,8], [6, 7, 8, 9], [2, 10, 15],
        "alpha": [0.9], # [0.85, 0.9, 0.95], [0.7, 0.9, 1], only used if loss="huber"
    }

    grid_search = GridSearchCV(
        estimator=gbr,
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
# GridSearchCV: {'alpha': 0.9, 'criterion': 'friedman_mse', 'learning_rate': 0.07, 'loss': 'huber', 'max_depth': 3, 'max_features': None, 
# 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 85, 'subsample': 0.9}
#%%
## feature selection
# tree importance
final_gbr = GradientBoostingRegressor(
    n_estimators=85,
    learning_rate=0.07,
    max_depth=3,
    max_features=None,
    subsample=0.9,
    loss="huber",
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=4,
    alpha=0.9,
)
final_gbr.fit(X_data, y_data)
importances = final_gbr.feature_importances_
feature_names = X_data.columns
tbi_df = pd.DataFrame({
    "Feature": feature_names,
    "Tree_based_importance": importances
}).sort_values("Tree_based_importance", ascending=False)
tbi_df.reset_index(drop=True, inplace=True)

# permutation importance
result = permutation_importance(
    estimator=final_gbr,
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
plt.savefig(f'data_analysis/tree_based_FI_gbr.png')

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
plt.savefig(f'data_analysis/PFI_gbr.png')