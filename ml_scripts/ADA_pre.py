#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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
param_grid = {
    "estimator": [
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=4),
        DecisionTreeRegressor(max_depth=8),
    ],
    "n_estimators": [20, 50, 80],
    "learning_rate": [0.1, 0.5, 1],
    "loss": ["linear", "square", "exponential"],
}

#%% hyperparameter tuning
# Instantiate the algorithm
ada = AdaBoostRegressor(random_state=42)

## random search
if random_search_controler:
    random_search = RandomizedSearchCV(
        estimator=ada,
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
# RandomizedSearchCV: {'n_estimators': 80, 'loss': 'linear', 'learning_rate': 0.1, 'estimator': DecisionTreeRegressor(max_depth=8)}

## refined grid search for best parameters
if grid_search_controler:
    refined_grid = {
        "estimator": [
            DecisionTreeRegressor(max_depth=8),
        ], # [7,8,9], [6,8,10]
        "n_estimators": [81], # [80, 81, 82], [79, 80, 81], [75, 80, 85], [70, 80, 90], [60, 80, 100]
        "learning_rate": [0.1], # [0.08, 0.1, 1.02], [0.05, 0.1, 0.2],
        "loss": ["linear"], # ["linear", "square", "exponential"],
    }
    
    grid_search = GridSearchCV(
        estimator=ada,
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
# GridSearchCV: {'estimator': DecisionTreeRegressor(max_depth=8), 'learning_rate': 0.1, 'loss': 'linear', 'n_estimators': 81}
#%%
## feature selection
# tree importance
final_ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=8),
    n_estimators=81,
    learning_rate=0.1,
    loss="linear",
    random_state=42
)
final_ada.fit(X_data, y_data)
importances = final_ada.feature_importances_
feature_names = X_data.columns
tbi_df = pd.DataFrame({
    "Feature": feature_names,
    "Tree_based_importance": importances
}).sort_values("Tree_based_importance", ascending=False)
tbi_df.reset_index(drop=True, inplace=True)

# permutation importance
result = permutation_importance(
    estimator=final_ada,
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
plt.savefig(f'data_analysis/tree_based_FI_ada.png')

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
plt.savefig(f'data_analysis/PFI_ada.png')