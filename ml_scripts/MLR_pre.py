from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from utils import get_data

#%%
df = get_data(transform=True)

# normalize data
scaler = MinMaxScaler()
scaler.fit(df)

data_trans = pd.DataFrame(scaler.transform(df), columns=df.columns)

# ML data
drop_features = ['POP', 'LACS_POP', 'LACS_HHNV', 'LACS_SNAP', 'LACS_LOWI', 'SWS']
X_data_mlr =data_trans.loc[:, ~data_trans.columns.isin(['FW'] + drop_features)]
X_data = data_trans.loc[:, ~data_trans.columns.isin(['FW'])]
y_data = data_trans['FW']

# create models
mlr = LinearRegression()
ridge = Ridge()
lasso = Lasso(max_iter=10000)
#%%
## hyperparameter grid (only for ridge and lasso)
param_grid_ridge = {
    'alpha': np.logspace(-4, 2, 20)
}
ridge_gs = GridSearchCV(
    ridge,
    param_grid_ridge,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0
)
ridge_gs.fit(X_data, y_data)
best_ridge = ridge_gs.best_estimator_
print(f"Best Ridge: {ridge_gs.best_params_}")
# Best Ridge: {'alpha': np.float64(0.29763514416313164)}

param_grid_lasso = {
    'alpha': np.logspace(-4, 2, 20)
}
lasso_gs = GridSearchCV(
    lasso,
    param_grid_lasso,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0
)
lasso_gs.fit(X_data, y_data)
best_lasso = lasso_gs.best_estimator_
print(f"Best Lasso: {lasso_gs.best_params_}")
# Best Lasso: {'alpha': np.float64(0.00020691380811147902)}

#%%
## feature importance
# MLR
mlr.fit(X_data_mlr, y_data)
coef_df_mlr = pd.DataFrame({
    "Feature": X_data_mlr.columns,
    "MLR_Coefficient": mlr.coef_
}).sort_values("MLR_Coefficient", key=abs, ascending=False).reset_index(drop=True)

# ridge and lasso
coef_df = pd.DataFrame({
    "Feature": X_data.columns,
    "Ridge_Coefficient": best_ridge.coef_,
    "Lasso_Coefficient": best_lasso.coef_
}).sort_values("Ridge_Coefficient", key=abs, ascending=False)

## permutation feature importance 
# needed when input features are highly correlated, therefore only for ridge and lasso
final_ridge = Ridge(alpha=0.2976)
final_lasso = Lasso(alpha=0.0002, max_iter=10000)
pfi_models = {
    "ridge": final_ridge,
    "lasso": final_lasso,
}
pfi_results = []
for name, model in pfi_models.items():
    model.fit(X_data, y_data)
    result = permutation_importance(
        estimator=model,
        X=X_data,
        y=y_data,
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    temp_df = pd.DataFrame({
        "Feature": X_data.columns,
        f"PFI_mean_{name}": result.importances_mean,
        f"PFI_std_{name}": result.importances_std
    })
    pfi_results.append(temp_df)
pfi_df = pfi_results[0]
for res in pfi_results[1:]:
    pfi_df = pfi_df.merge(res, on="Feature")
pfi_df = pfi_df.sort_values(by="PFI_mean_ridge", ascending=False)
pfi_df.reset_index(drop=True, inplace=True)

df_FI_combined = pfi_df.merge(coef_df, on="Feature", how="inner")
df_FI_combined = df_FI_combined.merge(coef_df_mlr, on="Feature", how="left")
print(df_FI_combined)