import numpy as np, pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pipelines import make_pipeline

def run_nested_cv(X, y, build_estimator, param_grid,
                  n_outer=5, n_inner=5, n_iter=30, seed=42, verbose=1, y_inverse=None):
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=seed)

    oof_pred = np.full(len(y), np.nan, dtype=float) # out of fold predictions (leave-one-out)
    fold_metrics = [] # collect metrics for all folds
    best_params_list = []

    for train_ind, test_ind in outer.split(X, y):
        pipe = make_pipeline(build_estimator())
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter, cv=inner, n_jobs=-1, random_state=seed,
            scoring="neg_mean_squared_error", verbose=verbose
        )
        search.fit(X.iloc[train_ind], y.iloc[train_ind])
        best_params_list.append(search.best_params_)
        y_hat_bc = search.predict(X.iloc[test_ind]) # prediction results on BC scale
        y_true_bc = y.iloc[test_ind].to_numpy()

        # invert to original scale if provided
        if y_inverse is not None:
            y_hat = y_inverse(y_hat_bc)
            y_true = y_inverse(y_true_bc)
        else:
            y_hat, y_true = y_hat_bc, y_true_bc
        
        oof_pred[test_ind] = y_hat
        fold_metrics.append({
            "r2":   r2_score(y_true, y_hat),
            "rmse": np.sqrt(mean_squared_error(y_true, y_hat)),
            "mae":  mean_absolute_error(y_true, y_hat),
        })

    df_metrics = pd.DataFrame(fold_metrics)
    summary = {
        "mean": df_metrics.mean().to_dict(),
        "std":  df_metrics.std(ddof=1).to_dict()
    }
    return summary, oof_pred, df_metrics, best_params_list