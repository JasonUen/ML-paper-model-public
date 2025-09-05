import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.RF import build_estimator as build_rf
from algorithms.SVR import build_estimator as build_svr
from algorithms.ADA import build_estimator as build_ada
from algorithms.GBR import build_estimator as build_gbr
from algorithms.DNN import build_estimator as build_dnn
from algorithms.Ridge import build_estimator as build_ridge
from algorithms.Lasso import build_estimator as build_lasso
from algorithms.MLR import build_estimator as build_mlr


from pipelines import make_pipeline
from utils import get_data
from sklearn.inspection import permutation_importance

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
df = get_data(transform=True)          # Box–Cox features/target
X = df.drop(columns="FW")
y = df["FW"]

model_dict = {
    "RF": build_rf,
    "SVR": build_svr,
    "ADA": build_ada,
    "GBR": build_gbr,
    "DNN": build_dnn,
    "Ridge": build_ridge,
    "Lasso": build_lasso,
    "MLR": build_mlr
}

# add optimizer map for DNN params
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, RMSprop
OPTIMIZER_MAP = {
    "<class 'keras.src.optimizers.adam.Adam'>": Adam,
    "<class 'keras.src.optimizers.nadam.Nadam'>": Nadam,
    "<class 'keras.src.optimizers.adamax.Adamax'>": Adamax,
    "<class 'keras.src.optimizers.rmsprop.RMSprop'>": RMSprop,
}


def get_optimal_params(optimal_params: pd.DataFrame) -> dict:
    params = optimal_params.mode().iloc[0].to_dict()
    # Cast numeric params that should be integers
    for k, v in list(params.items()):
        # fix integer-like floats
        if isinstance(v, float) and v.is_integer():
            params[k] = int(v)
        # fix optimizer strings for DNN
        if "optimizer" in k and isinstance(v,str) and v in OPTIMIZER_MAP:
            params[k] = OPTIMIZER_MAP[v]
    return params

for name, model_spec in model_dict.items():
    out = RES / name
    out.mkdir(parents=True, exist_ok=True)
    param_path = out / "best_params_per_fold.csv"
    if param_path.exists():
        params = get_optimal_params(pd.read_csv(param_path))
    else:
        params = {}

    pipe = make_pipeline(model_spec()).set_params(**params)
    pipe.fit(X, y)

    estimator = pipe.regressor_.named_steps["model"] # get model inside the pipeline

    # PFI analysis
    pfi = permutation_importance(pipe, X, y, scoring="neg_mean_squared_error",
                                  n_repeats=10, random_state=42, n_jobs=-1)
    df_pfi = pd.DataFrame({"feature": X.columns,
                           "pfi_mean": pfi.importances_mean,
                           "pfi_std":  pfi.importances_std}).sort_values("pfi_mean", ascending=False)
    df_pfi.to_csv(out / "pfi_results.csv", index=False)
    plt.figure(figsize=(10,6))
    plt.barh(df_pfi["feature"], df_pfi["pfi_mean"], xerr=df_pfi["pfi_std"], capsize=4)
    plt.gca().invert_yaxis()
    plt.title(f"{name.upper()}")
    plt.tight_layout()
    plt.savefig(out / f"{name}_pfi.png", dpi=200)

    # Tree-based FI
    if hasattr(estimator, "feature_importances_"):
        tbi = pd.DataFrame({"feature": X.columns,
                            "tbi": estimator.feature_importances_}).sort_values("tbi", ascending=False)
        tbi.to_csv(out / "fi_tree_results.csv", index=False)
        plt.figure(figsize=(10,6))
        plt.barh(tbi["feature"], tbi["tbi"])
        plt.gca().invert_yaxis()
        plt.title(f"{name.upper()} Tree-based FI")
        plt.tight_layout()
        plt.savefig(out / f"{name}_fi_tree.png", dpi=200)

    # Coef FI (linear models only)
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        cdf = pd.DataFrame({"feature": X.columns, "coef": coef}).sort_values("coef", key=np.abs, ascending=False)
        cdf.to_csv(out / "fi_coef_results.csv", index=False)
        plt.figure(figsize=(10,6))
        plt.barh(cdf["feature"], cdf["coef"])
        plt.gca().invert_yaxis(); plt.title(f"{name.upper()} Coef FI")
        plt.tight_layout()
        plt.savefig(out / f"{name}_fi_coef.png", dpi=200)

    print(f"Finished {name}")
