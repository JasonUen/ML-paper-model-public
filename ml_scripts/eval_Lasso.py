import pandas as pd
import pathlib
from run_cv import run_nested_cv
from utils import get_data
from scipy.special import inv_boxcox
import importlib
name = "Lasso"
module = importlib.import_module(f"algorithms.{name}")
build_estimator = module.build_estimator
param_grid = module.param_grid

ROOT = pathlib.Path(__file__).resolve().parents[1] # Path to project root (one level up from this script)
RESULTS_DIR = ROOT / f"results/{name}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
lam_path = ROOT / "data_analysis" / "BoxCox_Lambda.csv"

df = get_data(transform=True)   # Box-Cox transformed
lamda = pd.read_csv(lam_path, index_col=0).loc["FW", "Lambda"] # for inverse_box-cox transformation
def inv_box_cox(fw_data):
    """
    Inverse Box-Cox transformation for FW data.
    """
    return inv_boxcox(fw_data, lamda)

X = df.drop(columns="FW")
y = df["FW"]

summary, oof, df_metrics, best_params_list = run_nested_cv(
    X, y,
    build_estimator=build_estimator,
    param_grid=param_grid,
    n_outer=10, n_inner=10, n_iter=20, # n_iter is the number of selected combinations from the param_grid
    seed=42, verbose=1, y_inverse=inv_box_cox
)

# export results
pd.DataFrame([{
    "model": name,
    "r2_mean": summary["mean"]["r2"], "r2_std": summary["std"]["r2"],
    "rmse_mean": summary["mean"]["rmse"], "rmse_std": summary["std"]["rmse"],
    "mae_mean": summary["mean"]["mae"], "mae_std": summary["std"]["mae"],
}]).to_csv(RESULTS_DIR / "metrics.csv", index=False)
df_metrics.assign(model=name).to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
pd.DataFrame({"y": inv_box_cox(y.values), "y_pred_oof": oof}).to_csv(RESULTS_DIR / "oof_prediction.csv", index=False)
print(f"{name} nested CV:", summary)
pd.DataFrame(best_params_list).to_csv(RESULTS_DIR / "best_params_per_fold.csv", index=False)