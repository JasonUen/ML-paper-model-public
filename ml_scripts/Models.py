# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.special import inv_boxcox
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Nadam
import yaml
from utils import get_data
import time

# unit conversion
MetricTon_to_kt = 10**(-3)
start_time = time.time()
# %%
df = get_data(transform=True)

# normalize data
scaler = MinMaxScaler()
scaler.fit(df)
data_trans = pd.DataFrame(scaler.transform(df), columns=df.columns)

# read tuned parameters as a config
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "model_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# extract important target and input features
feature_lists = [
    config["MLR"]["selected_features"],
    config["Ridge"]["selected_features"],
    config["Lasso"]["selected_features"],
    config["SVR"]["selected_features"],
    config["ADA"]["selected_features"],
    config["GBR"]["selected_features"],
    config["RF"]["selected_features"],
    config["DNN"]["params"].get("selected_features", df.columns.drop("FW").tolist())
]

def make_dnn_model(params, input_dim):
    """ Create a KerasRegressor for the DNN model with specified parameters.
    params: dictionary containing model parameters
    input_dim: number of input features
    """
    from DNN_pre import build_model
    optimizer_map = {"Nadam": Nadam}
    return KerasRegressor(
        model=build_model,
        model__input_dim=input_dim,
        **{f"model__{k}": v for k, v in params.items() if k not in ["batch_size", "epochs", "optimizer"]},
        model__optimizer=optimizer_map[params["optimizer"]],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=0
    )

## create models
models = {
    "MLR": LinearRegression(),
    "Ridge": Ridge(**config["Ridge"]["params"]),
    "Lasso": Lasso(**config["Lasso"]["params"]),
    "SVR": SVR(**config["SVR"]["params"]),
    "ADA": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(**config["ADA"]["params"]["base_estimator"]["params"]),
        n_estimators=config["ADA"]["params"]["n_estimators"],
        learning_rate=config["ADA"]["params"]["learning_rate"],
        loss=config["ADA"]["params"]["loss"],
        random_state=42
    ),
    "GBR": GradientBoostingRegressor(**config["GBR"]["params"], random_state=42),
    "RF": RandomForestRegressor(**config["RF"]["params"], random_state=42, n_jobs=-1),
    "DNN": make_dnn_model(config["DNN"]["params"], input_dim=len(df.columns.drop("FW").tolist()))
}

#X_data = data_trans[GBR_SELECTED_FEATURES] # RF_SELECTED_FEATURES
y_data = data_trans["FW"]

# read transformation factor and normalization scaler
df_lambda = pd.read_csv("data_analysis/BoxCox_Lambda.csv", index_col=0)
fw_lambda = df_lambda.loc["FW", "Lambda"]
inv_scaler = MinMaxScaler()
inv_scaler.fit(df["FW"].values.reshape(-1, 1))

# repeated CV evaluation
def repeated_cv_predictions(model, X, y, n_splits=5, n_repeats=20):
    """
    model: models with tuned parameters
    X: input features
    y: target feature
    n_splits: number of splits for KFold
    n_repeats: number of repeats for cross-validation
    """
    r2_all = []
    rmse_all = []
    mae_all = []
    y_pred_repeats = []

    for i in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        y_pred = cross_val_predict(clone(model), X, y, cv=kf, n_jobs=-1) # use clone to ensure a clean, unfitted model for each fold

        ## get original prediction
        # apply inverse normalization
        y_pred_boxcox = inv_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_boxcox = inv_scaler.inverse_transform(y.values.reshape(-1, 1)).flatten()

        # apply inverse-boxcox transformation
        y_pred_original = inv_boxcox(y_pred_boxcox, fw_lambda)
        y_true_original = inv_boxcox(y_true_boxcox, fw_lambda)

        r2_all.append(r2_score(y_true_original, y_pred_original))
        rmse_all.append(np.sqrt(mean_squared_error(y_true_original, y_pred_original)))
        mae_all.append(mean_absolute_error(y_true_original, y_pred_original))
        y_pred_repeats.append(y_pred_original)

    return {"r2": r2_all, "rmse": rmse_all, "mae": mae_all, "y_pred_matrix": np.vstack(y_pred_repeats), "y_true_original": y_true_original}

# def plot residuals
def plot_abs_residual(model_name, y_true, y_pred):
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true*MetricTon_to_kt, y_true - y_pred, alpha=0.3)
    plt.xlabel("Actual FW (kt/y)")
    plt.ylabel("Residual")
    plt.title(f"{model_name}")
    plt.axhline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_abs_residuals.png')
    #plt.show()

# def plot residuals
def plot_relative_error(model_name, y_true, y_pred):
    epsilon = 1e-8
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true*MetricTon_to_kt, np.abs(y_true - y_pred)/ (y_true + epsilon), alpha=0.3)
    plt.xlabel("Actual FW (kt/y)")
    plt.ylabel("Relative Error")
    plt.title(f"{model_name}")
    plt.axhline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_rel_error.png')
    #plt.show()

results = []
for (model_name, model), features in zip(models.items(), feature_lists):
    print(f"Running {model_name}...")
    X_data = data_trans[features]
    performance = repeated_cv_predictions(
        model, X_data, y_data, n_splits=5,
    )
    results.append(
        {
            "Model": model_name,
            "Mean R2": np.mean(performance["r2"]),
            "Std R2": np.std(performance["r2"]),
            "Mean RMSE": np.mean(performance["rmse"]),
            "Std RMSE": np.std(performance["rmse"]),
            "Mean MAE": np.mean(performance["mae"]),
            "Std MAE": np.std(performance["mae"]),
        }
    )
    y_pred_mean = performance["y_pred_matrix"].mean(axis=0)
    plot_abs_residual(model_name, performance["y_true_original"], y_pred_mean)
    plot_relative_error(model_name, performance["y_true_original"], y_pred_mean)

results_df = pd.DataFrame(results)
results_df.to_csv("results/cv_results.csv")
print(results_df)

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")