# DNN model builder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from scipy.special import inv_boxcox
from utils import get_data

def build_model(hidden_units1=64, hidden_units2=64, hidden_units3=0, dropout_rate=0.2, learning_rate=0.001, 
                activation1='relu', activation2='relu', activation3='linear', activation_final='linear',
                k_init_1='glorot_uniform', k_init_2='glorot_uniform', k_init_3='glorot_uniform', optimizer="Adam", l2_reg=0, input_dim=None):
    """ Build a DNN model with specified parameters.
    hidden_units1: number of units in the first hidden layer
    hidden_units2: number of units in the second hidden layer
    hidden_units3: number of units in the third hidden layer (0 means no third layer)
    dropout_rate: dropout rate for regularization
    learning_rate: learning rate for the optimizer
    activation1, activation2, activation3, activation_final: activation functions for each layer
    k_init_1, k_init_2, k_init_3: kernel initializers for each layer
    l2_reg: L2 regularization factor
    input_dim: number of input features required for the first layer
    """
    if input_dim is None:
        raise ValueError("Please provide input_dim when building the DNN")
    if activation1 == "selu":
        k_init_1 = "lecun_normal"
        dropout_layer = AlphaDropout(dropout_rate)
    else:
        dropout_layer = Dropout(dropout_rate)
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(
        hidden_units1, 
        activation = activation1,
        kernel_initializer = k_init_1,
        kernel_regularizer=l2(l2_reg)
        )) # first Dense layer

    model.add(dropout_layer) # Regularization
    model.add(Dense(
        hidden_units2,
        activation=activation2,
        kernel_initializer=k_init_2,
        kernel_regularizer=l2(l2_reg)
    ))
    if hidden_units3 > 0:
        model.add(Dense(hidden_units3, activation=activation3, kernel_initializer=k_init_3, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1, activation=activation_final, kernel_regularizer=l2(l2_reg))) # softplus, relu, 
    optimizer = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__": # this block is added to allow other scripts to import this module without executing the code below
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

    # Wrap model with scikeras
    dnn = KerasRegressor(model=build_model, verbose=0, epochs=100, batch_size=32, input_dim = X_data.shape[1])

    # Hyperparameter grid
    """
    The parameters below are for RandomizedSearchCV below
    """
    param_grid = {
        "model__hidden_units1": [64, 128],
        "model__hidden_units2": [32],
        "model__dropout_rate": [0.2],
        "model__learning_rate": [0.0005, 0.001],
        "model__k_init_1": ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"], # "lecun_uniform"
        "model__k_init_2": ["glorot_uniform", "he_uniform"], # "lecun_uniform"
        "model__optimizer": [Adam, Nadam], # Adamax, RMSprop
        "model__activation1": ["relu", "elu"], # "selu", "tanh"
        "model__activation2": ["relu", "elu"], # "selu", "tanh"
        "model__activation3": ["linear", "softplus"], # "relu"
        "batch_size": [32],
        "epochs": [50]
    }

    #%%
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    # Randomized Search
    if random_search_controler:
        random_search = RandomizedSearchCV(
            estimator=dnn,
            param_distributions=param_grid,
            n_iter=30,
            cv=3,
            verbose=2,
            n_jobs=1,  # Keras does not always play nicely with n_jobs >1
            random_state=42,
            scoring=make_scorer(mean_squared_error, greater_is_better=False) 
        )
        random_search.fit(X_data, y_data)
        print("Best parameters from RandomizedSearchCV:")
        print(random_search.best_params_)
    # Best parameters from RandomizedSearchCV:
    # {'model__optimizer': <class 'keras.src.optimizers.nadam.Nadam'>, 'model__learning_rate': 0.001, 'model__k_init_2': 'he_uniform', 
    # 'model__k_init_1': 'glorot_normal', 'model__hidden_units2': 32, 'model__hidden_units1': 128, 'model__dropout_rate': 0.2, 
    # 'model__activation3': 'softplus', 'model__activation2': 'elu', 'model__activation1': 'elu', 'epochs': 50, 'batch_size': 32}

    #%%
    # Grid Search
    """
    The parameters below are refined based on the results from the RandomizedSearchCV above
    The commented parameters in brackets are the ones that were tested.
    """
    if grid_search_controler:
        refined_grid = {
            "model__hidden_units1": [124], # [118, 124, 130],[100, 112, 124], [112, 128, 144], [96, 128, 160],
            "model__hidden_units2": [40], # [32, 40, 48], [24, 32, 40],
            "model__hidden_units3": [0], # [0, 32],
            "model__dropout_rate": [0.25], # [0.25, 0.3], [0.15, 0.2, 0.25],
            "model__learning_rate": [0.001],  # 0.0005, 0.001], [0.001, 0.002],[0.002, 0.003, 0.004],  
            "model__k_init_1": ["glorot_uniform"], # ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"],
            "model__k_init_2": ["glorot_uniform"], # ["glorot_uniform", "he_uniform"],
            "model__k_init_3": ["he_uniform"], # ["glorot_uniform", "he_uniform"],
            "model__optimizer": [Nadam], # [Nadam, Adamax],  
            "model__activation1": ["relu"], # ["relu", "elu", "selu", "tanh"], 
            "model__activation2": ["tanh"], # ["relu", "elu", "selu", "tanh"],
            "model__activation3": ["softplus"], # ["linear", "softplus", "relu"],
            "model__l2_reg": [0.0001], # [0.0, 1e-4, 1e-3],
            "batch_size": [48], # [40, 44, 48], [32, 40, 48], [40, 48, 56], [48, 64, 80], [32, 64, 128],
            "epochs": [50] # [30,50,70]
        }
        grid_search = GridSearchCV(
            estimator=dnn,
            param_grid=refined_grid,
            cv=5,
            verbose=2,
            n_jobs=1,
            scoring='neg_mean_squared_error',
        )
        grid_search.fit(X_data, y_data)
        print("Best parameters from GridSearchCV:")
        print(grid_search.best_params_)
        results_df = pd.DataFrame(grid_search.cv_results_)
        print("Fit time (s):", results_df['mean_fit_time'].iloc[0])

    # Evaluate predict performance
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_data)

    # read transformation factor and normalization scaler
    df_lambda = pd.read_csv("data_analysis/BoxCox_Lambda.csv", index_col=0)
    fw_lambda = df_lambda.loc["FW", "Lambda"]
    inv_scaler = MinMaxScaler()
    inv_scaler.fit(df[["FW"]])

    # get original prediction
    y_pred_boxcox = inv_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true_boxcox = inv_scaler.inverse_transform(
        y_data.values.reshape(-1, 1)
    ).flatten()

    y_pred_original = inv_boxcox(y_pred_boxcox, fw_lambda)
    y_true_original = inv_boxcox(y_true_boxcox, fw_lambda)

    # Compute original scale for RMSE and MAE
    mse_original = mean_squared_error(y_true_original, y_pred_original)
    print("RMSE on original scale:", np.sqrt(mse_original))

    mae_original = mean_absolute_error(y_true_original, y_pred_original)
    print("MAE on original scale:", mae_original)

    # Best parameters from GridSearchCV: 
    # {'batch_size': 32, 'epochs': 50, 'model__activation1': 'elu', 'model__activation2': 'elu', 'model__activation3': 'softplus', 'model__dropout_rate': 0.25, 'model__hidden_units1': 124, 'model__hidden_units2': 40, 'model__hidden_units3': 0, 'model__k_init_1': 'glorot_normal', 'model__k_init_2': 'he_uniform', 'model__k_init_3': 'he_uniform', 'model__l2_reg': 0.0001, 'model__learning_rate': 0.001, 'model__optimizer': <class 'keras.src.optimizers.nadam.Nadam'>}
    # Fit time (s): 34.84431481361389
    # RMSE on original scale: 5504.326108346252
    # MAE on original scale: 966.6780786276453