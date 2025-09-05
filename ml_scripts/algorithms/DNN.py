from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from scikeras.wrappers import KerasRegressor

def build_model(hidden_units1=64, hidden_units2=64, hidden_units3=0, dropout_rate=0.2, learning_rate=0.001, 
                activation1='relu', activation2='relu', activation3='linear', activation_final='linear',
                k_init_1='glorot_uniform', k_init_2='glorot_uniform', k_init_3='glorot_uniform', 
                optimizer=Adam, l2_reg=0, input_dim=None):
    """ Build a DNN model with specified parameters.
    hidden_units1: number of units in the first hidden layer
    hidden_units2: number of units in the second hidden layer
    hidden_units3: number of units in the third hidden layer (0 means no third layer)
    dropout_rate: dropout rate for regularization
    learning_rate: learning rate for the optimizer
    activation1, activation2, activation3, activation_final: activation functions for each layer
    k_init_1, k_init_2, k_init_3: kernel initializers for each layer
    l2_reg: L2 regularization factor
    input_dim: Number of input features.
    """

    if input_dim is None:
        raise ValueError("Please provide input_dim when building the DNN")

    Drop = AlphaDropout if activation1 == "selu" else Dropout
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units1,
                    activation=activation1,
                    kernel_initializer=k_init_1,
                    kernel_regularizer=l2(l2_reg)))  # first Dense layer
    model.add(Drop(dropout_rate))  # Regularization
    model.add(Dense(hidden_units2,
                    activation=activation2,
                    kernel_initializer=k_init_2,
                    kernel_regularizer=l2(l2_reg)))
    if hidden_units3 > 0:
        model.add(Dense(hidden_units3, activation=activation3, kernel_initializer=k_init_3, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1, activation=activation_final, kernel_regularizer=l2(l2_reg))) # softplus, relu, 
    optimizer = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_estimator():
    # scikit‑learn compatible wrapper
    return KerasRegressor(model=build_model, verbose=0)

param_grid = {
    "regressor__model__model__hidden_units1": [64, 128],
    "regressor__model__model__hidden_units2": [32, 64],
    "regressor__model__model__hidden_units3": [0, 32],
    "regressor__model__model__dropout_rate": [0.1, 0.2, 0.3],
    "regressor__model__model__learning_rate": [5e-4, 1e-3],
    "regressor__model__model__l2_reg": [0.0, 1e-4, 1e-3],
    "regressor__model__model__activation1": ["relu", "elu"],
    "regressor__model__model__activation2": ["relu", "elu"],
    "regressor__model__model__activation3": ["linear", "softplus"],
    "regressor__model__model__activation_final": ["linear"],
    "regressor__model__model__k_init_1": ["glorot_uniform", "he_uniform"],
    "regressor__model__model__k_init_2": ["glorot_uniform", "he_uniform"],
    "regressor__model__model__k_init_3": ["glorot_uniform"],
    "regressor__model__model__optimizer": [Adam, Nadam],

    # training params live on the KerasRegressor itself:
    "regressor__model__batch_size": [32, 48, 64],
    "regressor__model__epochs": [50, 100],
}
