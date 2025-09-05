from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor

def make_pipeline(estimator):
    """
    Create a pipeline with transformation and fitting within inner CV
    """
    x_pipe = Pipeline([
        ("scale", MinMaxScaler()), # transform X
        ("model", estimator)
    ])
    return TransformedTargetRegressor(
        regressor=x_pipe,
        transformer=MinMaxScaler() # transform y
    )