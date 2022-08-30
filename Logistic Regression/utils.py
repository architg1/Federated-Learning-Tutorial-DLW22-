from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression


# logistic regression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
        model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklearn LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # Network Intrusion data has 2 classes
    n_features = 10  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


# Data Processing

def categoriseFeatures(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['protocol_type'] = df['protocol_type'].cat.codes

    df['service'] = df['service'].astype('category')
    df['service'] = df['service'].cat.codes

    df['flag'] = df['flag'].astype('category')
    df['flag'] = df['flag'].cat.codes

    df['class'] = df['class'].astype('category')
    df['class'] = df['class'].cat.codes

    return df


def dataProcessing(df):
    df = categoriseFeatures(df)

    return df


def featureSelection(df):
    return correlationFeatures(df)


def correlationFeatures(df):
    x = df[[
        (df.corr()['class']).sort_values()[0:5].index[0],
        (df.corr()['class']).sort_values()[0:5].index[1],
        (df.corr()['class']).sort_values()[0:5].index[2],
        (df.corr()['class']).sort_values()[0:5].index[3],
        (df.corr()['class']).sort_values()[0:5].index[4],
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[1],
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[2],
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[3],
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[4],
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[5]
    ]]

    y = df[[
        (df.corr()['class']).sort_values(ascending=False)[0:6].index[0],
    ]]

    return [x, y]









