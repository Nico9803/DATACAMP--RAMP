import os
import pickle

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedGroupKFold

problem_title = 'Image reconstruction'

_NB_CHANNELS = 1


class Score(rw.score_types.BaseScoreType):
    """Root-mean-square error. Measures the RMSE
     between the true and predicted values of all on channels."""

    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE'):
        self.name = name

    def __call__(self, y_true, y_pred):
        on_y_true = np.array([t for y in y_true for t in y if t != 0])
        on_y_pred = np.array([p for y_hat, y in zip(y_pred, y_true) for p, t in zip(y_hat, y) if t != 0])

        if (on_y_pred < 0).any():
            return self.worst

        return np.sqrt(np.mean(np.square(on_y_true - on_y_pred)))


workflow = rw.workflows.Regressor()
Predictions = rw.prediction_types.make_regression(list(range(_NB_CHANNELS)))
score_types = [
    Score()
]


def _get_data(path='.', split='train'):
    data_path = os.path.join(path, "data", split + ".csv")
    with open(os.path.join(data_path, 'X.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(data_path, 'y.pkl'), 'rb') as f:
        y = pickle.load(f)
    return X, y


def get_train_data(path='.'):
    return _get_data(path, split="train")


def get_test_data(path='.'):
    return _get_data(path, split="test")


def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y)
