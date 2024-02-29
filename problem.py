import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedGroupKFold

problem_title = 'picture_reconstruction'

_NB_CHANNELS = 1


class Score(rw.score_types.BaseScoreType):
    """Root-mean-square error. Measures the RMSE
     between the true and predicted values of all on channels."""

    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        
        on_y_true = np.array([t for y in y_true for t in y if t != 0])
        on_y_pred = np.array([p for y_hat, y in zip(y_pred, y_true) for p, t in zip(y_hat, y) if t != 0])

        if (on_y_pred < 0).any():
            return self.worst 

        return np.sqrt(np.mean(np.square(on_y_true - on_y_pred)))


workflow = rw.workflows.Regressor()
Predictions = rw.prediction_types.make_regression(list(range(_NB_CHANNELS)))
score_types = [
    Score(precision=4)
]


def _get_data(path="./picture_reconstruction_dataset", split='Train'):
    assert split in ['Train', 'Test'], 'split must be either Train or Test'
    path = os.path.join(path, split)
    photos_path = Path(path)
    file_list = os.listdir(photos_path)
    counter = 0
    data_x = []
    data_y = []
    for f in file_list:  # iterate through the files
        fpath = os.path.join(photos_path, f)
        fpath = fpath.replace('\\', '/')
        fpath = fpath.replace('._', '')
        if fpath.endswith('_hi.jpg'):
            # get the high resolution image
            img = mpimg.imread(fpath)
            data_x.append(img)
            # get the corresponding low resolution image
            fpath = fpath.replace('_hi.jpg', '_lo.jpg')
            fpath = fpath.replace('H', 'L')
            img = mpimg.imread(fpath)
            data_y.append(img)
        counter += 1
        if counter >= 25000:
            break
    return data_x, data_y


def get_train_data(path='./picture_reconstruction_dataset'):
    return _get_data(path, split="Train")


def get_test_data(path='./picture_reconstruction_dataset'):
    return _get_data(path, split="Test")


def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y)
