import os
import pickle
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim

problem_title = 'picture_reconstruction'

_NB_CHANNELS = 128*128


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
        
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

class SSI_Score(rw.score_types.BaseScoreType):
    """Structural Similarity Index. Measures the SSIM
     between the true and predicted images."""

    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='SSIM', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 128, 128)
        y_pred = y_pred.reshape(-1, 128, 128)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ssim_scores = [ssim(t, p, data_range=t.max()-t.min(), full=False, win_size=13) for t, p in zip(y_true, y_pred)]
        
        ssim_scores = [s for s in ssim_scores if not np.isnan(s)]
        return np.mean(ssim_scores)

workflow = rw.workflows.Regressor()
Predictions = rw.prediction_types.make_regression(list(range(_NB_CHANNELS)))
score_types = [
    Score(precision=4),
    SSI_Score(precision=4)
]


def _get_data(path="./data", split='train'):

    assert split in ['train', 'test'], 'split must be either train or test'

    ## Low resolution images
    data_x = np.load(os.path.join(path, f'X{split}.npy'))
    
    ## High resolution images
    data_y = np.load(os.path.join(path, f'Y{split}.npy'))
    
    data_x = data_x.astype(np.float32) / 255.0
    data_y = data_y.astype(np.float32) / 255.0
    _, H_y, W_y  = data_y.shape
    data_y = data_y.reshape(-1, H_y*W_y ) ## [N, 128*128], for the score function
    
    return data_x, data_y ## [N, 64, 64], [N, 128*128]


def get_train_data(path='./data/public'):
    return _get_data(path, split="train")


def get_test_data(path='./data/public'):
    return _get_data(path, split="test")


def get_cv(X, y):
    cv = KFold(n_splits=4, shuffle=True, random_state=2)
    return cv.split(X, y)
