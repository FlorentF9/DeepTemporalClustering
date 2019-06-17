"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
"""

import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder

ucr = UCR_UEA_datasets()
# try to use UCR/UEA univariate and multivariate datasets.
# requires forked version of tslearn from https://github.com/yichangwang/tslearn
try:
    all_ucr_datasets = ucr.list_datasets() + ucr._multivariate_dataset
except AttributeError:
    all_ucr_datasets = ucr.list_datasets()


def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y


def load_data(dataset_name):
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)
