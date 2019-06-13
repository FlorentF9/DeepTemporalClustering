"""
Implementation of the Deep Temporal Clustering model
Time Series distances

@author Florent Forest (FlorentF9)
"""

import numpy as np


def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    assert(len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0))
    ce_y = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0))
    assert((ce_x > 0.0).all() and (ce_y > 0.0).all())  # avoid division by zero
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.maximum(ce_x, ce_y) / np.maximum(ce_x, ce_y)
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    x_norm = (x - x.mean(axis=0)) / x.std(axis=0)
    y_norm = (y - y.mean(axis=0)) / y.std(axis=0)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    raise NotImplementedError
