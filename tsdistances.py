"""
Implementation of the Deep Temporal Clustering model
Time Series distances

@author Florent Forest (FlorentF9)
"""

import numpy as np
from statsmodels.tsa import stattools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    Reference: Batista, Wang & Keogh (2011). A Complexity-Invariant Distance Measure for Time Series. https://doi.org/10.1137/1.9781611972818.60
    """
    assert(len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y))
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    Computes a linearly weighted euclidean distance between the autocorrelation coefficients of the input time series.
    Reference: Galeano & Pena (2000). Multivariate Analysis in Vector Time Series.
    """
    assert (len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)
    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))
    return np.sum(d)
