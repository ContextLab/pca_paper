# coding: utf-8
from __future__ import division
import numpy as np
import scipy.spatial.distance as sd
from scipy.special import gamma
from scipy.linalg import toeplitz
from sklearn import linear_model
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist, cdist
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from scipy.optimize import minimize
from scipy.stats import ttest_1samp as ttest
import hypertools as hyp
import pandas as pd
import warnings
from matplotlib import pyplot as plt


def r2z(r):
    """
    Function that calculates the Fisher z-transformation

    Parameters
    ----------
    r : int or ndarray
        Correlation value

    Returns
    ----------
    result : int or ndarray
        Fishers z transformed correlation value

    """
    return 0.5*(np.log(1+r) - np.log(1-r))


def z2r(z):
    """
    Function that calculates the inverse Fisher z-transformation

    Parameters
    ----------
    z : int or ndarray
        Fishers z transformed correlation value

    Returns
    ----------
    result : int or ndarray
        Correlation value

    """
    r = np.divide((np.exp(2*z) - 1), (np.exp(2*z) + 1))
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = np.sign(r)[np.isinf(r)]
    return r


def mean_combine(vals):
    '''
    Compute the element-wise mean across each matrix in a list.

    :param vals: a matrix, or a list of matrices
    :return: a mean matrix
    '''
    if not (type(vals) == list):
        return vals
    else:
        return np.mean(np.stack(vals, axis=2), axis=2)

def corrmean_combine(corrs):
    '''
    Compute the mean element-wise correlation across each matrix in a list.

    :param corrs: a matrix of vectorized correlation matrices (output of mat2vec), or a list
                  of such matrices
    :return: a mean vectorized correlation matrix
    '''
    if not (type(corrs) == list):
        return corrs

    elif np.shape(corrs)[0] == 1:
        return corrs

    else:
        return z2r(np.mean(r2z(np.stack(corrs, axis=2)), axis=2))

def mean_combine(vals):
    '''
    Compute the element-wise mean across each matrix in a list.

    :param vals: a matrix, or a list of matrices
    :return: a mean matrix
    '''
    if not (type(vals) == list):
        return vals
    else:
        return np.mean(np.stack(vals, axis=2), axis=2)

def decoder(corrs):

    next_results_pd = pd.DataFrame({'rank': [0], 'accuracy': [0], 'error': [0]})
    for t in np.arange(corrs.shape[0]):
        decoded_inds = np.argmax(corrs[t, :])
        next_results_pd['error'] += np.mean(np.abs(decoded_inds - np.array(t))) / corrs.shape[0]
        next_results_pd['accuracy'] += np.mean(decoded_inds == np.array(t))
        next_results_pd['rank'] += np.mean(list(map((lambda x: int(x)), (corrs[t, :] <= corrs[t, t]))))

    next_results_pd['error'] =  next_results_pd['error'].values / corrs.shape[0]
    next_results_pd['accuracy'] = next_results_pd['accuracy'].values / corrs.shape[0]
    next_results_pd['rank']= next_results_pd['rank'].values / corrs.shape[0]

    return next_results_pd

def get_xval_assignments(ndata, nfolds):
    group_assignments = np.zeros(ndata)
    groupsize = int(np.ceil(ndata / nfolds))

    for i in range(1, nfolds):
        inds = np.arange(i * groupsize, np.min([(i + 1) * groupsize, ndata]))
        group_assignments[inds] = i
    np.random.shuffle(group_assignments)
    return group_assignments


class CorrelationClassifier(BaseEstimator):

    def __init__(self):
        self.feature_importances_ = None

    def fit(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.feature_importances_ = np.zeros(X.shape[1])

        full_corrs = 1 - pdist(X, metric='correlation')
        for i in range(X.shape[1]):
            next_X = np.delete(X.copy(), i, axis=1)
            next_corrs = 1 - pdist(next_X, metric='correlation')
            self.feature_importances_[i] = pearsonr(full_corrs, next_corrs)[0]

    def predict(self, X):
       corrs = 1 - cdist(X, self.X, metric='correlation')
       estimated_labels = []
       for i in range(X.shape[0]):
           estimated_labels.append(self.y[np.argmax(corrs[i, :])])

       return np.array(estimated_labels)


def decoding(in_data, out_data, n_features, **kwargs):

    y = np.array(range(0, in_data.shape[0]))
    estimator = CorrelationClassifier()

    selector = RFE(estimator, n_features, step=1)
    selector = selector.fit(in_data, y)

    predictions = selector.predict(out_data)

    acc = np.sum(y == predictions.round().astype(int))/in_data.shape[0]

    return acc

def find_features(data, dims=10):

    pca_data = np.asarray(hyp.reduce(list(data), ndims=dims))

    in_all_data = np.asarray(mean_combine([x for x in pca_data]))

    y = np.array(range(0, in_all_data.shape[0]))
    model = CorrelationClassifier()
    model.fit(in_all_data, y)
    importance = model.feature_importances_

    return importance

def pca_decoder_rfe(data, nfolds=2, dims=10):
    """
    :param data: a list of number-of-observations by number-of-features matrices
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """

    assert len(np.unique(
        list(map(lambda x: x.shape[0], data)))) == 1, 'all data matrices must have the same number of timepoints'
    assert len(np.unique(
        list(map(lambda x: x.shape[1], data)))) == 1, 'all data matrices must have the same number of features'


    pca_data = np.asarray(hyp.reduce(list(data), ndims=dims))

    group_assignments = get_xval_assignments(len(pca_data), nfolds)
    results_pd_full = pd.DataFrame()

    for i in range(0, nfolds):

        in_data = np.asarray([x for x in pca_data[group_assignments == i]])
        out_data = np.asarray([x for x in pca_data[group_assignments != i]])

        in_smooth = np.asarray(mean_combine([x for x in in_data]))
        out_smooth = np.asarray(mean_combine([x for x in out_data]))

        for d in reversed(range(2, dims + 1)):

            decode_elim = decoding(in_smooth, out_smooth, n_features=d)

            results_pd = pd.DataFrame({'accuracy': [decode_elim]})
            results_pd['dims'] = d
            results_pd['folds'] = i

            results_pd_full = pd.concat([results_pd_full, results_pd])


    return try_results_pd_full


def pca_decoder(data, nfolds=2, dims=10):
    """
    :param data: a list of number-of-observations by number-of-features matrices
    :param nfolds: number of cross-validation folds (train using out-of-fold data;
                   test using in-fold data)
    :return: results dictionary with the following keys:
       'rank': mean percentile rank (across all timepoints and folds) in the
               decoding distribution of the true timepoint
       'accuracy': mean percent accuracy (across all timepoints and folds)
       'error': mean estimation error (across all timepoints and folds) between
                the decoded and actual window numbers, expressed as a percentage
                of the total number of windows
    """

    assert len(np.unique(
        list(map(lambda x: x.shape[0], data)))) == 1, 'all data matrices must have the same number of timepoints'
    assert len(np.unique(
        list(map(lambda x: x.shape[1], data)))) == 1, 'all data matrices must have the same number of features'


    pca_data = np.asarray(hyp.reduce(list(data), ndims=dims))

    group_assignments = get_xval_assignments(len(pca_data), nfolds)
    results_pd = pd.DataFrame()

    for i in range(0, nfolds):
        for d in range(1, dims + 1):

            in_data = np.asarray([x for x in pca_data[group_assignments == i]])[:, :, :d]
            out_data = np.asarray([x for x in pca_data[group_assignments != i]])[:, :, :d]

            in_smooth = np.asarray(mean_combine([x for x in in_data]))
            out_smooth = np.asarray(mean_combine([x for x in out_data]))

            if d < 3:
                in_smooth = np.hstack((in_smooth, np.zeros((in_smooth.shape[0], 3 - in_smooth.shape[1]))))
                out_smooth = np.hstack((out_smooth, np.zeros((out_smooth.shape[0], 3 - out_smooth.shape[1]))))
            corrs = (1 - sd.cdist(in_smooth, out_smooth, 'correlation'))

            corrs = np.array(corrs)
            next_results_pd = decoder(corrs)
            next_results_pd['dims'] = d
            next_results_pd['folds'] = i

            results_pd = pd.concat([results_pd, next_results_pd])

    return results_pd

