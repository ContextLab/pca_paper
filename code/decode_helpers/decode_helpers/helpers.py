# coding: utf-8
from __future__ import division
import numpy as np
import scipy.spatial.distance as sd
from scipy.special import gamma
from scipy.linalg import toeplitz
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


