import logging
import numpy as np
from numba import njit
from scipy.stats import norm, multivariate_normal, poisson
import scipy.special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.mixture import GaussianMixture
from tqdm import trange
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import copy
from utils_distribution_fitting import *


############################################################
# E step related
############################################################

@njit
def np_max_ax_squeeze(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros(arr.shape[1])
        for i in range(len(result)):
            result[i] = np.max(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = np.max(arr[i, :])
    return result


@njit
def np_max_ax_keep(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros( (1, arr.shape[1]) )
        for i in range(result.shape[1]):
            result[:, i] = np.max(arr[:, i])
    else:
        result = np.zeros( (arr.shape[0], 1) )
        for i in range(result.shape[0]):
            result[i, :] = np.max(arr[i, :])
    return result


@njit
def np_sum_ax_squeeze(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros(arr.shape[1])
        for i in range(len(result)):
            result[i] = np.sum(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = np.sum(arr[i, :])
    return result


@njit
def np_sum_ax_keep(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros( (1, arr.shape[1]) )
        for i in range(result.shape[1]):
            result[:, i] = np.sum(arr[:, i])
    else:
        result = np.zeros( (arr.shape[0], 1) )
        for i in range(result.shape[0]):
            result[i, :] = np.sum(arr[i, :])
    return result


@njit
def np_mean_ax_squeeze(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros(arr.shape[1])
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result

@njit
def np_mean_ax_keep(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros( (1, arr.shape[1]) )
        for i in range(result.shape[1]):
            result[:, i] = np.mean(arr[:, i])
    else:
        result = np.zeros( (arr.shape[0], 1) )
        for i in range(result.shape[0]):
            result[i, :] = np.mean(arr[i, :])
    return result


@njit 
def mylogsumexp(a):
    # get max
    a_max = np.max(a)
    if (np.isinf(a_max)):
        return a_max
    # exponential
    tmp = np.exp(a - a_max)
    # summation
    s = np.sum(tmp)
    s = np.log(s)
    return s + a_max


@njit 
def mylogsumexp_ax_keep(a, axis):
    # get max
    a_max = np_max_ax_keep(a, axis=axis)
    # if a_max.ndim > 0:
    #     a_max[~np.isfinite(a_max)] = 0
    # elif not np.isfinite(a_max):
    #     a_max = 0
    # exponential
    tmp = np.exp(a - a_max)
    # summation
    s = np_sum_ax_keep(tmp, axis=axis)
    s = np.log(s)
    return s + a_max


def compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = log_mu.shape[0]
    # initialize log_emission
    log_emission = np.zeros((2 * n_states, n_obs, n_spots))
    for i in np.arange(n_states):
        for s in np.arange(n_spots):
            # expression from NB distribution
            nb_mean = base_nb_mean[:,s] * np.exp(log_mu[i])
            nb_std = np.sqrt(nb_mean + alphas[i] * nb_mean**2)
            n, p = convert_params(nb_mean, nb_std)
            log_emission[i, :, s] = scipy.stats.nbinom.logpmf(X[:, 0, s], n, p)
            log_emission[i + n_states, :, s] = log_emission[i, :, s]
            # AF from BetaBinom distribution
            idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
            log_emission[i, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], p_binom[i] * taus[i], (1-p_binom[i]) * taus[i])
            log_emission[i + n_states, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], (1-p_binom[i]) * taus[i], p_binom[i] * taus[i])
    return log_emission


@njit 
def forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
    '''
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_alpha: size n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
    '''
    n_obs = log_emission.shape[1]
    n_states = int(np.ceil(log_emission.shape[0] / 2))
    assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
    assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    # initialize log_alpha
    log_alpha = np.zeros((log_emission.shape[0], n_obs))
    buf = np.zeros(log_emission.shape[0])
    cumlen = 0
    for le in lengths:
        # start prob
        combined_log_startprob = np.log(0.5) + np.append(log_startprob,log_startprob)
        log_alpha[:, cumlen] = combined_log_startprob + np_mean_ax_squeeze(log_emission[:, cumlen, :], axis=1)
        for t in np.arange(1, le):
            phases_switch_mat = np.array([[log_sitewise_self_transmat[cumlen + t-1], log_sitewise_transmat[cumlen + t-1]], [log_sitewise_transmat[cumlen + t-1], log_sitewise_self_transmat[cumlen + t-1] ]])
            combined_transmat = np.kron( np.exp(phases_switch_mat), np.exp(log_transmat) )
            combined_transmat = np.log(combined_transmat)
            for j in np.arange(log_emission.shape[0]):
                for i in np.arange(log_emission.shape[0]):
                    buf[i] = log_alpha[i, (cumlen + t - 1)] + combined_transmat[i, j]
                log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.mean(log_emission[j, (cumlen + t), :])
        cumlen += le
    return log_alpha


@njit 
def old_forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
    '''
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_alpha: size n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
    '''
    n_obs = log_emission.shape[1]
    n_states = int(log_emission.shape[0] / 2)
    assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
    assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    # initialize log_alpha
    log_alpha = np.zeros((2 * n_states, n_obs))
    buf = np.zeros(2 * n_states)
    cumlen = 0
    for le in lengths:
        # start prob
        log_alpha[:, cumlen] = np.log(0.5) + np.append(log_startprob,log_startprob) + np_mean_ax_squeeze(log_emission[:, cumlen, :], axis=1)
        for t in np.arange(1, le):
            for j in np.arange(2*n_states):
                for i in np.arange(2*n_states):
                    buf[i] = log_alpha[i, (cumlen + t - 1)] + log_transmat[i - n_states * int(i/n_states), j - n_states * int(j/n_states)]
                    buf[i] += log_sitewise_self_transmat[cumlen + t-1] if (i < n_states) == (j < n_states) else log_sitewise_transmat[cumlen + t-1]
                log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.mean(log_emission[j, (cumlen + t), :])
        cumlen += le
    return log_alpha


@njit 
def backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
    '''
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        X: size n_observations * n_components * n_spots.
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_beta: size n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
    '''
    n_obs = log_emission.shape[1]
    n_states = int(np.ceil(log_emission.shape[0] / 2))
    assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
    assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    # initialize log_beta
    log_beta = np.zeros((log_emission.shape[0], n_obs))
    buf = np.zeros(log_emission.shape[0])
    cumlen = 0
    for le in lengths:
        # start prob
        log_beta[:, (cumlen + le - 1)] = 0
        for t in np.arange(le-2, -1, -1):
            phases_switch_mat = np.array([[log_sitewise_self_transmat[cumlen + t], log_sitewise_transmat[cumlen + t]], [log_sitewise_transmat[cumlen + t], log_sitewise_self_transmat[cumlen + t] ]])
            combined_transmat = np.kron( np.exp(phases_switch_mat), np.exp(log_transmat) )
            combined_transmat = np.log(combined_transmat)
            for i in np.arange(log_emission.shape[0]):
                for j in np.arange(log_emission.shape[0]):
                    buf[j] = log_beta[j, (cumlen + t + 1)] + combined_transmat[i, j] + np.mean(log_emission[j, (cumlen + t + 1), :])
                log_beta[i, (cumlen + t)] = mylogsumexp(buf)
        cumlen += le
    return log_beta


@njit 
def old_backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
    '''
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        X: size n_observations * n_components * n_spots.
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_beta: size n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
    '''
    n_obs = log_emission.shape[1]
    n_states = int(log_emission.shape[0] / 2)
    assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
    assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    # initialize log_beta
    log_beta = np.zeros((2 * n_states, n_obs))
    buf = np.zeros(2 * n_states)
    cumlen = 0
    for le in lengths:
        # start prob
        log_beta[:, (cumlen + le - 1)] = 0
        for t in np.arange(le-2, -1, -1):
            for i in np.arange(2*n_states):
                for j in np.arange(2*n_states):
                    buf[j] = log_beta[j, (cumlen + t + 1)] + log_transmat[i - n_states * int(i/n_states), j - n_states * int(j/n_states)] + np.mean(log_emission[j, (cumlen + t + 1), :])
                    buf[j] += log_sitewise_self_transmat[cumlen + t] if (i < n_states) == (j < n_states) else log_sitewise_transmat[cumlen + t]
                log_beta[i, (cumlen + t)] = mylogsumexp(buf)
        cumlen += le
    return log_beta


def compute_posterior_obs(log_alpha, log_beta):
    '''
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
    Output:
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda). gamma[i, t] propto alpha[i,t] * beta[i,t]
    '''
    n_states = log_alpha.shape[0]
    n_obs = log_alpha.shape[1]
    # initial log_gamma
    log_gamma = np.zeros((n_states, n_obs))
    # compute log_gamma
    # for j in np.arange(n_states):
    #     for t in np.arange(n_obs):
    #         log_gamma[j, t] = log_alpha[j, t] +  log_beta[j, t]
    log_gamma = log_alpha + log_beta
    if np.any( np.sum(log_gamma, axis=0) == 0 ):
        raise Exception("Sum of posterior probability is zero for some observations!")
    log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)
    return log_gamma


@njit
def compute_posterior_transition_sitewise(log_alpha, log_beta, log_transmat, log_emission):
    '''
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_emission: n_states * n_observations * n_spots. Log probability.
    Output:
        log_xi: size n_states * n_states * (n_observations-1). xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    '''
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]
    # initialize log_xi
    log_xi = np.zeros((2*n_states, 2*n_states, n_obs-1))
    # compute log_xi
    for i in np.arange(2*n_states):
        for j in np.arange(2*n_states):
            for t in np.arange(n_obs-1):
                log_xi[i, j, t] = log_alpha[i, t] + log_transmat[i - n_states * int(i/n_states), j - n_states * int(j/n_states)] + np.mean(log_emission[j, t+1, :]) + log_beta[j, t+1]
    # normalize
    for t in np.arange(n_obs-1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])
    return log_xi


############################################################
# M step related
############################################################
@njit
def update_startprob_sitewise(lengths, log_gamma):
    '''
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size 2 * n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after loog transformation.
    '''
    n_states = int(log_gamma.shape[0] / 2)
    n_obs = log_gamma.shape[1]
    assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the second dimension of log_gamma!"
    # indices of the start of sequences, given that the length of each sequence is in lengths
    cumlen = 0
    indices_start = []
    for le in lengths:
        indices_start.append(cumlen)
        cumlen += le
    indices_start = np.array(indices_start)
    # initialize log_startprob
    log_startprob = np.zeros(n_states)
    # compute log_startprob of 2 * n_states
    log_startprob = mylogsumexp_ax_keep(log_gamma[:, indices_start], axis=1)
    # merge (CNV state, phase A) and (CNV state, phase B)
    log_startprob = log_startprob.flatten().reshape(2,-1)
    log_startprob = mylogsumexp_ax_keep(log_startprob, axis=0)
    # normalize such that startprob sums to 1
    log_startprob -= mylogsumexp(log_startprob)
    return log_startprob


def update_transition_sitewise(log_xi, is_diag=False):
    '''
    Input
        log_xi: size (2*n_states) * (2*n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    '''
    n_states = int(log_xi.shape[0] / 2)
    n_obs = log_xi.shape[2]
    # initialize log_transmat
    log_transmat = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            log_transmat[i, j] = scipy.special.logsumexp( np.concatenate([log_xi[i, j, :], log_xi[i+n_states, j, :], \
                log_xi[i, j+n_states, :], log_xi[i + n_states, j + n_states, :]]) )
    # row normalize log_transmat
    if not is_diag:
        for i in np.arange(n_states):
            rowsum = scipy.special.logsumexp(log_transmat[i, :])
            log_transmat[i, :] -= rowsum
    else:
        diagsum = scipy.special.logsumexp(np.diag(log_transmat))
        totalsum = scipy.special.logsumexp(log_transmat)
        t = diagsum - totalsum
        rest = np.log( (1 - np.exp(t)) / (n_states-1) )
        log_transmat = np.ones(log_transmat.shape) * rest
        np.fill_diagonal(log_transmat, t)
    return log_transmat


def update_emission_params_nb_bb_sitewise(X, log_gamma, base_nb_mean, alphas, total_bb_RD, taus, \
    start_log_mu=None, start_p_binom=None, \
    fix_NB_dispersion=False, shared_NB_dispersion=False, fix_BB_dispersion=False, shared_BB_dispersion=False, \
    percent_threshold=0.99, min_binom_prob=0.01, max_binom_prob=0.99):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        log_mu = np.zeros(n_states)
        for i in range(n_states):
            model = sm.GLM(X[:,0, :].flatten(), np.ones(n_obs * n_spots).reshape(-1,1), \
                        family=sm.families.NegativeBinomial(alpha=alphas[i]), \
                        exposure=base_nb_mean.flatten(), var_weights=np.repeat(gamma[i,:]+gamma[i+n_states,:], n_spots))
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            log_mu[i] = res.params[0]
            if not (start_log_mu is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=start_log_mu, xtol=1e-4, ftol=1e-4)
                log_mu[i] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
    else:
        log_mu = np.zeros(n_states)
        alphas = np.zeros(n_states)
        if not shared_NB_dispersion:
            for i in range(n_states):
                model = Weighted_NegativeBinomial(X[:,0,:].flatten(), \
                            np.ones(n_obs * n_spots).reshape(-1,1), \
                            weights=np.repeat(gamma[i,:]+gamma[i+n_states,:], n_spots), exposure=base_nb_mean.flatten())
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                log_mu[i] = res.params[0]
                alphas[i] = res.params[-1]
                if not (start_log_mu is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_log_mu[i]], [alphas[i]]), xtol=1e-4, ftol=1e-4)
                    log_mu[i] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                    alphas[i] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            all_states_nb_mean = np.tile(base_nb_mean.flatten(), n_states)
            all_states_y = np.tile(X[:,0,:].flatten(), n_states)
            all_states_weights = np.concatenate([np.repeat(gamma[i,:]+gamma[i+n_states,:], n_spots) for i in range(n_states)])
            all_states_features = np.zeros((n_states*n_obs*n_spots, n_states))
            for i in np.arange(n_states):
                all_states_features[(i*n_obs*n_spots):((i+1)*n_obs*n_spots), i] = 1
            model = Weighted_NegativeBinomial(all_states_y, all_states_features, weights=all_states_weights, exposure=all_states_nb_mean)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            log_mu = res.params[:-1]
            alphas[:] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=np.append(start_log_mu, [alphas[0]]), xtol=1e-4, ftol=1e-4)
                log_mu = res.params[:-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[:-1]
                alphas[:] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
    # allele frequeency signal by BetaBinom distribution
    if fix_BB_dispersion:
        p_binom = np.zeros(n_states)
        for i in range(n_states):
            model = Weighted_BetaBinom_fixdispersion(np.append(X[:,1,:].flatten(), total_bb_RD.flatten()-X[:,1,:].flatten()), \
                np.ones(2*n_obs * n_spots).reshape(-1,1), \
                taus[i], \
                weights=np.append(np.repeat(gamma[i,:], n_spots), np.repeat(gamma[i+n_states,:], n_spots)), \
                exposure=np.append(total_bb_RD.flatten(),total_bb_RD.flatten()) )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            p_binom[i] = res.params[0]
            if not (start_p_binom is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=start_p_binom, xtol=1e-4, ftol=1e-4)
                p_binom[i] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
    else:
        p_binom = np.zeros(n_states)
        taus = np.zeros(n_states)
        if not shared_BB_dispersion:
            for i in range(n_states):
                model = Weighted_BetaBinom(np.append(X[:,1,:].flatten(), total_bb_RD.flatten()-X[:,1,:].flatten()), \
                    np.ones(2*n_obs * n_spots).reshape(-1,1), \
                    weights=np.append(np.repeat(gamma[i,:], n_spots), np.repeat(gamma[i+n_states,:], n_spots)), \
                    exposure=np.append(total_bb_RD.flatten(),total_bb_RD.flatten()) )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                p_binom[i] = res.params[0]
                taus[i] = res.params[-1]
                if not (start_p_binom is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_p_binom[i]], [taus[i]]), xtol=1e-4, ftol=1e-4)
                    p_binom[i] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                    taus[i] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            # all_states_exposure = np.tile( np.append(total_bb_RD.flatten(), total_bb_RD.flatten()), n_states)
            # all_states_y = np.tile( np.append(X[:,1,:].flatten(), total_bb_RD.flatten()-X[:,1,:].flatten()), n_states)
            # all_states_weights = np.concatenate([ np.append(gamma[i,:],gamma[i+n_states,:]) for i in range(n_states) ])
            # all_states_features = np.zeros((2*n_states*n_obs, n_states))
            # for i in np.arange(n_states):
            #     all_states_features[(i*2*n_obs):((i+1)*2*n_obs), i] = 1
            all_states_exposure = []
            all_states_y = []
            all_states_weights = []
            all_states_features = []
            for i in np.arange(2*n_states):
                idx_sort = np.argsort(gamma[i,:])[::-1]
                tmp = np.cumsum(gamma[i,idx_sort])
                idx_select = idx_sort[tmp <= tmp[-1] * percent_threshold]
                all_states_exposure.append( total_bb_RD.flatten()[idx_select] )
                all_states_y.append( X[:,1,:].flatten()[idx_select] if i < n_states else total_bb_RD.flatten()[idx_select]-X[:,1,:].flatten()[idx_select] )
                all_states_weights.append( gamma[i,idx_select] )
                tmp_features = np.zeros((len(idx_select), n_states))
                tmp_features[:, i % n_states] = 1
                all_states_features.append( tmp_features )
            all_states_exposure = np.concatenate(all_states_exposure)
            all_states_y = np.concatenate(all_states_y)
            all_states_weights = np.concatenate(all_states_weights)
            all_states_features = np.concatenate(all_states_features)
            model = Weighted_BetaBinom(all_states_y, all_states_features, weights=all_states_weights, exposure=all_states_exposure)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            p_binom = res.params[:-1]
            p_binom[p_binom < min_binom_prob] = min_binom_prob
            p_binom[p_binom > max_binom_prob] = max_binom_prob
            if res.params[-1] > 0:
                taus[:] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=np.append(start_p_binom, [taus[0]]), xtol=1e-4, ftol=1e-4)
                p_binom = res.params[:-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[:-1]
                p_binom[p_binom < min_binom_prob] = min_binom_prob
                p_binom[p_binom > max_binom_prob] = max_binom_prob
                if res2.params[-1] > 0:
                    taus[:] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
    return log_mu, alphas, p_binom, taus


############################################################
# whole inference
############################################################

def run_baum_welch_nb_bb_sitewise(X, lengths, n_states, base_nb_mean, total_bb_RD, log_sitewise_transmat, fix_NB_dispersion=False, shared_NB_dispersion=False, fix_BB_dispersion=False, shared_BB_dispersion=False, is_diag=False, init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None, max_iter=100, tol=1e-4):
    '''
    Input
        X: size n_observations * n_components * n_spots.
        lengths: sum of lengths = n_observations.
        base_nb_mean: size of n_observations * n_spots.
        In NB-BetaBinom model, n_components = 2
    Intermediate
        log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
        alpha: size of n_states. Dispersioon parameter of each HMM state.
    '''
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    assert n_comp == 2
    # initialize NB logmean shift and BetaBinom prob
    log_mu = np.linspace(-0.1, 0.1, n_states) if init_log_mu is None else init_log_mu
    p_binom = np.linspace(0.05, 0.45, n_states) if init_p_binom is None else init_p_binom
    # initialize (inverse of) dispersion param in NB and BetaBinom
    alphas = np.array([0.01] * n_states) if init_alphas is None else init_alphas
    taus = np.array([30] * n_states) if init_taus is None else init_taus
    # initialize start probability and emission probability
    log_startprob = np.log( np.ones(n_states) / n_states )
    t = 1 - 1e-4
    transmat = np.ones((n_states, n_states)) * (1-t) / (n_states-1)
    np.fill_diagonal(transmat, t)
    log_transmat = np.log(transmat)
    # EM algorithm
    for r in trange(max_iter):
        # E step
        log_emission = compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus)
        log_alpha = forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
        log_beta = backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
        log_gamma = compute_posterior_obs(log_alpha, log_beta)
        log_xi = compute_posterior_transition_sitewise(log_alpha, log_beta, log_transmat, log_emission)
        # M step
        new_log_startprob = update_startprob_sitewise(lengths, log_gamma)
        new_log_startprob = new_log_startprob.flatten()
        new_log_transmat = update_transition_sitewise(log_xi, is_diag=is_diag)
        # if r > 10:
        new_log_mu, new_alphas, new_p_binom, new_taus = update_emission_params_nb_bb_sitewise(X, log_gamma, base_nb_mean, alphas, total_bb_RD, taus, \
            start_log_mu=log_mu, start_p_binom=p_binom,
            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
            fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion)
        # else:
        #     new_log_mu, new_alphas, new_p_binom, new_taus = update_emission_params_nb_bb_sitewise(X, log_gamma, base_nb_mean, alphas, total_bb_RD, taus, \
        #         fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
        #         fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion)
        # check convergence
        print( np.mean(np.abs( np.exp(new_log_startprob) - np.exp(log_startprob) )), \
            np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )), \
            np.mean(np.abs(new_log_mu - log_mu)),\
            np.mean(np.abs(new_p_binom - p_binom)) < tol )
        print( np.vstack([new_log_mu, new_p_binom]).T )
        if np.mean(np.abs( np.exp(new_log_startprob) - np.exp(log_startprob) )) < tol and \
           np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )) < tol and \
           np.mean(np.abs(new_log_mu - log_mu)) < tol and np.mean(np.abs(new_p_binom - p_binom)) < tol:
           break
        log_startprob = new_log_startprob
        log_transmat = new_log_transmat
        log_mu = new_log_mu
        alphas = new_alphas
        p_binom = new_p_binom
        taus = new_taus
    return new_log_mu, new_alphas, new_p_binom, new_taus, new_log_startprob, new_log_transmat


def posterior_nb_bb_sitewise(X, lengths, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, log_startprob, log_transmat, log_sitewise_transmat):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    lengths : array, shape (n_chromosomes,)
        Number genes (or bins) per chromosome, the sum of this vector should be equal to n_observations.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    log_mu : array, shape (n_states,)
        Log read depth shift of each CNV states.

    alphas : array, shape (n_states,)
        Inverse of dispersion in NB distribution of each state.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.

    p_binom : array, shape (n_states,)
        MAF of each CNV states.

    taus : array, shape (n_states,)
        Inverse of dispersion of Beta-Binomial distribution of each state.

    log_startprob : array, shape (n_states,)
        Log of start probability.

    log_transmat : array, shape (n_states, n_states)
        Log of transition probability across states.

    log_sitewise_transmat : array, shape (n_observations)
        Log of phase switch probability of each gene (or bin).
    """
    log_emission = compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus)
    log_alpha = forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
    log_beta = backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
    log_gamma = compute_posterior_obs(log_alpha, log_beta)
    return log_gamma



def viterbi_nb_bb_sitewise(X, lengths, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, log_startprob, log_transmat, log_sitewise_transmat):
    '''
    Input
        X: size n_observations * n_components * n_spots.
        lengths: sum of lengths = n_observations.
        exposures: size of n_observations * n_spots.
        base_prob: size of n_observations. The expression probability derived from normal spots.
        log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
        alpha: size of n_states. Dispersioon parameter of each HMM state.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
    Output
#        log_prob: a scalar.
        labels: size of n_observations.
    Intermediate
        log_emission: n_states * n_observations * n_spots. Log probability.
        log_v: n_states * n_observations per chromosome. Log of viterbi DP table. v[i,t] = max_{q_1, ..., q_{t-1}} P(o_1, q_1, ..., o_{t-1}, q_{t-1}, o_t, q_t=i | lambda).
    '''
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = log_transmat.shape[0]
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    log_emission = compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus)
    # initialize viterbi DP table and backtracking table
    labels = np.array([])
    merged_labels = np.array([])
    cumlen = 0
    for le in lengths:
        log_v = np.zeros((2*n_states, le))
        bt = np.zeros((2*n_states, le))
        for t in np.arange(le):
            if cumlen == 0 and t == 0:
                log_v[:, 0] = np.mean(log_emission[:,0,:], axis=1) + np.append(log_startprob,log_startprob) + np.log(0.5)
                continue
            for i in np.arange(2*n_states):
                if t > 0:
                    tmp = log_v[:, (t-1)] + np.append(log_transmat[:,i - n_states * int(i/n_states)], log_transmat[:,i - n_states * int(i/n_states)]) + np.mean(log_emission[i, (cumlen+t), :])
                else:
                    tmp = np.append(log_startprob[i - n_states * int(i/n_states)], log_startprob[i - n_states * int(i/n_states)]) + np.mean(log_emission[i, (cumlen+t), :])
                bt[i, t] = np.argmax(tmp)
                log_v[i, t] = np.max(tmp)
        # backtracking to get the sequence
        chr_labels = [ np.argmax(log_v[:,-1]) ]
        
        if cumlen == 0:
            for t2 in np.arange(le-1, 0, -1):
                chr_labels.append( int(bt[chr_labels[-1],t2]))
        else:
            for t2 in np.arange(le-2, -1, -1):
                chr_labels.append( int(bt[chr_labels[-1],t2]))

        chr_labels = np.array(chr_labels[::-1]).astype(int)
        # merge two phases
        chr_merged_labels = copy.copy(chr_labels)
        chr_merged_labels[chr_merged_labels >= n_states] = chr_merged_labels[chr_merged_labels >= n_states] - n_states
        
        if cumlen == 0:
            labels = chr_labels
            merged_labels = chr_merged_labels
        else:
            labels = np.append(labels, chr_labels)
            merged_labels = np.append(merged_labels, chr_merged_labels)
        
        cumlen += le
    return labels, merged_labels
