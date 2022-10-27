import logging
import numpy as np
from numba import njit
from scipy.stats import norm, multivariate_normal, poisson
import scipy.special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.mixture import GaussianMixture
from tqdm import trange
import copy


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



def compute_emission_probability_gaussian(X, rdr_mean, rdr_std, p_mean, p_std):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    log_mu : array, shape (n_states, n_spots)
        Log of read depth change due to CNV. Mean of NB distributions in HMM per state per spot.

    alphas : array, shape (n_states, n_spots)
        Over-dispersion of NB distributions in HMM per state per spot.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.

    p_mean : array, shape (n_states, n_spots)
        BAF due to CNV. Mean of Beta Binomial distribution in HMM per state per spot.

    p_std : array, shape (n_states, n_spots)
        Over-dispersion of Beta Binomial distribution in HMM per state per spot.
    
    Returns
    ----------
    log_emission : array, shape (2*n_states, n_obs, n_spots)
        Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = rdr_mean.shape[0]
    # initialize log_emission
    log_emission = np.zeros((2 * n_states, n_obs, n_spots))
    for i in np.arange(n_states):
        for s in np.arange(n_spots):
            # expression from Gaussian distribution
            if np.any(X[:,0,s] > 0):
                log_emission[i, :, s] = scipy.stats.norm.logpdf(X[:, 0, s], loc=rdr_mean[i,s], scale=rdr_std[i,s])
                log_emission[i + n_states, :, s] = log_emission[i, :, s]
            # BAF from Gaussian distribution
            if np.any(X[:,1,s] > 0):
                log_emission[i, :, s] += scipy.stats.norm.logpdf(X[:,1,s], loc=p_mean[i, s], scale=p_std[i,s])
                log_emission[i + n_states, :, s] += scipy.stats.norm.logpdf(X[:,1,s], loc=1-p_mean[i, s], scale=p_std[i,s])
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
        log_alpha: size 2n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
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
        # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities. 
        # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
        log_alpha[:, cumlen] = combined_log_startprob + np_sum_ax_squeeze(log_emission[:, cumlen, :], axis=1)
        for t in np.arange(1, le):
            phases_switch_mat = np.array([[log_sitewise_self_transmat[cumlen + t-1], log_sitewise_transmat[cumlen + t-1]], [log_sitewise_transmat[cumlen + t-1], log_sitewise_self_transmat[cumlen + t-1] ]])
            combined_transmat = np.kron( np.exp(phases_switch_mat), np.exp(log_transmat) )
            combined_transmat = np.log(combined_transmat)
            for j in np.arange(log_emission.shape[0]):
                for i in np.arange(log_emission.shape[0]):
                    buf[i] = log_alpha[i, (cumlen + t - 1)] + combined_transmat[i, j]
                log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(log_emission[j, (cumlen + t), :])
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
        log_beta: size 2*n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
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
        # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities. 
        # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
        log_beta[:, (cumlen + le - 1)] = 0
        for t in np.arange(le-2, -1, -1):
            phases_switch_mat = np.array([[log_sitewise_self_transmat[cumlen + t], log_sitewise_transmat[cumlen + t]], [log_sitewise_transmat[cumlen + t], log_sitewise_self_transmat[cumlen + t] ]])
            combined_transmat = np.kron( np.exp(phases_switch_mat), np.exp(log_transmat) )
            combined_transmat = np.log(combined_transmat)
            for i in np.arange(log_emission.shape[0]):
                for j in np.arange(log_emission.shape[0]):
                    buf[j] = log_beta[j, (cumlen + t + 1)] + combined_transmat[i, j] + np.sum(log_emission[j, (cumlen + t + 1), :])
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
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities. 
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = log_alpha[i, t] + log_transmat[i - n_states * int(i/n_states), j - n_states * int(j/n_states)] + np.sum(log_emission[j, t+1, :]) + log_beta[j, t+1]
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


def weighted_gaussian_fitting(x, weights):
    """
    x : 1d array.
    weights : 1d array
    """
    mu = weights.dot(x) / np.sum(weights)
    v = weights.dot( np.square(x - mu) ) / np.sum(weights)
    std = np.sqrt(v)
    return mu, std


def weighted_gaussian_fitting_sharestd(X, Weights):
    """
    X : array, (n_obs, n_cluster).
        Each cluster has an equal number of observations n_obs. Each cluster has its own mean, but the std is shared across clusters.
    Weights : array, (n_obs, n_cluster).
    """
    n_clusters = X.shape[1]
    mus = np.zeros(n_clusters)
    ssr = np.zeros(X.shape)
    for i in range(n_clusters):
        mus[i] = Weights[:,i].dot(X[:,i]) / np.sum(Weights[:,i])
        ssr[:,i] = np.square(X[:,i] - mus[i])
    v = Weights.flatten().dot(ssr.flatten()) / np.sum(Weights)
    stds = np.ones(n_clusters) * np.sqrt(v)
    return mus, stds


def update_emission_params_rdr_sitewise(X_rdr, log_gamma, rdr_std, \
    start_rdr_mean=None, shared_rdr_std=False):
    """
    Attributes
    ----------
    X_nb : array, shape (n_observations, n_spots)
        Observed expression UMI count UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    n_spots = X_rdr.shape[1]
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    new_rdr_mean = copy.copy(start_rdr_mean) if not start_rdr_mean is None else np.ones((n_states, n_spots))
    new_rdr_std = copy.copy(rdr_std)
    # expression signal by NB distribution
    if not shared_rdr_std:
        for s in range(n_spots):
            for i in range(n_states):
                mu, std = weighted_gaussian_fitting( X_rdr[:,s], gamma[i,:]+gamma[i+n_states,:] )
                new_rdr_mean[i, s] = mu
                new_rdr_std[i,s] = std
    else:
        for s in range(n_spots):
            mus, stds = weighted_gaussian_fitting_sharestd( np.vstack([ X_rdr[:,s] for i in range(n_states) ]).T, \
                (gamma[:n_states, :] + gamma[n_states:, :]).T )
            new_rdr_mean[:,s] = mus
            new_rdr_std[:,s] = stds
    return new_rdr_mean, new_rdr_std


def update_emission_params_baf_sitewise(X_baf, log_gamma, p_std, \
    start_p_mean=None, shared_p_std=False, min_binom_prob=0.01, max_binom_prob=0.99):
    """
    Attributes
    ----------
    X_baf : array, shape (n_observations, n_spots)
        Observed allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_spots = X_baf.shape[1]
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_p_mean = copy.copy(start_p_mean) if not start_p_mean is None else np.ones((n_states, n_spots)) * 0.5
    new_p_std = copy.copy(p_std)
    if not shared_p_std:
        for s in np.arange(X_baf.shape[1]):
            for i in range(n_states):
                mu, std = weighted_gaussian_fitting( np.append(X_baf[:,s], 1-X_baf[:,s]), np.append(gamma[i,:], gamma[i+n_states,:]) )
                new_p_mean[i, s] = mu
                new_p_std[i, s] = std
    else:
        for s in np.arange(X_baf.shape[1]):
            concat_X_baf = np.append(X_baf[:,s], 1-X_baf[:,s])
            concat_gamma = np.hstack([gamma[:n_states,:], gamma[n_states:, :]])
            mus, stds = weighted_gaussian_fitting_sharestd( np.vstack([ concat_X_baf for i in range(n_states) ]).T, concat_gamma.T)
            new_p_mean[:,s] = mus
            new_p_std[:,s] = stds
            new_p_mean[new_p_mean[:,s] < min_binom_prob, s] = min_binom_prob
            new_p_mean[new_p_mean[:,s] > max_binom_prob, s] = max_binom_prob
    return new_p_mean, new_p_std


############################################################
# whole inference
############################################################

class hmm_gaussian_sitewise(object):
    def __init__(self, params="stmp", t=1-1e-4):
        """
        Attributes
        ----------
        params : str
            Codes for parameters that need to be updated. The corresponding parameter can only be updated if it is included in this argument. "s" for start probability; "t" for transition probability; "m" for Negative Binomial RDR signal; "p" for Beta Binomial BAF signal.

        t : float
            Determine initial self transition probability to be 1-t.
        """
        self.params = params
        self.t = t
    #
    def run_baum_welch_nb_bb_sitewise(self, X, lengths, n_states, log_sitewise_transmat, \
        shared_rdr_std=False, shared_p_std=False, \
        is_diag=False, init_rdr_mean=None, init_p_mean=None, init_rdr_std=None, init_p_std=None, max_iter=100, tol=1e-4):
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
        rdr_mean = np.vstack([np.linspace(0.5, 3, n_states) for r in range(n_spots)]).T if init_rdr_mean is None else init_rdr_mean
        p_mean = np.vstack([np.linspace(0.05, 0.45, n_states) for r in range(n_spots)]).T if init_p_mean is None else init_p_mean
        # initialize (inverse of) dispersion param in NB and BetaBinom
        rdr_std = 0.5 * np.ones((n_states, n_spots)) if init_rdr_std is None else init_rdr_std
        p_std = 0.1 * np.ones((n_states, n_spots)) if init_p_std is None else init_p_std
        # initialize start probability and emission probability
        log_startprob = np.log( np.ones(n_states) / n_states )
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1-self.t) / (n_states-1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1,1))
        # EM algorithm
        for r in trange(max_iter):
            # E step
            log_emission = compute_emission_probability_gaussian(X, rdr_mean, rdr_std, p_mean, p_std)
            log_alpha = forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
            log_beta = backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
            log_gamma = compute_posterior_obs(log_alpha, log_beta)
            log_xi = compute_posterior_transition_sitewise(log_alpha, log_beta, log_transmat, log_emission)
            # M step
            if "s" in self.params:
                new_log_startprob = update_startprob_sitewise(lengths, log_gamma)
                new_log_startprob = new_log_startprob.flatten()
            else:
                new_log_startprob = log_startprob
            if "t" in self.params:
                new_log_transmat = update_transition_sitewise(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat
            if "m" in self.params:
                new_rdr_mean, new_rdr_std = update_emission_params_rdr_sitewise(X[:,0,:], log_gamma, rdr_std, start_rdr_mean=rdr_mean, shared_rdr_std=shared_rdr_std)
            else:
                new_rdr_mean = rdr_mean
                new_rdr_std = rdr_std
            if "p" in self.params:
                new_p_mean, new_p_std = update_emission_params_baf_sitewise(X[:,1,:], log_gamma, p_std, start_p_mean=p_mean, \
                    shared_p_std=shared_p_std)
            else:
                new_p_mean = p_mean
                new_p_std = p_std
            # check convergence
            print( np.mean(np.abs( np.exp(new_log_startprob) - np.exp(log_startprob) )), \
                np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )), \
                np.mean(np.abs(new_rdr_mean - rdr_mean)),\
                np.mean(np.abs(new_p_mean - p_mean)) )
            print( np.hstack([new_rdr_mean, new_p_mean]) )
            if np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )) < tol and \
                np.mean(np.abs(new_rdr_mean - rdr_mean)) < tol and np.mean(np.abs(new_p_mean - p_mean)) < tol:
                break
            log_startprob = new_log_startprob
            log_transmat = new_log_transmat
            rdr_mean = new_rdr_mean
            rdr_std = new_rdr_std
            p_mean = new_p_mean
            p_std = new_p_std
        return new_rdr_mean, new_rdr_std, new_p_mean, new_p_std, new_log_startprob, new_log_transmat


# def posterior_nb_bb_sitewise(X, lengths, rdr_mean, rdr_std, p_mean, p_std, log_startprob, log_transmat, log_sitewise_transmat):
#     """
#     Attributes
#     ----------
#     X : array, shape (n_observations, n_components, n_spots)
#         Observed expression UMI count and allele frequency UMI count.

#     lengths : array, shape (n_chromosomes,)
#         Number genes (or bins) per chromosome, the sum of this vector should be equal to n_observations.

#     base_nb_mean : array, shape (n_observations, n_spots)
#         Mean expression under diploid state.

#     log_mu : array, shape (n_states, n_spots)
#         Log read depth shift of each CNV states.

#     alphas : array, shape (n_states, n_spots)
#         Inverse of dispersion in NB distribution of each state.

#     total_bb_RD : array, shape (n_observations, n_spots)
#         SNP-covering reads for both REF and ALT across genes along genome.

#     p_mean : array, shape (n_states, n_spots)
#         MAF of each CNV states.

#     p_std : array, shape (n_states, n_spots)
#         Inverse of dispersion of Beta-Binomial distribution of each state.

#     log_startprob : array, shape (n_states,)
#         Log of start probability.

#     log_transmat : array, shape (n_states, n_states)
#         Log of transition probability across states.

#     log_sitewise_transmat : array, shape (n_observations)
#         Log of phase switch probability of each gene (or bin).
#     """
#     log_emission = compute_emission_probability_gaussian(X, rdr_mean, rdr_std, p_mean, p_std)
#     log_alpha = forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
#     log_beta = backward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
#     log_gamma = compute_posterior_obs(log_alpha, log_beta)
#     return log_gamma


# def loglikelihood_nb_bb_sitewise(X, lengths, rdr_mean, rdr_std, p_mean, p_std, log_startprob, log_transmat, log_sitewise_transmat):
#     """
#     Attributes
#     ----------
#     X : array, shape (n_observations, n_components, n_spots)
#         Observed expression UMI count and allele frequency UMI count.

#     lengths : array, shape (n_chromosomes,)
#         Number genes (or bins) per chromosome, the sum of this vector should be equal to n_observations.

#     base_nb_mean : array, shape (n_observations, n_spots)
#         Mean expression under diploid state.

#     log_mu : array, shape (n_states, n_spots)
#         Log read depth shift of each CNV states.

#     alphas : array, shape (n_states, n_spots)
#         Inverse of dispersion in NB distribution of each state.

#     total_bb_RD : array, shape (n_observations, n_spots)
#         SNP-covering reads for both REF and ALT across genes along genome.

#     p_mean : array, shape (n_states, n_spots)
#         MAF of each CNV states.

#     p_std : array, shape (n_states, n_spots)
#         Inverse of dispersion of Beta-Binomial distribution of each state.

#     log_startprob : array, shape (n_states,)
#         Log of start probability.

#     log_transmat : array, shape (n_states, n_states)
#         Log of transition probability across states.

#     log_sitewise_transmat : array, shape (n_observations)
#         Log of phase switch probability of each gene (or bin).
#     """
#     log_emission = compute_emission_probability_gaussian(X, rdr_mean, rdr_std, p_mean, p_std)
#     log_alpha = forward_lattice_sitewise(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
#     return np.sum(scipy.special.logsumexp(log_alpha[:,np.cumsum(lengths)-1], axis=0)), log_alpha


# def viterbi_nb_bb_sitewise(X, lengths, base_nb_mean, log_mu, alphas, total_bb_RD, p_mean, p_std, log_startprob, log_transmat, log_sitewise_transmat):
#     '''
#     Input
#         X: size n_observations * n_components * n_spots.
#         lengths: sum of lengths = n_observations.
#         exposures: size of n_observations * n_spots.
#         base_prob: size of n_observations. The expression probability derived from normal spots.
#         log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
#         alpha: size of n_states. Dispersioon parameter of each HMM state.
#         log_transmat: n_states * n_states. Transition probability after log transformation.
#         log_startprob: n_states. Start probability after log transformation.
#     Output
# #        log_prob: a scalar.
#         labels: size of n_observations.
#     Intermediate
#         log_emission: n_states * n_observations * n_spots. Log probability.
#         log_v: n_states * n_observations per chromosome. Log of viterbi DP table. v[i,t] = max_{q_1, ..., q_{t-1}} P(o_1, q_1, ..., o_{t-1}, q_{t-1}, o_t, q_t=i | lambda).
#     '''
#     n_obs = X.shape[0]
#     n_comp = X.shape[1]
#     n_spots = X.shape[2]
#     n_states = log_transmat.shape[0]
#     log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
#     log_emission = compute_emission_probability_gaussian(X, rdr_mean, rdr_std, p_mean, p_std)
#     # initialize viterbi DP table and backtracking table
#     labels = np.array([])
#     merged_labels = np.array([])
#     cumlen = 0
#     for le in lengths:
#         log_v = np.zeros((2*n_states, le))
#         bt = np.zeros((2*n_states, le))
#         for t in np.arange(le):
#             if cumlen == 0 and t == 0:
#                 log_v[:, 0] = np.mean(log_emission[:,0,:], axis=1) + np.append(log_startprob,log_startprob) + np.log(0.5)
#                 continue
#             for i in np.arange(2*n_states):
#                 if t > 0:
#                     tmp = log_v[:, (t-1)] + np.append(log_transmat[:,i - n_states * int(i/n_states)], log_transmat[:,i - n_states * int(i/n_states)]) + np.sum(log_emission[i, (cumlen+t), :])
#                 else:
#                     tmp = np.append(log_startprob[i - n_states * int(i/n_states)], log_startprob[i - n_states * int(i/n_states)]) + np.sum(log_emission[i, (cumlen+t), :])
#                 bt[i, t] = np.argmax(tmp)
#                 log_v[i, t] = np.max(tmp)
#         # backtracking to get the sequence
#         chr_labels = [ np.argmax(log_v[:,-1]) ]
        
#         if cumlen == 0:
#             for t2 in np.arange(le-1, 0, -1):
#                 chr_labels.append( int(bt[chr_labels[-1],t2]))
#         else:
#             for t2 in np.arange(le-2, -1, -1):
#                 chr_labels.append( int(bt[chr_labels[-1],t2]))

#         chr_labels = np.array(chr_labels[::-1]).astype(int)
#         # merge two phases
#         chr_merged_labels = copy.copy(chr_labels)
#         chr_merged_labels[chr_merged_labels >= n_states] = chr_merged_labels[chr_merged_labels >= n_states] - n_states
        
#         if cumlen == 0:
#             labels = chr_labels
#             merged_labels = chr_merged_labels
#         else:
#             labels = np.append(labels, chr_labels)
#             merged_labels = np.append(merged_labels, chr_merged_labels)
        
#         cumlen += le
#     return labels, merged_labels


from sklearn.mixture import GaussianMixture
def initialization_gaussianhmm_by_gmm(n_states, X, params, random_state=None, min_binom_prob=0.1, max_binom_prob=0.9):
    # prepare gmm input of RDR and BAF separately
    X_gmm_rdr = None
    X_gmm_baf = None
    if "m" in params:
        X_gmm_rdr = np.vstack([ X[:,0,s] for s in range(X.shape[2]) ]).T
    if "p" in params:
        X_gmm_baf = np.vstack([ X[:,1,s] for s in range(X.shape[2]) ]).T
        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob

    # combine RDR and BAF
    if ("m" in params) and ("p" in params):
        indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
        X_gmm = X_gmm_baf
    assert not np.any(np.isnan(X_gmm))
    # run GMM
    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=1).fit(X_gmm)
    else:
        gmm = GaussianMixture(n_components=n_states, max_iter=1, random_state=random_state).fit(X_gmm)
    # turn gmm fitted parameters to HMM rdr_mean and p_mean parameters
    if ("m" in params) and ("p" in params):
        gmm_rdr_mean = gmm.means_[:,:X.shape[2]]
        gmm_p_mean = gmm.means_[:, X.shape[2]:]
    elif "m" in params:
        gmm_rdr_mean = gmm.means_
        gmm_p_mean = None
    elif "p" in params:
        gmm_rdr_mean = None
        gmm_p_mean = gmm.means_
    return gmm_rdr_mean, gmm_p_mean


def pipeline_gaussian_baum_welch(X, lengths, n_states, log_sitewise_transmat, params="smp", t=1-1e-6, random_state=0, \
    shared_rdr_std=True, shared_p_std=True, init_rdr_mean=None, init_p_mean=None, init_rdr_std=None, init_p_std=None, \
    is_diag=True, max_iter=100, tol=1e-4):
    # initialization
    n_spots = X.shape[2]
    if ((init_rdr_mean is None) and ("m" in params)) or ((init_p_mean is None) and ("p" in params)):
        tmp_rdr_mean, tmp_p_mean = initialization_gaussianhmm_by_gmm(n_states, X, params, random_state=random_state)
        if (init_rdr_mean is None) and ("m" in params):
            init_rdr_mean = tmp_rdr_mean
        if (init_p_mean is None) and ("p" in params):
            init_p_mean = tmp_p_mean
    print(f"init_log_mu = {init_rdr_mean}")
    print(f"init_p_mean = {init_p_mean}")
    
    # fit HMM-NB-BetaBinom
    hmmmodel = hmm_gaussian_sitewise(params=params, t=t)
    new_rdr_mean, new_rdr_std, new_p_mean, new_p_std, new_log_startprob, new_log_transmat = hmmmodel.run_baum_welch_nb_bb_sitewise(X, lengths, \
        n_states, log_sitewise_transmat, shared_rdr_std=shared_rdr_std, shared_p_std=shared_p_std, is_diag=is_diag, \
        init_rdr_mean=init_rdr_mean, init_p_mean=init_p_mean, init_rdr_std=init_rdr_std, init_p_std=init_p_std, max_iter=max_iter, tol=tol)
    
    # likelihood, posterior and prediction
    log_emission = compute_emission_probability_gaussian(X, new_rdr_mean, new_rdr_std, new_p_mean, new_p_std)
    log_alpha = forward_lattice_sitewise(lengths, new_log_transmat, new_log_startprob, log_emission, log_sitewise_transmat)
    log_beta = backward_lattice_sitewise(lengths, new_log_transmat, new_log_startprob, log_emission, log_sitewise_transmat)
    log_gamma = compute_posterior_obs(log_alpha, log_beta)
    pred = np.argmax(log_gamma, axis=0)
    pred_cnv = pred % n_states
    llf = np.sum(scipy.special.logsumexp(log_alpha[:,np.cumsum(lengths)-1], axis=0))

    # save results
    res = {"new_rdr_mean":new_rdr_mean, "new_rdr_std":new_rdr_std, "new_p_mean":new_p_mean, "new_p_std":new_p_std, \
            "new_log_startprob":new_log_startprob, "new_log_transmat":new_log_transmat, "log_gamma":log_gamma, "pred_cnv":pred_cnv, "llf":llf}
    return res

