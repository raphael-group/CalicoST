import numpy as np
from numba import njit
import scipy.special
from tqdm import trange
from sklearn.mixture import GaussianMixture


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


def construct_unique_matrix(obs_count, total_count):
    """
    Attributes
    ----------
    allele_count : array, shape (n_observations, n_spots)
        Observed A allele counts per SNP per spot.
        
    total_bb_RD : array, shape (n_observations, n_spots)
        Total SNP-covering reads per SNP per spot.
    """
    n_obs = obs_count.shape[0]
    n_spots = obs_count.shape[1]
    unique_values = []
    mapping_matrices = []
    for s in range(n_spots):
        if total_count.dtype == int:
            pairs = np.unique( np.vstack([obs_count[:,s], total_count[:,s]]).T, axis=0 )
        else:
            pairs = np.unique( np.vstack([obs_count[:,s], total_count[:,s]]).T.round(decimals=4), axis=0 )
        unique_values.append( pairs )
        pair_index = {(pairs[i,0], pairs[i,1]):i for i in range(pairs.shape[0])}
        # construct mapping matrix
        mat_row = np.arange(n_obs)
        mat_col = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            if total_count.dtype == int:
                tmpidx = pair_index[(obs_count[i,s], total_count[i,s])]
            else:
                tmpidx = pair_index[(obs_count[i,s], total_count[i,s].round(decimals=4))]
            mat_col[i] = tmpidx
        mapping_matrices.append( scipy.sparse.csr_matrix((np.ones(len(mat_row)), (mat_row, mat_col) )) )
    return unique_values, mapping_matrices


def initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params, random_state=None, in_log_space=True, only_minor=True, min_binom_prob=0.1, max_binom_prob=0.9):
    # prepare gmm input of RDR and BAF separately
    X_gmm_rdr = None
    X_gmm_baf = None
    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack([ np.log(X[:,0,s]/base_nb_mean[:,s]) for s in range(X.shape[2]) ]).T
            offset = np.mean(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            normalizetomax1 = np.max(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]) - np.min(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
        else:
            X_gmm_rdr = np.vstack([ X[:,0,s]/base_nb_mean[:,s] for s in range(X.shape[2]) ]).T
            offset = 0
            normalizetomax1 = np.max(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
    if "p" in params:
        X_gmm_baf = np.vstack([ X[:,1,s] / total_bb_RD[:,s] for s in range(X.shape[2]) ]).T
        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob
    # combine RDR and BAF
    if ("m" in params) and ("p" in params):
        # indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        # X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        # indexes = np.where(X_gmm_baf[:,0] > 0.5)[0]
        # X_gmm_baf[indexes,:] = 1 - X_gmm_baf[indexes,:]
        X_gmm = X_gmm_baf
    # deal with NAN
    for k in range(X_gmm.shape[1]):
        last_idx_notna = -1
        for i in range(X_gmm.shape[0]):
            if last_idx_notna >= 0 and np.isnan(X_gmm[i, k]):
                X_gmm[i, k] = X_gmm[last_idx_notna, k]
            elif not np.isnan(X_gmm[i, k]):
                last_idx_notna = i
    X_gmm = X_gmm[np.sum(np.isnan(X_gmm), axis=1) == 0, :]
    # run GMM
    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=1).fit(X_gmm)
    else:
        gmm = GaussianMixture(n_components=n_states, max_iter=1, random_state=random_state).fit(X_gmm)
    # turn gmm fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = gmm.means_[:,:X.shape[2]] * normalizetomax1 + offset if in_log_space else np.log(gmm.means_[:,:X.shape[2]] * normalizetomax1 + offset)
        gmm_p_binom = gmm.means_[:, X.shape[2]:]
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1-gmm_p_binom, gmm_p_binom)
    elif "m" in params:
        gmm_log_mu = gmm.means_ * normalizetomax1 + offset if in_log_space else np.log(gmm.means_[:,:X.shape[2]] * normalizetomax1 + offset)
        gmm_p_binom = None
    elif "p" in params:
        gmm_log_mu = None
        gmm_p_binom = gmm.means_
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1-gmm_p_binom, gmm_p_binom)
    return gmm_log_mu, gmm_p_binom

