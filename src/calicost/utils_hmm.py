import numpy as np
from numba import njit
import copy
import scipy.special
from numba import njit
from tqdm import trange
from sklearn.mixture import GaussianMixture
from calicost.utils_distribution_fitting import *

logger = logging.getLogger(__name__)

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
        result = np.zeros((1, arr.shape[1]))
        for i in range(result.shape[1]):
            result[:, i] = np.max(arr[:, i])
    else:
        result = np.zeros((arr.shape[0], 1))
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
        result = np.zeros((1, arr.shape[1]))
        for i in range(result.shape[1]):
            result[:, i] = np.sum(arr[:, i])
    else:
        result = np.zeros((arr.shape[0], 1))
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
        result = np.zeros((1, arr.shape[1]))
        for i in range(result.shape[1]):
            result[:, i] = np.mean(arr[:, i])
    else:
        result = np.zeros((arr.shape[0], 1))
        for i in range(result.shape[0]):
            result[i, :] = np.mean(arr[i, :])
    return result


@njit
def mylogsumexp(a):
    # get max
    a_max = np.max(a)
    if np.isinf(a_max):
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
            pairs = np.unique(np.vstack([obs_count[:, s], total_count[:, s]]).T, axis=0)
        else:
            pairs = np.unique(
                np.vstack([obs_count[:, s], total_count[:, s]]).T.round(decimals=4),
                axis=0,
            )
        unique_values.append(pairs)
        pair_index = {(pairs[i, 0], pairs[i, 1]): i for i in range(pairs.shape[0])}
        # construct mapping matrix
        mat_row = np.arange(n_obs)
        mat_col = np.zeros(n_obs, dtype=int)
        for i in range(n_obs):
            if total_count.dtype == int:
                tmpidx = pair_index[(obs_count[i, s], total_count[i, s])]
            else:
                tmpidx = pair_index[
                    (obs_count[i, s], total_count[i, s].round(decimals=4))
                ]
            mat_col[i] = tmpidx
        mapping_matrices.append(
            scipy.sparse.csr_matrix((np.ones(len(mat_row)), (mat_row, mat_col)))
        )
    return unique_values, mapping_matrices


def initialization_by_gmm(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    min_binom_prob=0.1,
    max_binom_prob=0.9,
):
    # prepare gmm input of RDR and BAF separately
    X_gmm_rdr = None
    X_gmm_baf = None
    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            offset = np.mean(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            ) - np.min(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            offset = 0
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            )
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T
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
        gmm = GaussianMixture(
            n_components=n_states, max_iter=1, random_state=random_state
        ).fit(X_gmm)
    # turn gmm fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = (
            gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = gmm.means_[:, X.shape[2] :]
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1 - gmm_p_binom, gmm_p_binom)
    elif "m" in params:
        gmm_log_mu = (
            gmm.means_ * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = None
    elif "p" in params:
        gmm_log_mu = None
        gmm_p_binom = gmm.means_
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1 - gmm_p_binom, gmm_p_binom)
    return gmm_log_mu, gmm_p_binom


############################################################
# E step related
############################################################


def compute_posterior_obs(log_alpha, log_beta):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
    Output:
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda). gamma[i, t] propto alpha[i,t] * beta[i,t]
    """
    n_states = log_alpha.shape[0]
    n_obs = log_alpha.shape[1]
    # initial log_gamma
    log_gamma = np.zeros((n_states, n_obs))
    # compute log_gamma
    # for j in np.arange(n_states):
    #     for t in np.arange(n_obs):
    #         log_gamma[j, t] = log_alpha[j, t] +  log_beta[j, t]
    log_gamma = log_alpha + log_beta
    if np.any(np.sum(log_gamma, axis=0) == 0):
        raise Exception("Sum of posterior probability is zero for some observations!")
    log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)
    return log_gamma


@njit
def compute_posterior_transition_sitewise(
    log_alpha, log_beta, log_transmat, log_emission
):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_emission: n_states * n_observations * n_spots. Log probability.
    Output:
        log_xi: size n_states * n_states * (n_observations-1). xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    """
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]
    # initialize log_xi
    log_xi = np.zeros((2 * n_states, 2 * n_states, n_obs - 1))
    # compute log_xi
    for i in np.arange(2 * n_states):
        for j in np.arange(2 * n_states):
            for t in np.arange(n_obs - 1):
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = (
                    log_alpha[i, t]
                    + log_transmat[
                        i - n_states * int(i / n_states),
                        j - n_states * int(j / n_states),
                    ]
                    + np.sum(log_emission[j, t + 1, :])
                    + log_beta[j, t + 1]
                )
    # normalize
    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])
    return log_xi


@njit
def compute_posterior_transition_nophasing(
    log_alpha, log_beta, log_transmat, log_emission
):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_emission: n_states * n_observations * n_spots. Log probability.
    Output:
        log_xi: size n_states * n_states * (n_observations-1). xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    """
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]
    # initialize log_xi
    log_xi = np.zeros((n_states, n_states, n_obs - 1))
    # compute log_xi
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            for t in np.arange(n_obs - 1):
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = (
                    log_alpha[i, t]
                    + log_transmat[i, j]
                    + np.sum(log_emission[j, t + 1, :])
                    + log_beta[j, t + 1]
                )
    # normalize
    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])
    return log_xi


############################################################
# M step related (HMM phasing)
############################################################


@njit
def update_startprob_sitewise(lengths, log_gamma):
    """
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size 2 * n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after loog transformation.
    """
    n_states = int(log_gamma.shape[0] / 2)
    n_obs = log_gamma.shape[1]
    assert (
        np.sum(lengths) == n_obs
    ), "Sum of lengths must be equal to the second dimension of log_gamma!"
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
    log_startprob = log_startprob.flatten().reshape(2, -1)
    log_startprob = mylogsumexp_ax_keep(log_startprob, axis=0)
    # normalize such that startprob sums to 1
    log_startprob -= mylogsumexp(log_startprob)
    return log_startprob


def update_transition_sitewise(log_xi, is_diag=False):
    """
    Input
        log_xi: size (2*n_states) * (2*n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    """
    n_states = int(log_xi.shape[0] / 2)
    n_obs = log_xi.shape[2]
    # initialize log_transmat
    log_transmat = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            log_transmat[i, j] = scipy.special.logsumexp(
                np.concatenate(
                    [
                        log_xi[i, j, :],
                        log_xi[i + n_states, j, :],
                        log_xi[i, j + n_states, :],
                        log_xi[i + n_states, j + n_states, :],
                    ]
                )
            )
    # row normalize log_transmat
    if not is_diag:
        for i in np.arange(n_states):
            rowsum = scipy.special.logsumexp(log_transmat[i, :])
            log_transmat[i, :] -= rowsum
    else:
        diagsum = scipy.special.logsumexp(np.diag(log_transmat))
        totalsum = scipy.special.logsumexp(log_transmat)
        t = diagsum - totalsum
        rest = np.log((1 - np.exp(t)) / (n_states - 1))
        log_transmat = np.ones(log_transmat.shape) * rest
        np.fill_diagonal(log_transmat, t)
    return log_transmat


def update_emission_params_nb_sitewise_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    base_nb_mean,
    alphas,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
    min_estep_weight=0.1,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    logger.info("Computing emission params for Negative Binomial (sitewise, unique).")

    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)

    # initialization
    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)

    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(
                        disp=0,
                        maxiter=1500,
                        start_params=np.array([start_log_mu[i, s]]),
                        xtol=1e-4,
                        ftol=1e-4,
                    )
                    new_log_mu[i, s] = (
                        res.params[0]
                        if -model.loglike(res.params) < -model.loglike(res2.params)
                        else res2.params[0]
                    )
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        penalty=0,
                    )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.append(
                                [start_log_mu[i, s]], [alphas[i, s]]
                            ),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_log_mu[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
                        new_alphas[i, s] = (
                            res.params[-1]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[-1]
                        )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate(
                    [
                        tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero]
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1])
                        >= min_estep_weight
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_NegativeBinomial(
                y, features, weights=weights, exposure=exposure
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:, :] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_log_mu[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * alphas[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:, :] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr

    logger.info("Computed emission params for Negative Binomial (sitewise, unique).")

    return new_log_mu, new_alphas


def update_emission_params_nb_sitewise_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    base_nb_mean,
    alphas,
    tumor_prop,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    logger.info("Computing emission params for Negative Binomial Mix (sitewise, unique).")

    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(
                        disp=0,
                        maxiter=1500,
                        start_params=np.array([start_log_mu[i, s]]),
                        xtol=1e-4,
                        ftol=1e-4,
                    )
                    new_log_mu[i, s] = (
                        res.params[0]
                        if -model.loglike(res.params) < -model.loglike(res2.params)
                        else res2.params[0]
                    )
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    model = Weighted_NegativeBinomial_mix(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s], penalty=0)
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.append(
                                [start_log_mu[i, s]], [alphas[i, s]]
                            ),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_log_mu[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
                        new_alphas[i, s] = (
                            res.params[-1]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[-1]
                        )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [
                        tmp[i, idx_nonzero] + tmp[i + n_states, idx_nonzero]
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])
                # tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_NegativeBinomial_mix(
                y,
                features,
                weights=weights,
                exposure=exposure,
                tumor_prop=tp,
                penalty=0,
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:, :] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_log_mu[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * alphas[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:, :] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr

    logger.info("Computed emission params for Negative Binomial Mix (sitewise, unique).")

    return new_log_mu, new_alphas


def update_emission_params_bb_sitewise_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    total_bb_RD,
    taus,
    start_p_binom=None,
    fix_BB_dispersion=False,
    shared_BB_dispersion=False,
    percent_threshold=0.99,
    min_binom_prob=0.01,
    max_binom_prob=0.99,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    logger.info("Computing emission params for Beta Binomial (sitewise, unique).")

    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        for s in np.arange(len(unique_values)):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if (
                    np.sum(tmp[i, idx_nonzero]) + np.sum(tmp[i + n_states, idx_nonzero])
                    >= 0.1
                ):
                    model = Weighted_BetaBinom_fixdispersion(
                        np.append(
                            unique_values[s][idx_nonzero, 0],
                            unique_values[s][idx_nonzero, 1]
                            - unique_values[s][idx_nonzero, 0],
                        ),
                        np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=np.append(
                            tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                        ),
                        exposure=np.append(
                            unique_values[s][idx_nonzero, 1],
                            unique_values[s][idx_nonzero, 1],
                        ),
                    )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.array(start_p_binom[i, s]),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_p_binom[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
    else:
        if not shared_BB_dispersion:
            for s in np.arange(len(unique_values)):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if (
                        np.sum(tmp[i, idx_nonzero])
                        + np.sum(tmp[i + n_states, idx_nonzero])
                        >= 0.1
                    ):
                        model = Weighted_BetaBinom(
                            np.append(
                                unique_values[s][idx_nonzero, 0],
                                unique_values[s][idx_nonzero, 1]
                                - unique_values[s][idx_nonzero, 0],
                            ),
                            np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                            weights=np.append(
                                tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                            ),
                            exposure=np.append(
                                unique_values[s][idx_nonzero, 1],
                                unique_values[s][idx_nonzero, 1],
                            ),
                        )
                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(
                                disp=0,
                                maxiter=1500,
                                start_params=np.append(
                                    [start_p_binom[i, s]], [taus[i, s]]
                                ),
                                xtol=1e-4,
                                ftol=1e-4,
                            )
                            new_p_binom[i, s] = (
                                res.params[0]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[0]
                            )
                            new_taus[i, s] = (
                                res.params[-1]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[-1]
                            )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in np.arange(len(unique_values)):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 1],
                        unique_values[s][idx_nonzero, 1],
                    ),
                    n_states,
                )
                this_y = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 0],
                        unique_values[s][idx_nonzero, 1]
                        - unique_values[s][idx_nonzero, 0],
                    ),
                    n_states,
                )
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate(
                    [
                        np.append(tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero])
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((2 * n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * 2 * len(idx_nonzero)) : ((i + 1) * 2 * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_BetaBinom(y, features, weights=weights, exposure=exposure)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:, :] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_p_binom[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * taus[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:, :] = res2.params[-1]
    new_p_binom[new_p_binom < min_binom_prob] = min_binom_prob
    new_p_binom[new_p_binom > max_binom_prob] = max_binom_prob

    logger.info("Computed emission params for Beta Binomial (sitewise, unique).")

    return new_p_binom, new_taus


def update_emission_params_bb_sitewise_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    total_bb_RD,
    taus,
    tumor_prop,
    start_p_binom=None,
    fix_BB_dispersion=False,
    shared_BB_dispersion=False,
    percent_threshold=0.99,
    min_binom_prob=0.01,
    max_binom_prob=0.99,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    logger.info("Computing emission params for Beta Binomial Mix (sitewise, unique).")

    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if (
                    np.sum(tmp[i, idx_nonzero]) + np.sum(tmp[i + n_states, idx_nonzero])
                    >= 0.1
                ):
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    assert np.all(this_tp < 1 + 1e-4)
                    model = Weighted_BetaBinom_fixdispersion_mix(
                        np.append(
                            unique_values[s][idx_nonzero, 0],
                            unique_values[s][idx_nonzero, 1]
                            - unique_values[s][idx_nonzero, 0],
                        ),
                        np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=np.append(
                            tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                        ),
                        exposure=np.append(
                            unique_values[s][idx_nonzero, 1],
                            unique_values[s][idx_nonzero, 1],
                        ),
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s] )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.array(start_p_binom[i, s]),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_p_binom[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
    else:
        if not shared_BB_dispersion:
            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if (
                        np.sum(tmp[i, idx_nonzero])
                        + np.sum(tmp[i + n_states, idx_nonzero])
                        >= 0.1
                    ):
                        this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                            idx_nonzero
                        ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                            idx_nonzero
                        ]
                        assert np.all(this_tp < 1 + 1e-4)
                        model = Weighted_BetaBinom_mix(
                            np.append(
                                unique_values[s][idx_nonzero, 0],
                                unique_values[s][idx_nonzero, 1]
                                - unique_values[s][idx_nonzero, 0],
                            ),
                            np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                            weights=np.append(
                                tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero]
                            ),
                            exposure=np.append(
                                unique_values[s][idx_nonzero, 1],
                                unique_values[s][idx_nonzero, 1],
                            ),
                            tumor_prop=this_tp,
                        )
                        # tumor_prop=tumor_prop )
                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(
                                disp=0,
                                maxiter=1500,
                                start_params=np.append(
                                    [start_p_binom[i, s]], [taus[i, s]]
                                ),
                                xtol=1e-4,
                                ftol=1e-4,
                            )
                            new_p_binom[i, s] = (
                                res.params[0]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[0]
                            )
                            new_taus[i, s] = (
                                res.params[-1]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[-1]
                            )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in np.arange(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 1],
                        unique_values[s][idx_nonzero, 1],
                    ),
                    n_states,
                )
                this_y = np.tile(
                    np.append(
                        unique_values[s][idx_nonzero, 0],
                        unique_values[s][idx_nonzero, 1]
                        - unique_values[s][idx_nonzero, 0],
                    ),
                    n_states,
                )
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [
                        np.append(tmp[i, idx_nonzero], tmp[i + n_states, idx_nonzero])
                        for i in range(n_states)
                    ]
                )
                this_features = np.zeros((2 * n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * 2 * len(idx_nonzero)) : ((i + 1) * 2 * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])
                # tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_BetaBinom_mix(
                y, features, weights=weights, exposure=exposure, tumor_prop=tp
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:, :] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_p_binom[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * taus[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:, :] = res2.params[-1]
    new_p_binom[new_p_binom < min_binom_prob] = min_binom_prob
    new_p_binom[new_p_binom > max_binom_prob] = max_binom_prob

    logger.info("Computed emission params for Beta Binomial Mix (sitewise, unique).")

    return new_p_binom, new_taus


############################################################
# M step related (no phasing)
############################################################
@njit
def update_startprob_nophasing(lengths, log_gamma):
    """
    Input
        lengths: sum of lengths = n_observations.
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda).
    Output
        log_startprob: n_states. Start probability after loog transformation.
    """
    n_states = log_gamma.shape[0]
    n_obs = log_gamma.shape[1]
    assert (
        np.sum(lengths) == n_obs
    ), "Sum of lengths must be equal to the second dimension of log_gamma!"
    # indices of the start of sequences, given that the length of each sequence is in lengths
    cumlen = 0
    indices_start = []
    for le in lengths:
        indices_start.append(cumlen)
        cumlen += le
    indices_start = np.array(indices_start)
    # initialize log_startprob
    log_startprob = np.zeros(n_states)
    # compute log_startprob of n_states
    log_startprob = mylogsumexp_ax_keep(log_gamma[:, indices_start], axis=1)
    # normalize such that startprob sums to 1
    log_startprob -= mylogsumexp(log_startprob)
    return log_startprob


def update_transition_nophasing(log_xi, is_diag=False):
    """
    Input
        log_xi: size (n_states) * (n_states) * n_observations. xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    Output
        log_transmat: n_states * n_states. Transition probability after log transformation.
    """
    n_states = log_xi.shape[0]
    n_obs = log_xi.shape[2]
    # initialize log_transmat
    log_transmat = np.zeros((n_states, n_states))
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            log_transmat[i, j] = scipy.special.logsumexp(log_xi[i, j, :])
    # row normalize log_transmat
    if not is_diag:
        for i in np.arange(n_states):
            rowsum = scipy.special.logsumexp(log_transmat[i, :])
            log_transmat[i, :] -= rowsum
    else:
        diagsum = scipy.special.logsumexp(np.diag(log_transmat))
        totalsum = scipy.special.logsumexp(log_transmat)
        t = diagsum - totalsum
        rest = np.log((1 - np.exp(t)) / (n_states - 1))
        log_transmat = np.ones(log_transmat.shape) * rest
        np.fill_diagonal(log_transmat, t)
    return log_transmat


def update_emission_params_nb_nophasing_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    alphas,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """

    logger.info("Computing emission params for Negative Binomial (no phasing, unique).")

    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero],
                )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(
                        disp=0,
                        maxiter=1500,
                        start_params=np.array([start_log_mu[i, s]]),
                        xtol=1e-4,
                        ftol=1e-4,
                    )
                    new_log_mu[i, s] = (
                        res.params[0]
                        if -model.loglike(res.params) < -model.loglike(res2.params)
                        else res2.params[0]
                    )
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        penalty=0,
                    )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.append(
                                [start_log_mu[i, s]], [alphas[i, s]]
                            ),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_log_mu[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
                        new_alphas[i, s] = (
                            res.params[-1]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[-1]
                        )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_NegativeBinomial(
                y, features, weights=weights, exposure=exposure
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:, :] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_log_mu[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * alphas[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:, :] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr

    logger.info("Computed emission params for Negative Binomial (no phasing, unique).")

    return new_log_mu, new_alphas


def update_emission_params_nb_nophasing_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    alphas,
    tumor_prop,
    start_log_mu=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=False,
    min_log_rdr=-2,
    max_log_rdr=2,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.
    """
    logger.info("Computing emission params for Negative Binomial Mix (no phasing, unique).")

    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_log_mu = (
        copy.copy(start_log_mu)
        if not start_log_mu is None
        else np.zeros((n_states, n_spots))
    )
    new_alphas = copy.copy(alphas)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(
                    unique_values[s][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=alphas[i, s]),
                    exposure=unique_values[s][idx_nonzero, 1],
                    var_weights=tmp[i, idx_nonzero],
                )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(
                        disp=0,
                        maxiter=1500,
                        start_params=np.array([start_log_mu[i, s]]),
                        xtol=1e-4,
                        ftol=1e-4,
                    )
                    new_log_mu[i, s] = (
                        res.params[0]
                        if -model.loglike(res.params) < -model.loglike(res2.params)
                        else res2.params[0]
                    )
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    model = Weighted_NegativeBinomial_mix(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s], penalty=0)
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.append(
                                [start_log_mu[i, s]], [alphas[i, s]]
                            ),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_log_mu[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
                        new_alphas[i, s] = (
                            res.params[-1]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[-1]
                        )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])
                # tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_NegativeBinomial_mix(
                y,
                features,
                weights=weights,
                exposure=exposure,
                tumor_prop=tp,
                penalty=0,
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:, :] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_log_mu[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * alphas[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:, :] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr

    logger.info("Computed emission params for Negative Binomial Mix (no phasing, unique).")

    return new_log_mu, new_alphas


def update_emission_params_bb_nophasing_uniqvalues(
    unique_values,
    mapping_matrices,
    log_gamma,
    taus,
    start_p_binom=None,
    fix_BB_dispersion=False,
    shared_BB_dispersion=False,
    percent_threshold=0.99,
    min_binom_prob=0.01,
    max_binom_prob=0.99,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    logger.info("Computing emission params for Beta Binomial (no phasing, unique).")

    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        for s in np.arange(len(unique_values)):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                    model = Weighted_BetaBinom_fixdispersion(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                    )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.array(start_p_binom[i, s]),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_p_binom[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
    else:
        if not shared_BB_dispersion:
            for s in np.arange(len(unique_values)):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                        model = Weighted_BetaBinom(
                            unique_values[s][idx_nonzero, 0],
                            np.ones(len(idx_nonzero)).reshape(-1, 1),
                            weights=tmp[i, idx_nonzero],
                            exposure=unique_values[s][idx_nonzero, 1],
                        )
                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(
                                disp=0,
                                maxiter=1500,
                                start_params=np.append(
                                    [start_p_binom[i, s]], [taus[i, s]]
                                ),
                                xtol=1e-4,
                                ftol=1e-4,
                            )
                            new_p_binom[i, s] = (
                                res.params[0]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[0]
                            )
                            new_taus[i, s] = (
                                res.params[-1]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[-1]
                            )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            for s in np.arange(len(unique_values)):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            model = Weighted_BetaBinom(y, features, weights=weights, exposure=exposure)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:, :] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_p_binom[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * taus[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:, :] = res2.params[-1]

    new_p_binom[new_p_binom < min_binom_prob] = min_binom_prob
    new_p_binom[new_p_binom > max_binom_prob] = max_binom_prob

    logger.info("Computed emission params for Beta Binomial (no phasing, unique).")

    return new_p_binom, new_taus


def update_emission_params_bb_nophasing_uniqvalues_mix(
    unique_values,
    mapping_matrices,
    log_gamma,
    taus,
    tumor_prop,
    start_p_binom=None,
    fix_BB_dispersion=False,
    shared_BB_dispersion=False,
    percent_threshold=0.99,
    min_binom_prob=0.01,
    max_binom_prob=0.99,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    log_gamma : array, (n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    logger.info("Computing emission params for Beta Binomial Mix (no phasing, unique).")

    n_spots = len(unique_values)
    n_states = log_gamma.shape[0]
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = (
        copy.copy(start_p_binom)
        if not start_p_binom is None
        else np.ones((n_states, n_spots)) * 0.5
    )
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                    this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                        idx_nonzero
                    ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ]
                    assert np.all(this_tp < 1 + 1e-4)
                    model = Weighted_BetaBinom_fixdispersion_mix(
                        unique_values[s][idx_nonzero, 0],
                        np.ones(len(idx_nonzero)).reshape(-1, 1),
                        taus[i, s],
                        weights=tmp[i, idx_nonzero],
                        exposure=unique_values[s][idx_nonzero, 1],
                        tumor_prop=this_tp,
                    )
                    # tumor_prop=tumor_prop[s] )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(
                            disp=0,
                            maxiter=1500,
                            start_params=np.array(start_p_binom[i, s]),
                            xtol=1e-4,
                            ftol=1e-4,
                        )
                        new_p_binom[i, s] = (
                            res.params[0]
                            if model.nloglikeobs(res.params)
                            < model.nloglikeobs(res2.params)
                            else res2.params[0]
                        )
    else:
        if not shared_BB_dispersion:
            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if np.sum(tmp[i, idx_nonzero]) >= 0.1:
                        this_tp = (mapping_matrices[s].T @ tumor_prop[:, s])[
                            idx_nonzero
                        ] / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                            idx_nonzero
                        ]
                        assert np.all(this_tp < 1 + 1e-4)
                        model = Weighted_BetaBinom_mix(
                            unique_values[s][idx_nonzero, 0],
                            np.ones(len(idx_nonzero)).reshape(-1, 1),
                            weights=tmp[i, idx_nonzero],
                            exposure=unique_values[s][idx_nonzero, 1],
                            tumor_prop=this_tp,
                        )
                        # tumor_prop=tumor_prop[s] )
                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(
                                disp=0,
                                maxiter=1500,
                                start_params=np.append(
                                    [start_p_binom[i, s]], [taus[i, s]]
                                ),
                                xtol=1e-4,
                                ftol=1e-4,
                            )
                            new_p_binom[i, s] = (
                                res.params[0]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[0]
                            )
                            new_taus[i, s] = (
                                res.params[-1]
                                if model.nloglikeobs(res.params)
                                < model.nloglikeobs(res2.params)
                                else res2.params[-1]
                            )
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in np.arange(n_spots):
                idx_nonzero = np.where(unique_values[s][:, 1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero, 1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero, 0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_tp = np.tile(
                    (mapping_matrices[s].T @ tumor_prop[:, s])[idx_nonzero]
                    / (mapping_matrices[s].T @ np.ones(tumor_prop.shape[0]))[
                        idx_nonzero
                    ],
                    n_states,
                )
                assert np.all(this_tp < 1 + 1e-4)
                this_weights = np.concatenate(
                    [tmp[i, idx_nonzero] for i in range(n_states)]
                )
                this_features = np.zeros((n_states * len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[
                        (i * len(idx_nonzero)) : ((i + 1) * len(idx_nonzero)), i
                    ] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array(
                    [
                        i
                        for i in range(this_features.shape[1])
                        if np.sum(this_weights[this_features[:, i] == 1]) >= 0.1
                    ]
                )
                idx_row_posweight = np.concatenate(
                    [np.where(this_features[:, k] == 1)[0] for k in idx_state_posweight]
                )
                y.append(this_y[idx_row_posweight])
                exposure.append(this_exposure[idx_row_posweight])
                weights.append(this_weights[idx_row_posweight])
                features.append(
                    this_features[idx_row_posweight, :][:, idx_state_posweight]
                )
                state_posweights.append(idx_state_posweight)
                tp.append(this_tp[idx_row_posweight])
                # tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_BetaBinom_mix(
                y, features, weights=weights, exposure=exposure, tumor_prop=tp
            )
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s, idx_state_posweight in enumerate(state_posweights):
                l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:, :] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(
                    disp=0,
                    maxiter=1500,
                    start_params=np.concatenate(
                        [
                            start_p_binom[idx_state_posweight, s]
                            for s, idx_state_posweight in enumerate(state_posweights)
                        ]
                        + [np.ones(1) * taus[0, s]]
                    ),
                    xtol=1e-4,
                    ftol=1e-4,
                )
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s, idx_state_posweight in enumerate(state_posweights):
                        l1 = int(np.sum([len(x) for x in state_posweights[:s]]))
                        l2 = int(np.sum([len(x) for x in state_posweights[: (s + 1)]]))
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:, :] = res2.params[-1]
    new_p_binom[new_p_binom < min_binom_prob] = min_binom_prob
    new_p_binom[new_p_binom > max_binom_prob] = max_binom_prob

    logger.info("Computed emission params for Beta Binomial Mix (no phasing, unique).")

    return new_p_binom, new_taus
