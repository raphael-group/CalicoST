import logging
from turtle import reset
import numpy as np
import pandas as pd
import pickle
from numba import njit
import scipy.special
import scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from tqdm import trange
import copy
from pathlib import Path
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.utils_distribution_fitting import *
from calicost.utils_IO import *
from calicost.utils_hmrf import *

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

logger = logging.getLogger(__name__)


def evaluate_hmm_emission_probability(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    res,
    smooth_mat,
    hmmclass,
    nodepotential,
    lambd=None,
):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(res['log_gamma'].shape[1] / n_obs)
    n_states = res["new_p_binom"].shape[0]

    if np.sum(single_base_nb_mean) > 0:
        logmu_shift = []
        for c in range(n_clones):
            this_pred_cnv = (
                np.argmax(
                    res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)], axis=0
                )
                % n_states
            )
            logmu_shift.append(
                scipy.special.logsumexp(
                    res["new_log_mu"][this_pred_cnv, :] + np.log(lambd).reshape(-1, 1),
                    axis=0,
                )
            )
        logmu_shift = np.vstack(logmu_shift)
        kwargs = {
            "logmu_shift": logmu_shift,
            "sample_length": np.ones(n_clones, dtype=int) * n_obs,
        }
    else:
        kwargs = {}

    # log probability of emission of each spot under each clone
    single_llf = np.zeros((N, n_clones))

    for i in trange(N):
        idx = smooth_mat[i, :].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        tmp_log_emission_rdr, tmp_log_emission_baf = (
            hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"],
                res["new_alphas"],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"],
                res["new_taus"],
                np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                **kwargs,
            )
        )
        for c in range(n_clones):
            if (
                np.sum(single_base_nb_mean[:, i : (i + 1)] > 0) > 0
                and np.sum(single_total_bb_RD[:, i : (i + 1)] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, i : (i + 1)] > 0)
                    / np.sum(single_base_nb_mean[:, i : (i + 1)] > 0)
                )
            else:
                ratio_nonzeros = 1.0
                
            if nodepotential == 'weighted_sum':
                single_llf[i, c] = ratio_nonzeros * np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0]
                        + res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)],
                        axis=0,
                    )
                )
            else:
                this_pred = res['pred_cnv'][(c * n_obs) : (c * n_obs + n_obs)]
                single_llf[i, c] = ratio_nonzeros * np.sum(
                    tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])

    return single_llf


def vi_reassignment(
    single_llf,
    adjacency_mat,
    Q,
    sample_ids,
    log_persample_weights,
    spatial_weight
):
    # Potts model likelihood
    potts_llf = spatial_weight * (adjacency_mat @ Q)
    
    log_q_z = single_llf + potts_llf + log_persample_weights[:, sample_ids].T

    # normalize and update Q
    new_Q = log_q_z - scipy.special.logsumexp(log_q_z, axis=1, keepdims=True)
    new_Q = np.exp(new_Q)

    # compute elbo
    """
    ELBO term can be written as: \sum_i (\sum_c q(z_i = c) (log p(x_i | z_i = c)) 
                                - \sum_i entropy(q(z_i)) 
                                + \sum_{(i,j) \in E} \sum_{z_i} \sum_{z_j} q(z_i)q(z_j)log p(z_i, z_j)
                                + \sum_i \sum_{z_i} q(z_i) log p(z_i))
    Note that the last log p(z_i)) is the log cluster size, and log p(z_i, z_j) is the Potts model pairwise term.
    """
    # entropy of q(z)
    EPS = 1e-20
    entropy = -np.sum(Q * np.log(Q + EPS))
    # sum of potts llf
    sum_updated_potts_llf = spatial_weight * np.trace(new_Q.T @ adjacency_mat @ new_Q)

    elbo = ((single_llf + log_persample_weights[:, sample_ids].T) * new_Q).sum() - entropy + sum_updated_potts_llf
    
    return new_Q, elbo


def merge_pseudobulk_by_Q(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    Q,
    single_tumor_prop,
    threshold
):
    """
    Attributes:
    -----------
    - single_X: np.array, shape=(n_obs, 2, n_spots). The UMI counts of each spot.
    - single_base_nb_mean: np.array, shape=(n_obs, n_spots). The mean of the negative binomial distribution of each spot.
    - single_total_bb_RD: np.array, shape=(n_obs, n_spots). The total read depth of each spot.
    - Q: np.array, shape=(n_spots, n_clones). The probability of each spot belonging to each clone.
    - single_tumor_prop: np.array, shape=(n_spots,). The proportion of tumor cells in each spot.
    - threshold: float. The threshold of tumor proportion to determine whether a spot is above the threshold.
    """
    if not single_tumor_prop is None:
        high_purity_spots = np.where(single_tumor_prop > threshold)[0]
    else:
        high_purity_spots = np.arange(single_X.shape[2])

    # weighted sum of single_X[:,0,:] and single_X[:,1,:] according to Q if the spots are above tumor proportion threshold
    # single_X[:,0,:] @ Q for transcript counts, single_X[:,1,:] @ Q for B allele counts
    X = np.einsum('ijk,kl->ijl', single_X[:,:,high_purity_spots], Q[high_purity_spots,:])
    base_nb_mean = single_base_nb_mean[:, high_purity_spots] @ Q[high_purity_spots,:]
    total_bb_RD = single_total_bb_RD[:, high_purity_spots] @ Q[high_purity_spots,:]

    if not single_tumor_prop is None:
        tumor_prop = single_tumor_prop[high_purity_spots].dot(Q[high_purity_spots,:]) / np.sum(Q[high_purity_spots,:], axis=0)
    else:
        tumor_prop = None

    return X, base_nb_mean, total_bb_RD, tumor_prop


def vi_hmrfmix_concatenate_pipeline(
    outdir,
    prefix,
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    Q,
    n_states,
    log_sitewise_transmat,
    coords=None,
    smooth_mat=None,
    adjacency_mat=None,
    sample_ids=None,
    max_iter_outer=5,
    nodepotential="max",
    hmmclass=hmm_sitewise,
    params="stmp",
    t=1 - 1e-6,
    random_state=0,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    unit_xsquared=9,
    unit_ysquared=3,
    spatial_weight=1.0 / 6,
    tumorprop_threshold=0.5,
    MIN_ALLELE_COUNT_PER_BIN=20,
):
    """
    Changed parameters:
    - Q: np.array, shape=(n_spots, n_clones). The probability of each spot belonging to each clone.
    """
    
    logger.info("Solving hmrfmix_concatenate_pipeline.")

    n_obs, _, n_spots = single_X.shape
    n_clones = Q.shape[1]

    # save initial Q
    file_path = Path(
        f"{outdir}/{prefix}_nstates{n_states}_{params}.pkl"
    )

    if not file_path.exists():
        logger.info(f"Writing initial assignment to {file_path}")
            
        pickle.dump(
            [{'Q':Q, 
              'log_emission_prob': np.zeros((single_X.shape[2], n_clones)),
              "new_log_mu": np.zeros((n_states, 1)),
              "new_alphas": np.zeros((n_states, 1)),
              "new_p_binom": np.zeros((n_states, 1)),
              "new_taus": np.zeros((n_states, 1)),
              "new_log_startprob": np.zeros(n_states),
              "new_log_transmat": np.zeros((n_states, n_states)),
             "log_gamma": np.zeros((n_states, n_obs * n_clones)),
             "pred_cnv": np.zeros(n_obs * n_clones, dtype=int),
             "llf": 0,
             'elbo':0
            }],
            open(file_path, 'wb')
        )

    # NB checking inputs
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]: i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([tmp_map_index[x] for x in sample_ids])

    logger.info("Merging pseudobulk based on clone index")

    # NB baseline proportion of UMI counts
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    if (init_log_mu is None) or (init_p_binom is None):
        logger.info("Initializing HMM parameters by GMM")

        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_Q(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            Q,
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )

        init_log_mu, init_p_binom = initialization_by_gmm(
            n_states,
            np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                -1, 2, 1
            ),
            base_nb_mean.flatten("F").reshape(-1, 1),
            total_bb_RD.flatten("F").reshape(-1, 1),
            params,
            random_state=random_state,
            in_log_space=False,
            only_minor=False,
        )
    else:
        logger.info("Using provided HMM initialization parameters")

    # NB initialization parameters for HMM
    if ("m" in params) and ("p" in params):
        last_log_mu = init_log_mu
        last_p_binom = init_p_binom
    elif "m" in params:
        last_log_mu = init_log_mu
        last_p_binom = None
    elif "p" in params:
        last_log_mu = None
        last_p_binom = init_p_binom
    last_alphas = init_alphas
    last_taus = init_taus

    logger.info(f"Computing HMM for {max_iter_outer} iterations.")

    for r in range(1, max_iter_outer + 1):
        """
        NB assuming file f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" exists.
           When r == 0, f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" should
           contain two keys: "num_iterations" and f"round_-1_assignment" for clone
           initialization
        """
        logger.info(f"Loading {outdir}/{prefix}_nstates{n_states}_{params}.npz")

        allres = pickle.load(
            open(f"{outdir}/{prefix}_nstates{n_states}_{params}.pkl", 'rb')
        ) # a list of dictionaries, where each dictionary contains the results of one iteration. allres[0] is initialization.

        if len(allres) > r:
            logger.info(f"Resuming pre-computed HMM results for iteration {r}.")
            r = len(allres) - 1

        # placeholder for the results using the last iteration
        res = copy.deepcopy(allres[-1])

        # pseudobulking spots for each clone
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_Q(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res['Q'],
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )
        supported_clones = np.where(np.sum(total_bb_RD, axis=0) > MIN_ALLELE_COUNT_PER_BIN * n_obs)[0]
        
        sample_length = np.ones(len(supported_clones), dtype=int) * n_obs
        remain_kwargs = {"sample_length": sample_length, "lambd": lambd}

        if r > 1:
            remain_kwargs["log_gamma"] = res["log_gamma"]

        # log clone size
        log_persample_weights = np.sum(res['Q'][single_tumor_prop > tumorprop_threshold, :], axis=0) / np.sum(single_tumor_prop > tumorprop_threshold)
        log_persample_weights = np.where(log_persample_weights > 0, np.log(log_persample_weights), -50)
        log_persample_weights = np.tile(log_persample_weights.reshape(-1,1), [1, n_samples])

        logger.info(f"Computing HMM iteration {r}.")

        hmm_res = pipeline_baum_welch(
            None,
            np.vstack([X[:, 0, supported_clones].flatten("F"), X[:, 1, supported_clones].flatten("F")]).T.reshape(
                -1, 2, 1
            ),
            np.tile(lengths, len(supported_clones)),
            n_states,  # base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, \
            base_nb_mean[:,supported_clones].flatten("F").reshape(-1, 1),
            total_bb_RD[:,supported_clones].flatten("F").reshape(-1, 1),
            np.tile(log_sitewise_transmat, len(supported_clones)),
            np.repeat(tumor_prop[supported_clones], n_obs).reshape(-1, 1),
            hmmclass=hmmclass,
            params=params,
            t=t,
            random_state=random_state,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            is_diag=is_diag,
            init_log_mu=last_log_mu,
            init_p_binom=last_p_binom,
            init_alphas=last_alphas,
            init_taus=last_taus,
            max_iter=max_iter,
            tol=tol,
            **remain_kwargs,
        )

        # update res with the results of the HMM
        for i,c in enumerate(supported_clones):
            res['log_gamma'][:, (c * n_obs) : (c * n_obs + n_obs)] = hmm_res['log_gamma'][:, (i * n_obs) : (i * n_obs + n_obs)]
            res['pred_cnv'][(c * n_obs) : (c * n_obs + n_obs)] = hmm_res['pred_cnv'][(i * n_obs) : (i * n_obs + n_obs)]
        for k,v in hmm_res.items():
            if not k in ['log_gamma', 'pred_cnv']:
                res[k] = v

        # NB HMRF clone assignmment
        logger.info(
            f"Assigning HMRF clone for iteration {r} with nodepotential={nodepotential}."
        )
        single_llf = evaluate_hmm_emission_probability(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            res,
            smooth_mat,
            hmmclass,
            nodepotential,
            lambd=lambd,
        )

        new_Q, elbo = (
            vi_reassignment(
                single_llf,
                adjacency_mat,
                Q,
                sample_ids,
                log_persample_weights,
                spatial_weight=spatial_weight,
            )
        )
        res['log_emission_prob'] = single_llf
        res["Q"] = new_Q
        res["elbo"] = elbo
                
        allres.append( res )

        logger.info(
            f"Writing round ({r}) assignments for HMM iteration {r} to {outdir}/{prefix}_nstates{n_states}_{params}.npz"
        )

        pickle.dump(allres, open(f"{outdir}/{prefix}_nstates{n_states}_{params}.pkl", 'wb'))

        #####

        logger.info(f"Regrouping to pseudobulk for iteration {r}.")

        if "mp" in params:
            logger.info(
                f"Outer iteration {r}: mean abs. diff. (mu, p) = {np.mean(np.abs(allres[-1]['new_log_mu'] - allres[-2]['new_log_mu']))}, {np.mean(np.abs(allres[-1]['new_p_binom'] - allres[-2]['new_p_binom']))}"
            )
        elif "m" in params:
            logger.info(
                f"Outer iteration {r}: mean abs. diff. between NB parameters = {np.mean(np.abs(allres[-1]['new_log_mu'] - allres[-2]['new_log_mu']))}"
            )
        elif "p" in params:
            logger.info(
                f"Outer iteration {r}: BetaBinom parameters mean abs. diff. = {np.mean(np.abs(allres[-1]['new_p_binom'] - allres[-2]['new_p_binom']))}"
            )

        logger.info(
            f"Outer iteration {r}: abs difference between posterior Q = {np.sum(np.abs(allres[-1]['Q'] - allres[-2]['Q']))}"
        )

        logger.info(
            f"Outer iteration {r}: ELBO of the last five iterations = {[allres[i]['elbo'] for i in range(max(0, r - 5), r)]}"
        )

        if len(allres) > 2 and np.abs(allres[-1]['elbo'] - allres[-2]['elbo']) < tol * single_X.shape[2] * n_obs:
            break