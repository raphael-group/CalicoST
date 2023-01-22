import numpy as np
from numba import njit
import scipy.special
import scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import trange
import copy
from pathlib import Path
from hmm_NB_BB_phaseswitch import *
from utils_distribution_fitting import *
from utils_IO import *
from simple_sctransform import *

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning


def compute_adjacency_mat(coords, unit_xsquared=9, unit_ysquared=3):
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    # adjacency
    A = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.int8 )
    for i in range(coords.shape[0]):
        indexes = np.where(pairwise_squared_dist[i,:] <= unit_xsquared + unit_ysquared)[0]
        indexes = np.array([j for j in indexes if j != i])
        if len(indexes) > 0:
            A[i, indexes] = 1
    A = scipy.sparse.csr_matrix(A)
    return A


# def rectangle_initialize_initial_clone_mix(coords, n_clones, single_tumor_prop, threshold=0.5, random_state=0):
#     np.random.seed(random_state)
#     p = int(np.ceil(np.sqrt(n_clones)))
#     # partition the range of x and y axes based on tumor spots coordinates
#     idx_tumor = np.where(single_tumor_prop > threshold)[0]
#     px = np.random.dirichlet( np.ones(p) * 10 )
#     xboundary = np.percentile(coords[idx_tumor, 0], 100*np.cumsum(px))
#     xboundary[-1] = np.max(coords[:,0]) + 1
#     xdigit = np.digitize(coords[:,0], xboundary, right=True)
#     py = np.random.dirichlet( np.ones(p) * 10 )
#     yboundary = np.percentile(coords[idx_tumor, 1], 100*np.cumsum(py))
#     yboundary[-1] = np.max(coords[:,1]) + 1
#     ydigit = np.digitize(coords[:,1], yboundary, right=True)
#     block_id = xdigit * p + ydigit
#     # assigning blocks to clone (note that if sqrt(n_clone) is not an integer, multiple blocks can be assigneed to one clone)
#     block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)
#     while len(np.unique(block_clone_map)) < n_clones:
#         bc = np.bincount(block_clone_map, minlength=n_clones)
#         assert np.any(bc==0)
#         block_clone_map[np.where(block_clone_map==np.argmax(bc))[0][0]] = np.where(bc==0)[0][0]
#     block_clone_map = {i:block_clone_map[i] for i in range(len(block_clone_map))}
#     clone_id = np.array([block_clone_map[i] for i in block_id])
#     initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
#     return initial_clone_index


def rectangle_initialize_initial_clone_mix(coords, n_clones, single_tumor_prop, threshold=0.5, random_state=0):
    np.random.seed(random_state)
    p = int(np.ceil(np.sqrt(n_clones)))
    # partition the range of x and y axes based on tumor spots coordinates
    idx_tumor = np.where(single_tumor_prop > threshold)[0]
    px = np.random.dirichlet( np.ones(p) * 10 )
    xboundary = np.percentile(coords[idx_tumor, 0], 100*np.cumsum(px))
    xboundary[-1] = np.max(coords[:,0]) + 1
    xdigit = np.digitize(coords[:,0], xboundary, right=True)
    ydigit = np.zeros(coords.shape[0], dtype=int)
    for x in range(p):
        idx_tumor = np.where((single_tumor_prop > threshold) & (xdigit==x))[0]
        idx_both = np.where(xdigit == x)[0]
        py = np.random.dirichlet( np.ones(p) * 10 )
        yboundary = np.percentile(coords[idx_tumor, 1], 100*np.cumsum(py))
        yboundary[-1] = np.max(coords[:,1]) + 1
        ydigit[idx_both] = np.digitize(coords[idx_both,1], yboundary, right=True)
    block_id = xdigit * p + ydigit
    # assigning blocks to clone (note that if sqrt(n_clone) is not an integer, multiple blocks can be assigneed to one clone)
    block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)
    while len(np.unique(block_clone_map)) < n_clones:
        bc = np.bincount(block_clone_map, minlength=n_clones)
        assert np.any(bc==0)
        block_clone_map[np.where(block_clone_map==np.argmax(bc))[0][0]] = np.where(bc==0)[0][0]
    block_clone_map = {i:block_clone_map[i] for i in range(len(block_clone_map))}
    clone_id = np.array([block_clone_map[i] for i in block_id])
    initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
    return initial_clone_index


def merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop, threshold=0.5):
    n_obs = single_X.shape[0]
    n_spots = len(clone_index)
    X = np.zeros((n_obs, 2, n_spots))
    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))
    tumor_prop = np.zeros(n_spots)

    for k,idx in enumerate(clone_index):
        idx = idx[np.where(single_tumor_prop[idx] > threshold)[0]]
        X[:,:, k] = np.sum(single_X[:,:,idx], axis=2)
        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)
        tumor_prop[k] = np.mean(single_tumor_prop[idx])

    return X, base_nb_mean, total_bb_RD, tumor_prop


def hmrf_reassignment_mix(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom_mix(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], single_tumor_prop[i:(i+1)])
            single_llf[i,c] = np.sum(tmp_log_emission[pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrf_reassignment_posterior_mix(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom_mix(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], single_tumor_prop[i:(i+1)])
            single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission[:,:, 0] + res["log_gamma"], axis=0) )
        w_node = single_llf[i,:]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def similarity_components_pearsonresidual_silhouette_mix(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop):
    return NotImplemented


def hmrf_pipeline_mix(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, \
    n_states, log_sitewise_transmat, coords=None, adjacency_mat=None, max_iter_outer=5, nodepotential="max", params="stmp", t=1-1e-6, random_state=0, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
    # Note that initial_clone_index covers all spots, but all normal spots are assigned to one of the tumor clone but with tumor purity \approx 0
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    # pseudobulk
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop)
    # initialize HMM parameters by GMM
    tmp_log_mu, tmp_p_binom = initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params, random_state=random_state, in_log_space=False, remove_baf_zero=True)
    # initialization parameters for HMM
    if ("m" in params) and ("p" in params):
        last_log_mu = tmp_log_mu
        last_p_binom = tmp_p_binom
    elif "m" in params:
        last_log_mu = tmp_log_mu
        last_p_binom = None
    elif "p" in params:
        last_log_mu = None
        last_p_binom = tmp_p_binom
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)
    for c,idx in enumerate(initial_clone_index):
        last_assignment[idx] = c
    # HMM
    for r in range(max_iter_outer):
        if not Path(f"{outdir}/round{r}_nstates{n_states}_{params}.npz").exists():
            ##### initialize with the parameters of last iteration #####
            res = pipeline_baum_welch(None, X, lengths, n_states, base_nb_mean, total_bb_RD, \
                              log_sitewise_transmat, tumor_prop, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, is_diag=is_diag, \
                              init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = hmrf_reassignment_mix(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior_mix(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
            else:
                raise Exception("Unknown mode for nodepotential!")
            res["prev_assignment"] = last_assignment
            res["new_assignment"] = new_assignment
            res["total_llf"] = total_llf

            # save results
            np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)

        else:
            res = np.load(f"{outdir}/round{r}_nstates{n_states}_{params}.npz")

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop)

        # update last parameter
        if "mp" in params:
            print("outer iteration {}: total_llf = {}, difference between parameters = {}, {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: total_llf = {}, difference between NB parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: total_llf = {}, difference between BetaBinom parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        if np.all( last_assignment == res["new_assignment"] ):
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]


def hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    for i in trange(N):
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_mix_v2(single_X[:,:,i:(i+1)], \
                                            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], \
                                            single_total_bb_RD[:,i:(i+1)], res["new_p_binom"], res["new_taus"], single_tumor_prop[i:(i+1)])
        for c in range(n_clones):
            this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
    #
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


# def hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight, threshold=0.5):
#     N = single_X.shape[2]
#     n_obs = single_X.shape[0]
#     n_clones = int(len(pred) / n_obs)
#     n_states = res["new_p_binom"].shape[0]
#     single_llf = np.zeros((N, n_clones))
#     new_assignment = copy.copy(prev_assignment)
#     #
#     # first re-assign spots with tumor proportion > 0.5, the spots with smaller tumor proportion don't have a weight
#     idx_spots = np.where(single_tumor_prop > threshold)[0]
#     sub_adjacency_mat = adjacency_mat[np.ix_(idx_spots, idx_spots)]
#     for i,idx in enumerate(idx_spots):
#         tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_mix_v2(single_X[:,:,idx:(idx+1)], \
#                                             single_base_nb_mean[:,idx:(idx+1)], res["new_log_mu"], res["new_alphas"], \
#                                             single_total_bb_RD[:,idx:(idx+1)], res["new_p_binom"], res["new_taus"], single_tumor_prop[idx:(idx+1)])
#         for c in range(n_clones):
#             this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
#             if np.sum(single_base_nb_mean[:,idx:(idx+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,idx:(idx+1)] > 0) > 0:
#                 ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx:(idx+1)] > 0) / np.sum(single_base_nb_mean[:,idx:(idx+1)] > 0)
#                 single_llf[idx,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
#             else:
#                 single_llf[idx,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
#         w_node = single_llf[idx,:]
#         w_edge = np.zeros(n_clones)
#         for j in sub_adjacency_mat[i,:].nonzero()[1]:
#             w_edge[new_assignment[idx_spots[j]]] += 1
#         new_assignment[idx] = np.argmax( w_node + spatial_weight * w_edge )
#     #
#     # then re-assign the remaining spots, in this case surrounding spots with > threshold tumor proportion have a weight
#     idx_spots = np.where(single_tumor_prop <= threshold)[0]
#     for i,idx in enumerate(idx_spots):
#         tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_mix_v2(single_X[:,:,idx:(idx+1)], \
#                                             single_base_nb_mean[:,idx:(idx+1)], res["new_log_mu"], res["new_alphas"], \
#                                             single_total_bb_RD[:,idx:(idx+1)], res["new_p_binom"], res["new_taus"], single_tumor_prop[idx:(idx+1)])
#         for c in range(n_clones):
#             this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
#             if np.sum(single_base_nb_mean[:,idx:(idx+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,idx:(idx+1)] > 0) > 0:
#                 ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx:(idx+1)] > 0) / np.sum(single_base_nb_mean[:,idx:(idx+1)] > 0)
#                 single_llf[idx,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
#             else:
#                 single_llf[idx,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
#         w_node = single_llf[idx,:]
#         w_edge = np.zeros(n_clones)
#         for j in adjacency_mat[idx,:].nonzero()[1]:
#             w_edge[new_assignment[j]] += 1
#         new_assignment[idx] = np.argmax( w_node + spatial_weight * w_edge )
#     # compute total log likelihood log P(X | Z) + log P(Z)
#     total_llf = np.sum(single_llf[np.arange(N), new_assignment])
#     for i in range(N):
#         total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
#     return new_assignment, single_llf, total_llf


def hmrfmix_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_mix_v2(single_X[:,:,i:(i+1)], \
                                            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], \
                                            single_total_bb_RD[:,i:(i+1)], res["new_p_binom"], res["new_taus"], single_tumor_prop[i:(i+1)])
        for c in range(n_clones):
            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, (c*n_obs):(c*n_obs+n_obs), 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, (c*n_obs):(c*n_obs+n_obs), 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, (c*n_obs):(c*n_obs+n_obs), 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, (c*n_obs):(c*n_obs+n_obs), 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
        w_node = single_llf[i,:]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def aggr_hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, smooth_mat, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    for i in trange(N):
        idx = smooth_mat[i,:].nonzero()[1]
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_mix_v2( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"], np.array([np.mean(single_tumor_prop[idx]) ]) )
        for c in range(n_clones):
            this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        # new_assignment[i] = np.argmax( w_node )
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
    #
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states, log_sitewise_transmat, \
    coords=None, smooth_mat=None, adjacency_mat=None, max_iter_outer=5, nodepotential="max", params="stmp", t=1-1e-6, random_state=0, \
    init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    relative_rdr_weight=1.0, is_diag=True, \
    max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    # pseudobulk
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params, random_state=random_state, in_log_space=False, remove_baf_zero=False)
    # initialization parameters for HMM
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
    last_assignment = np.zeros(single_X.shape[2], dtype=int)
    for c,idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # HMM
    for r in range(max_iter_outer):
        # assuming file f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" exists. When r == 0, f"{outdir}/{prefix}_nstates{n_states}_{params}.npz" should contain two keys: "num_iterations" and f"round_-1_assignment" for clone initialization
        allres = np.load(f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", allow_pickle=True)
        allres = dict(allres)
        if allres["num_iterations"] > r:
            res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
                "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
                "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
                "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
                "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
        else:            
            res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), n_states, \
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, params=params, t=t, random_state=random_state, \
                            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                            is_diag=is_diag, relative_rdr_weight=relative_rdr_weight, \
                            init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrfmix_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, \
                    adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            # elif nodepotential == "test_sum":
            #     new_assignment, single_llf, total_llf = aggr_hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, \
            #         smooth_mat, adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            else:
                raise Exception("Unknown mode for nodepotential!")
            res["prev_assignment"] = last_assignment
            res["new_assignment"] = new_assignment
            res["total_llf"] = total_llf
            # append to allres
            for k,v in res.items():
                if k == "prev_assignment":
                    allres[f"round{r-1}_assignment"] = v
                elif k == "new_assignment":
                    allres[f"round{r}_assignment"] = v
                else:
                    allres[f"round{r}_{k}"] = v
            allres["num_iterations"] = r + 1
            np.savez(f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", **allres)
        #
        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop)
        #
        if "mp" in params:
            print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        # if np.all( last_assignment == res["new_assignment"] ):
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
