import logging
from turtle import reset
import numpy as np
import pandas as pd
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


############################################################
# Pure clone
############################################################

@njit(cache=True, parallel=False)
def edge_update(n_clones, idx, values):
    w_edge = np.zeros(n_clones, dtype=float)

    for i, value in enumerate(values):
        w_edge[idx[i]] += value

    return w_edge

def solve_edges(adjacency_mat, new_assignment, n_clones):
    new = True

    if new:
        neighbors = adjacency_mat[i,:].nonzero()[1]
        idx = np.where(new_assignment[neighbors] >= 0)[0]

        neighbors = neighbors[idx]
        values = adjacency_mat[i, neighbors].data

        w_edge = edge_update(n_clones, new_assignment[neighbors], values)
    else:
        w_edge = np.zeros(n_clones)

        for j in adjacency_mat[i,:].nonzero()[1]:
            if new_assignment[j] >= 0:
                # w_edge[new_assignment[j]] += 1                                                                                                                                                                                                                                                                                                               
                w_edge[new_assignment[j]] += adjacency_mat[i,j]

    return w_edge

@profile
def hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="hmrf_reassignment_posterior"):
        idx = smooth_mat[i,:].nonzero()[1]
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                                np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:,0] + res["log_gamma"][:,:,c], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:,0] + res["log_gamma"][:,:,c], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:,0] + res["log_gamma"][:,:,c], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:,0] + res["log_gamma"][:,:,c], axis=0) )
                    
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)
                
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="aggr_hmrf_reassignment"):
        idx = smooth_mat[i,:].nonzero()[1]
        # idx = np.append(idx, np.array([i]))
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                                np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
    
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)

        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def hmrf_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="hmrf_reassignment_posterior_concatenate"):
        idx = smooth_mat[i,:].nonzero()[1]
        tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"])
        for c in range(n_clones):
            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)

        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    """
    HMRF assign spots to tumor clones.

    Attributes
    ----------
    single_X : array, shape (n_bins, 2, n_spots)
        BAF and RD count matrix for all bins in all spots.

    single_base_nb_mean : array, shape (n_bins, n_spots)
        Diploid baseline of gene expression matrix.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele UMI count matrix.

    res : dictionary
        Dictionary of estimated HMM parameters.

    pred : array, shape (n_bins * n_clones)
        HMM states for all bins and all clones. (Derived from forward-backward algorithm)

    smooth_mat : array, shape (n_spots, n_spots)
        Matrix used for feature propagation for computing log likelihood.

    adjacency_mat : array, shape (n_spots, n_spots)
        Adjacency matrix used to evaluate label consistency in HMRF.

    prev_assignment : array, shape (n_spots,)
        Clone assignment of the previous iteration.

    spatial_weight : float
        Scaling factor for HMRF label consistency between adjacent spots.

    Returns
    ----------
    new_assignment : array, shape (n_spots,)
        Clone assignment of this new iteration.

    single_llf : array, shape (n_spots, n_clones)
        Log likelihood of each spot given that its label is each clone.

    total_llf : float
        The HMRF objective, which is the sum of log likelihood under the optimal labels plus the sum of edge potentials.
    """
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="aggr_hmrf_reassignment_concatenate"):
        idx = smooth_mat[i,:].nonzero()[1]
        # idx = np.append(idx, np.array([i]))
        tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"])
        for c in range(n_clones):
            this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]
        # new_assignment[i] = np.argmax( w_node )

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def merge_by_minspots(assignment, res, single_total_bb_RD, min_spots_thresholds=50, min_umicount_thresholds=0, single_tumor_prop=None, threshold=0.5):
    n_clones = len(np.unique(assignment))
    if n_clones == 1:
        merged_groups = [ [assignment[0]] ]
        return merged_groups, res

    n_obs = int(len(res["pred_cnv"]) / n_clones)
    new_assignment = copy.copy(assignment)
    if single_tumor_prop is None:
        tmp_single_tumor_prop = np.array([1] * len(assignment))
    else:
        tmp_single_tumor_prop = single_tumor_prop
    unique_assignment = np.unique(new_assignment)
    # find entries in unique_assignment such that either min_spots_thresholds or min_umicount_thresholds is not satisfied
    failed_clones = [ c for c in unique_assignment if (np.sum(new_assignment[tmp_single_tumor_prop > threshold] == c) < min_spots_thresholds) or \
                     (np.sum(single_total_bb_RD[:, (new_assignment == c)&(tmp_single_tumor_prop > threshold)]) < min_umicount_thresholds) ]
    # find the remaining unique_assigment that satisfies both thresholds
    successful_clones = [ c for c in unique_assignment if not c in failed_clones ]
    # initial merging groups: each successful clone is its own group
    merging_groups = [[i] for i in successful_clones]
    # for each failed clone, assign them to the closest successful clone
    if len(failed_clones) > 0:
        for c in failed_clones:
            idx_max = np.argmax([np.sum(single_total_bb_RD[:, (new_assignment == c_prime)&(tmp_single_tumor_prop > threshold)]) for c_prime in successful_clones])
            merging_groups[idx_max].append(c)
    map_clone_id = {}
    for i,x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in new_assignment])
    # while np.min(np.bincount(new_assignment[tmp_single_tumor_prop > threshold])) < min_spots_thresholds or \
    #     np.min([ np.sum(single_total_bb_RD[:, (new_assignment == c)&(tmp_single_tumor_prop > threshold)]) for c in unique_assignment ]) < min_umicount_thresholds:
    #     idx_min = np.argmin(np.bincount(new_assignment[tmp_single_tumor_prop > threshold]))
    #     idx_max = np.argmax(np.bincount(new_assignment[tmp_single_tumor_prop > threshold]))
    #     merging_groups = [ [i] for i in range(n_clones) if (i!=idx_min) and (i!=idx_max)] + [[idx_min, idx_max]]
    #     merging_groups.sort(key = lambda x:np.min(x))
    #     # clone assignment after merging
    #     map_clone_id = {}
    #     for i,x in enumerate(merging_groups):
    #         for z in x:
    #             map_clone_id[z] = i
    #     new_assignment = np.array([map_clone_id[x] for x in new_assignment])
    #     unique_assignment = np.unique(new_assignment)
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.nan
    merged_res["pred_cnv"] = np.concatenate([ res["pred_cnv"][(c[0]*n_obs):(c[0]*n_obs+n_obs)] for c in merging_groups ])
    merged_res["log_gamma"] = np.hstack([ res["log_gamma"][:, (c[0]*n_obs):(c[0]*n_obs+n_obs)] for c in merging_groups ])
    return merging_groups, merged_res

@profile
def hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states, \
    log_sitewise_transmat, coords=None, smooth_mat=None, adjacency_mat=None, sample_ids=None, max_iter_outer=5, nodepotential="max", \
    hmmclass=hmm_sitewise, params="stmp", t=1-1e-6, random_state=0, init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]:i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([ tmp_map_index[x] for x in sample_ids])
    log_persample_weights = np.ones((n_clones, n_samples)) * np.log(n_clones)
    # pseudobulk
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params, random_state=random_state, in_log_space=False, only_minor=False)
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
        if not Path(f"{outdir}/round{r}_nstates{n_states}_{params}.npz").exists():
            ##### initialize with the parameters of last iteration #####
            res = pipeline_baum_welch(None, X, lengths, n_states, base_nb_mean, total_bb_RD, log_sitewise_transmat, \
                              hmmclass=hmmclass, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                              is_diag=is_diag, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            else:
                raise Exception("Unknown mode for nodepotential!")
            # handle the case when one clone has zero spots
            if len(np.unique(new_assignment)) < X.shape[2]:
                res["assignment_before_reindex"] = new_assignment
                remaining_clones = np.sort(np.unique(new_assignment))
                re_indexing = {c:i for i,c in enumerate(remaining_clones)}
                new_assignment = np.array([re_indexing[x] for x in new_assignment])
            #
            res["prev_assignment"] = last_assignment
            res["new_assignment"] = new_assignment
            res["total_llf"] = total_llf

            # save results
            np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)

        else:
            res = np.load(f"{outdir}/round{r}_nstates{n_states}_{params}.npz")

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)

        # update last parameter
        if "mp" in params:
            print("outer iteration {}: total_llf = {}, difference between parameters = {}, {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: total_llf = {}, difference between NB parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: total_llf = {}, difference between BetaBinom parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99 or len(np.unique(res["new_assignment"])) == 1:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))
        for sidx in range(n_samples):
            index = np.where(sample_ids == sidx)[0]
            this_persample_weight = np.bincount(res["new_assignment"][index], minlength=X.shape[2]) / len(index)
            log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
            log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])

@profile
def hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states, \
    log_sitewise_transmat, coords=None, smooth_mat=None, adjacency_mat=None, sample_ids=None, max_iter_outer=5, nodepotential="max", hmmclass=hmm_sitewise, \
    params="stmp", t=1-1e-6, random_state=0, init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)
    # checking input
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]:i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([ tmp_map_index[x] for x in sample_ids])
    log_persample_weights = np.ones((n_clones, n_samples)) * np.log(n_clones)
    # pseudobulk
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params, random_state=random_state, in_log_space=False, only_minor=False)
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
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), \
                            hmmclass=hmmclass, params=params, t=t, random_state=random_state, \
                            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                            is_diag=is_diag, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # HMRF clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            else:
                raise Exception("Unknown mode for nodepotential!")
            # handle the case when one clone has zero spots
            if len(np.unique(new_assignment)) < X.shape[2]:
                res["assignment_before_reindex"] = new_assignment
                remaining_clones = np.sort(np.unique(new_assignment))
                re_indexing = {c:i for i,c in enumerate(remaining_clones)}
                new_assignment = np.array([re_indexing[x] for x in new_assignment])
                concat_idx = np.concatenate([ np.arange(c*n_obs, c*n_obs+n_obs) for c in remaining_clones ])
                res["log_gamma"] = res["log_gamma"][:,concat_idx]
                res["pred_cnv"] = res["pred_cnv"][concat_idx]
            #
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
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        #
        if "mp" in params:
            print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        # if np.all( last_assignment == res["new_assignment"] ):
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99 or len(np.unique(res["new_assignment"])) == 1:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))
        for sidx in range(n_samples):
            index = np.where(sample_ids == sidx)[0]
            this_persample_weight = np.bincount(res["new_assignment"][index], minlength=X.shape[2]) / len(index)
            log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
            log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])



############################################################
# Normal-tumor clone mixture
############################################################

@profile
def aggr_hmrfmix_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    #
    posterior = np.zeros((N, n_clones))
    #
    for i in trange(N, desc="aggr_hmrfmix_reassignment"):
        idx = smooth_mat[i,:].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0:
                mu = np.exp(res["new_log_mu"][(pred%n_states),:]) / np.sum(np.exp(res["new_log_mu"][(pred%n_states),:]) * lambd)
                weighted_tp = (np.mean(single_tumor_prop[idx]) * mu) / (np.mean(single_tumor_prop[idx]) * mu + 1 - np.mean(single_tumor_prop[idx]))
            else:
                weighted_tp = np.repeat(np.mean(single_tumor_prop[idx]), single_X.shape[0])
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], np.ones((n_obs,1)) * np.mean(single_tumor_prop[idx]), weighted_tp.reshape(-1,1) )
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
        #
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)

        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )
    #
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf
    
@profile
def hmrfmix_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="hmrfmix_reassignment_posterior"):
        idx = smooth_mat[i,:].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            if np.sum(single_base_nb_mean) > 0:
                this_pred_cnv = res["pred_cnv"][:,c]
                logmu_shift = np.array( scipy.special.logsumexp(res["new_log_mu"][this_pred_cnv,c] + np.log(lambd), axis=0) )
                kwargs = {"logmu_shift":logmu_shift.reshape(1,1), "sample_length":np.array([n_obs])}
            else:
                kwargs = {}
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], np.ones((n_obs,1)) * np.mean(single_tumor_prop[idx]), **kwargs )
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:,0] + res["log_gamma"][:,:,c], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:,0] + res["log_gamma"][:,:,c], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:,0] + res["log_gamma"][:,:,c], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:,0] + res["log_gamma"][:,:,c], axis=0) )
        
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)
        
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def hmrfmix_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states, log_sitewise_transmat, \
    coords=None, smooth_mat=None, adjacency_mat=None, sample_ids=None, max_iter_outer=5, nodepotential="max", hmmclass=hmm_sitewise, params="stmp", t=1-1e-6, random_state=0, \
    init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6, tumorprop_threshold=0.5):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]:i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([ tmp_map_index[x] for x in sample_ids])
    log_persample_weights = np.ones((n_clones, n_samples)) * np.log(n_clones)
    # pseudobulk
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop, threshold=tumorprop_threshold)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params, random_state=random_state, in_log_space=False, only_minor=False)
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
    n_clones = len(initial_clone_index)

    # HMM
    for r in range(max_iter_outer):
        allres = np.load(f"{outdir}/{prefix}_nstates{n_states}_{params}.npz", allow_pickle=True)
        allres = dict(allres)
        if allres["num_iterations"] > r:
            res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
                "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
                "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
                "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
                "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
        else:
            res = {"new_log_mu":[], "new_alphas":[], "new_p_binom":[], "new_taus":[], "new_log_startprob":[], "new_log_transmat":[], "log_gamma":[], "pred_cnv":[], "llf":[]}
            for c in range(n_clones):
                tmpres = pipeline_baum_welch(None, X[:,:,c:(c+1)], lengths, n_states, base_nb_mean[:,c:(c+1)], total_bb_RD[:,c:(c+1)],  log_sitewise_transmat, np.repeat(tumor_prop[c], X.shape[0]).reshape(-1,1), \
                            hmmclass=hmmclass, params=params, t=t, \
                            random_state=random_state, fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                            is_diag=is_diag, init_log_mu=last_log_mu[:,c:(c+1)], init_p_binom=last_p_binom[:,c:(c+1)], init_alphas=last_alphas[:,c:(c+1)], init_taus=last_taus[:,c:(c+1)], max_iter=max_iter, tol=tol)
                pred = np.argmax(tmpres["log_gamma"], axis=0)
                for k in res.keys():
                    res[k] = [res[k], tmpres[k]]
            res["new_log_mu"] = np.hstack(res["new_log_mu"])
            res["new_alphas"] = np.hstack(res["new_alphas"])
            res["new_p_binom"] = np.hstack(res["new_p_binom"])
            res["new_taus"] = np.hstack(res["new_taus"])
            res["new_log_startprob"] = np.hstack(res["new_log_startprob"])
            res["new_log_transmat"] = np.dstack(res["new_log_transmat"])
            res["log_gamma"] = np.hstack(res["log_gamma"])
            res["pred_cnv"] = np.vstack(res["pred_cnv"]).T

            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrfmix_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrfmix_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            else:
                raise Exception("Unknown mode for nodepotential!")
            # handle the case when one clone has zero spots
            if len(np.unique(new_assignment)) < X.shape[2]:
                res["assignment_before_reindex"] = new_assignment
                remaining_clones = np.sort(np.unique(new_assignment))
                re_indexing = {c:i for i,c in enumerate(remaining_clones)}
                new_assignment = np.array([re_indexing[x] for x in new_assignment])
            #
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

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop, threshold=tumorprop_threshold)

        # update last parameter
        if "mp" in params:
            print("outer iteration {}: total_llf = {}, difference between parameters = {}, {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: total_llf = {}, difference between NB parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: total_llf = {}, difference between BetaBinom parameters = {}".format( r, res["total_llf"], np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        # if np.all( last_assignment == res["new_assignment"] ):
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99 or len(np.unique(res["new_assignment"])) == 1:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))
        for sidx in range(n_samples):
            index = np.where(sample_ids == sidx)[0]
            this_persample_weight = np.bincount(res["new_assignment"][index], minlength=X.shape[2]) / len(index)
            log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
            log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])

@profile
def hmrfmix_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    if np.sum(single_base_nb_mean) > 0:
        logmu_shift = []
        for c in range(n_clones):
            this_pred_cnv = np.argmax(res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0)%n_states
            logmu_shift.append( scipy.special.logsumexp(res["new_log_mu"][this_pred_cnv,:] + np.log(lambd).reshape(-1,1), axis=0) )
        logmu_shift = np.vstack(logmu_shift)
        kwargs = {"logmu_shift":logmu_shift, "sample_length":np.ones(n_clones,dtype=int) * n_obs}
    else:
        kwargs = {}
    #
    posterior = np.zeros((N, n_clones))

    for i in trange(N, desc="hmrfmix_reassignment_posterior_concatenate"):
        idx = smooth_mat[i,:].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"], np.ones((n_obs,1)) * np.mean(single_tumor_prop[idx]), **kwargs )

            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) ) + \
                    np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)], axis=0) )
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)

        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def aggr_hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, smooth_mat, adjacency_mat, prev_assignment, sample_ids, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise, return_posterior=False):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    #
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    #
    posterior = np.zeros((N, n_clones))
    #
    for i in trange(N, desc="aggr_hmrfmix_reassignment_concatenate"):
        idx = smooth_mat[i,:].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            this_pred = pred[(c*n_obs):(c*n_obs+n_obs)]
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0:
                mu = np.exp(res["new_log_mu"][(this_pred%n_states),:]) / np.sum(np.exp(res["new_log_mu"][(this_pred%n_states),:]) * lambd)
                weighted_tp = (np.mean(single_tumor_prop[idx]) * mu) / (np.mean(single_tumor_prop[idx]) * mu + 1 - np.mean(single_tumor_prop[idx]))
            else:
                weighted_tp = np.repeat(np.mean(single_tumor_prop[idx]), single_X.shape[0])
            tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"], np.ones((n_obs,1)) * np.mean(single_tumor_prop[idx]), weighted_tp.reshape(-1,1) )

            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                # ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        w_node += log_persample_weights[:,sample_ids[i]]

        w_edge = solve_edges(adjacency_mat, new_assignment, n_clones)
        
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
        
        posterior[i,:] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )
    #
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

@profile
def hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states, log_sitewise_transmat, \
    coords=None, smooth_mat=None, adjacency_mat=None, sample_ids=None, max_iter_outer=5, nodepotential="max", hmmclass=hmm_sitewise, params="stmp", t=1-1e-6, random_state=0, \
    init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6, tumorprop_threshold=0.5):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    if sample_ids is None:
        sample_ids = np.zeros(n_spots, dtype=int)
        n_samples = len(np.unique(sample_ids))
    else:
        unique_sample_ids = np.unique(sample_ids)
        n_samples = len(unique_sample_ids)
        tmp_map_index = {unique_sample_ids[i]:i for i in range(len(unique_sample_ids))}
        sample_ids = np.array([ tmp_map_index[x] for x in sample_ids])
    log_persample_weights = np.ones((n_clones, n_samples)) * (-np.log(n_clones))
    # pseudobulk
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop, threshold=tumorprop_threshold)
    # baseline proportion of UMI counts
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params, random_state=random_state, in_log_space=False, only_minor=False)
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
            sample_length = np.ones(X.shape[2],dtype=int) * X.shape[0]
            remain_kwargs = {"sample_length":sample_length, "lambd":lambd}
            if f"round{r-1}_log_gamma" in allres:
                remain_kwargs["log_gamma"] = allres[f"round{r-1}_log_gamma"]
            res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), n_states, \
                            # base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, \
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), np.repeat(tumor_prop, X.shape[0]).reshape(-1,1), \
                            hmmclass=hmmclass, params=params, t=t, random_state=random_state, \
                            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                            is_diag=is_diag, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol, **remain_kwargs)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrfmix_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrfmix_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res, \
                    smooth_mat, adjacency_mat, last_assignment, sample_ids, log_persample_weights, spatial_weight=spatial_weight, hmmclass=hmmclass)
            else:
                raise Exception("Unknown mode for nodepotential!")
            # handle the case when one clone has zero spots
            if len(np.unique(new_assignment)) < X.shape[2]:
                res["assignment_before_reindex"] = new_assignment
                remaining_clones = np.sort(np.unique(new_assignment))
                re_indexing = {c:i for i,c in enumerate(remaining_clones)}
                new_assignment = np.array([re_indexing[x] for x in new_assignment])
                concat_idx = np.concatenate([ np.arange(c*n_obs, c*n_obs+n_obs) for c in remaining_clones ])
                res["log_gamma"] = res["log_gamma"][:,concat_idx]
                res["pred_cnv"] = res["pred_cnv"][concat_idx]
            # add to results
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
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop, threshold=tumorprop_threshold)
        #
        if "mp" in params:
            print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        # if np.all( last_assignment == res["new_assignment"] ):
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99 or len(np.unique(res["new_assignment"])) == 1:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))
        for sidx in range(n_samples):
            index = np.where(sample_ids == sidx)[0]
            this_persample_weight = np.bincount(res["new_assignment"][index], minlength=X.shape[2]) / len(index)
            log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
            log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])


############################################################
# Final posterior using integer copy numbers
############################################################
@profile
def clonelabel_posterior_withinteger(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, state_cnv, res, pred, smooth_mat, adjacency_mat, prev_assignment, sample_ids, base_nb_mean, log_persample_weights, spatial_weight, hmmclass=hmm_sitewise):
    """
    single_X : array, (n_obs, 2, n_spots)

    single_base_nb_mean : array, (n_obs, n_spots)

    single_total_bb_RD : array, (n_obs, n_spots)

    single_tumor_prop : array, (n_spots,) or None

    state_cnv : DataFrame, (n_clones, n_states)

    adjacency_mat : sparse array, (n_spots, n_spots)

    prev_assignment : array, (n_spot,)

    sample_ids : array, (n_spots,)

    base_nb_mean : array, (n_obs, n_clones)

    log_persample_weights : array, (n_clones, n_samples)

    spatial_weight : float
    """
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    # clone IDs
    tmp_clone_ids = np.array([x[5:].split(" ")[0] for x in state_cnv.columns if x[:5] == "clone"])
    clone_ids = np.array([x for i,x in enumerate(tmp_clone_ids) if i == 0 or x != tmp_clone_ids[i-1]])
    n_clones = len(clone_ids)
    n_states = state_cnv.shape[0]
    # parameter based on integer copy numbers
    lambd = base_nb_mean / np.sum(base_nb_mean, axis=0, keepdims=True) if n_clones == base_nb_mean.shape[1] else base_nb_mean[:,1:] / np.sum(base_nb_mean[:,1:], axis=0, keepdims=True)
    log_mu_icn = np.zeros((n_states, n_clones))
    for c,cid in enumerate(clone_ids):
        log_mu_icn[:,c] = np.log( (state_cnv[f"clone{cid} A"] + state_cnv[f"clone{cid} B"]) / lambd[:,c].dot( (state_cnv[f"clone{cid} A"] + state_cnv[f"clone{cid} B"])[pred[:,c]] ) )
    p_binom_icn = np.array([ state_cnv[f"clone{cid} A"] / (state_cnv[f"clone{cid} A"] + state_cnv[f"clone{cid} B"]) for cid in clone_ids ]).T
    # handle 0 in p_binom_icn
    if n_clones == res["new_p_binom"].shape[1]:
        p_binom_icn[((p_binom_icn == 0) | (p_binom_icn == 1))] = res["new_p_binom"][((p_binom_icn == 0) | (p_binom_icn == 1))]
    elif n_clones + 1 == res["new_p_binom"].shape[1]:
        p_binom_icn[((p_binom_icn == 0) | (p_binom_icn == 1))] = res["new_p_binom"][:,1:][((p_binom_icn == 0) | (p_binom_icn == 1))]
    # over-dispersion
    new_alphas = copy.copy(res["new_alphas"]) if n_clones == res["new_p_binom"].shape[1] else copy.copy(res["new_alphas"][:,1:])
    new_alphas[:,:] = np.max(new_alphas)
    new_taus = copy.copy(res["new_taus"]) if n_clones == res["new_p_binom"].shape[1] else copy.copy(res["new_taus"][:,1:])
    new_taus[:,:] = np.min(new_taus)
    # result variables
    single_llf_rdr = np.zeros((N, n_clones))
    single_llf_baf = np.zeros((N, n_clones))
    single_llf = np.zeros((N, n_clones))
    df_posterior = pd.DataFrame({k:np.zeros(N) for k in [f"post_BAF_clone_{cid}" for cid in clone_ids] + [f"post_RDR_clone_{cid}" for cid in clone_ids] + \
                                 [f"post_nodellf_clone_{cid}" for cid in clone_ids] + [f"post_combine_clone_{cid}" for cid in clone_ids] })
    #
    for i in trange(N, desc="clonelabel_posterior_withinteger"):
        idx = smooth_mat[i,:].nonzero()[1]
        if not (single_tumor_prop is None):
            idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            if single_tumor_prop is None:
                tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), log_mu_icn[:,c:(c+1)], new_alphas[:,c:(c+1)], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), p_binom_icn[:,c:(c+1)], new_taus[:,c:(c+1)] )
            else:
                tmp_log_emission_rdr, tmp_log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), log_mu_icn[:,c:(c+1)], new_alphas[:,c:(c+1)], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), p_binom_icn[:,c:(c+1)], new_taus[:,c:(c+1)], np.repeat(np.mean(single_tumor_prop[idx]), single_X.shape[0]).reshape(-1,1) )
            assert not np.any(np.isnan(tmp_log_emission_rdr))
            assert not np.any(np.isnan(tmp_log_emission_baf))
            # !!! tmp_log_emission_baf may be NAN
            # Because LoH leads to Beta-binomial p = 0 or 1, but both A and B alleles are observed in the data, which leads to Nan.
            # We don't directly model the erroneous measurements associated with LoH.
            #
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                single_llf_rdr[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0])
                single_llf_baf[i,c] = np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
                single_llf[i,c] = single_llf_rdr[i,c] + single_llf_baf[i,c]
            else:
                single_llf_rdr[i,c] = np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0])
                single_llf_baf[i,c] = np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
                single_llf[i,c] = single_llf_rdr[i,c] + single_llf_baf[i,c]
        
        w_node = copy.copy(single_llf[i,:])
        w_node += log_persample_weights[:,sample_ids[i]]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            if n_clones == base_nb_mean.shape[1]:
                w_edge[prev_assignment[j]] += adjacency_mat[i,j]
            else:
                w_edge[prev_assignment[j] - 1] += adjacency_mat[i,j]
        #
        df_posterior.iloc[i,:n_clones] = np.exp( single_llf_baf[i,:] - scipy.special.logsumexp(single_llf_baf[i,:]) )
        df_posterior.iloc[i,n_clones:(2*n_clones)] = np.exp( single_llf_rdr[i,:] - scipy.special.logsumexp(single_llf_rdr[i,:]) )
        df_posterior.iloc[i,(2*n_clones):(3*n_clones)] = np.exp( single_llf[i,:] - scipy.special.logsumexp(single_llf[i,:]) )
        df_posterior.iloc[i,(3*n_clones):(4*n_clones)] = np.exp( w_node + spatial_weight * w_edge - scipy.special.logsumexp(w_node + spatial_weight * w_edge) )

    return df_posterior
