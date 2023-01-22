import logging
from turtle import reset
import numpy as np
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
from hmm_NB_BB_phaseswitch import *
from composite_hmm_NB_BB_phaseswitch import *
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


def compute_adjacency_mat_v2(coords, unit_xsquared=9, unit_ysquared=3, ratio=1):
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    # adjacency
    A = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.int8 )
    for i in range(coords.shape[0]):
        indexes = np.where(pairwise_squared_dist[i,:] <= ratio * (unit_xsquared + unit_ysquared))[0]
        indexes = np.array([j for j in indexes if j != i])
        if len(indexes) > 0:
            A[i, indexes] = 1
    A = scipy.sparse.csr_matrix(A)
    return A


def choose_adjacency_by_readcounts(coords, single_total_bb_RD, count_threshold=3000, unit_xsquared=9, unit_ysquared=3):
    # XXX: change from count_threshold 500 to 3000
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    tmp_pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    np.fill_diagonal(tmp_pairwise_squared_dist, np.max(tmp_pairwise_squared_dist))
    base_ratio = np.median(np.min(tmp_pairwise_squared_dist, axis=0)) / (unit_xsquared + unit_ysquared)
    selected_ratio = 0
    for ratio in range(0, 4):
        smooth_mat = compute_adjacency_mat_v2(coords, unit_xsquared, unit_ysquared, ratio * base_ratio)
        smooth_mat.setdiag(1)
        if np.median(smooth_mat.dot( np.sum(single_total_bb_RD, axis=0) )) > count_threshold:
            selected_ratio = ratio
            break
    for increment in np.arange(0.5, 10.5, 1):
        adjacency_mat = compute_adjacency_mat_v2(coords, unit_xsquared, unit_ysquared, (selected_ratio + increment) * base_ratio)
        adjacency_mat.setdiag(1)
        adjacency_mat = adjacency_mat - smooth_mat
        if np.sum(adjacency_mat) > 0:
            break
    sw_adjustment = 1.0 * np.median(np.sum(tmp_pairwise_squared_dist <= unit_xsquared + unit_ysquared, axis=0)) / np.median(np.sum(adjacency_mat.A, axis=0))
    return smooth_mat, adjacency_mat, sw_adjustment


def choose_adjacency_by_readcounts_slidedna(coords, single_total_bb_RD, maxknn=100, q=95, count_threshold=4):
    """
    Merge spots such that 95% quantile of read count per SNP per spot exceed count_threshold.
    """
    knnsize = 10
    for k in range(10, maxknn, 10):
        smooth_mat = kneighbors_graph(coords, n_neighbors=k)
        if np.percentile(smooth_mat.dot( single_total_bb_RD.T ), q) >= count_threshold:
            knnsize = k
            print(f"Picked spatial smoothing KNN K = {knnsize}")
            break
    adjacency_mat = kneighbors_graph(coords, n_neighbors=knnsize + 6)
    adjacency_mat = adjacency_mat - smooth_mat
    # sw_adjustment = 1.0 * 6 / np.median(np.sum(adjacency_mat.A, axis=0))
    sw_adjustment = 1
    return smooth_mat, adjacency_mat, sw_adjustment


def rectangle_initialize_initial_clone(coords, n_clones, random_state=0):
    np.random.seed(random_state)
    p = int(np.ceil(np.sqrt(n_clones)))
    # partition the range of x and y axes
    px = np.random.dirichlet( np.ones(p) * 10 )
    px[-1] += 1e-4
    xrange = [np.min(coords[:,0]), np.max(coords[:,0])]
    xdigit = np.digitize(coords[:,0], xrange[0] + (xrange[1] - xrange[0]) * np.cumsum(px), right=True)
    py = np.random.dirichlet( np.ones(p) * 10 )
    py[-1] += 1e-4
    yrange = [np.min(coords[:,1]), np.max(coords[:,1])]
    ydigit = np.digitize(coords[:,1], yrange[0] + (yrange[1] - yrange[0]) * np.cumsum(py), right=True)
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


def sample_initialize_initial_clone(adata, sample_list, n_clones, random_state=0):
    np.random.seed(random_state)
    occurences = 1 + np.random.multinomial(len(sample_list) - n_clones, pvals=np.ones(n_clones) / n_clones)
    sample_clone_id = sum([[i] * occurences[i] for i in range(len(occurences))], [])
    sample_clone_id = np.array(sample_clone_id)
    np.random.shuffle(sample_clone_id)
    print(sample_clone_id)
    clone_id = np.zeros(adata.shape[0], dtype=int)
    for i, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        clone_id[index] = sample_clone_id[i]
    print(np.bincount(clone_id))
    initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
    return initial_clone_index


def infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, n_states, log_sitewise_transmat, \
    params, t, random_state, fix_NB_dispersion, shared_NB_dispersion, fix_BB_dispersion, shared_BB_dispersion, max_iter, tol):
    # pseudobulk HMM for phase_prob
    res = pipeline_baum_welch(None, np.sum(single_X, axis=2, keepdims=True), lengths, n_states, \
                              np.sum(single_base_nb_mean, axis=1, keepdims=True), np.sum(single_total_bb_RD, axis=1, keepdims=True), log_sitewise_transmat, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, is_diag=True, \
                              init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None, max_iter=max_iter, tol=tol)
    # phase_prob = np.exp(scipy.special.logsumexp(res["log_gamma"][:n_states, :], axis=0))
    # return phase_prob
    pred = np.argmax(res["log_gamma"], axis=0)
    pred_cnv = pred % n_states
    phase_indicator = (pred < n_states)
    refined_lengths = []
    cumlen = 0
    for le in lengths:
        s = 0
        for i, k in enumerate(pred_cnv[cumlen:(cumlen+le)]):
            if i > 0 and pred_cnv[i] != pred_cnv[i-1]:
                refined_lengths.append(i - s)
                s = i
        refined_lengths.append(le - s)
        cumlen += le
    refined_lengths = np.array(refined_lengths)
    return phase_indicator, refined_lengths


def data_driven_initialize_initial_clone(single_X, single_total_bb_RD, phase_prob, n_states, n_clones, sorted_chr_pos, coords, random_state, genome_build="hg38"):
    ### arm-level BAF ###
    # smoothing based on adjacency
    if genome_build == "hg38":
        centromere_file = "/u/congma/ragr-data/datasets/ref-genomes/centromeres/hg38.centromeres.txt"
    elif genome_build == "hg19":
        centromere_file = "/u/congma/ragr-data/datasets/ref-genomes/centromeres/hg19.centromeres.txt"
    armlengths = get_lengths_by_arm(sorted_chr_pos, centromere_file)
    adjacency_mat = compute_adjacency_mat_v2(coords, ratio=10)
    smoothed_X_baf = single_X[:,1,:] @ adjacency_mat
    smoothed_total_bb_RD = single_total_bb_RD @ adjacency_mat
    # smoothed BAF
    chr_level_af = np.zeros((single_X.shape[2], len(armlengths)))
    for k,le in enumerate(armlengths):
        s = np.sum(armlengths[:k])
        t = s + le
        numer = phase_prob[s:t].dot(smoothed_X_baf[s:t,:]) + (1-phase_prob[s:t]).dot(smoothed_total_bb_RD[s:t,:] - smoothed_X_baf[s:t,:])
        denom = np.sum(smoothed_total_bb_RD[s:t,:], axis=0)
        chr_level_af[:,k] = numer / denom
    chr_level_af[np.isnan(chr_level_af)] = 0.5
    # Kmeans clustering based on BAF
    kmeans = KMeans(n_clusters=n_clones, random_state=random_state).fit(chr_level_af)
    initial_clone_index = [np.where(kmeans.labels_ == i)[0] for i in range(n_clones)]
    return initial_clone_index


def merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index):
    n_obs = single_X.shape[0]
    n_spots = len(clone_index)
    X = np.zeros((n_obs, 2, n_spots))
    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))

    for k,idx in enumerate(clone_index):
        X[:,:, k] = np.sum(single_X[:,:,idx], axis=2)
        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)

    return X, base_nb_mean, total_bb_RD


def hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], relative_rdr_weight)
            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred, np.arange(n_obs), 0])
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


def hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], relative_rdr_weight)
            if np.sum(single_base_nb_mean[:,i:(i+1)] > 0) > 0 and np.sum(single_total_bb_RD[:,i:(i+1)] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,i:(i+1)] > 0) / np.sum(single_base_nb_mean[:,i:(i+1)] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:, 0] + res["log_gamma"], axis=0) ) + np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:, 0] + res["log_gamma"], axis=0) )
            else:
                single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:, 0] + res["log_gamma"], axis=0) ) + np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:, 0] + res["log_gamma"], axis=0) )
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


def aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, smooth_mat, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        idx = smooth_mat[i,:].nonzero()[1]
        # idx = np.append(idx, np.array([i]))
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                                np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)], relative_rdr_weight)
            if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                ratio_nonzeros = 1.0 * np.sum(single_total_bb_RD[:,idx] > 0) / np.sum(single_base_nb_mean[:,idx] > 0)
                single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred, np.arange(n_obs), 0])
            else:
                single_llf[i,c] = np.sum(tmp_log_emission_rdr[pred, np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        # new_assignment[i] = np.argmax( w_node )
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2(single_X[:,:,i:(i+1)], \
                                            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], \
                                            single_total_bb_RD[:,i:(i+1)], res["new_p_binom"], res["new_taus"], relative_rdr_weight)
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

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrf_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2(single_X[:,:,i:(i+1)], \
                                            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], \
                                            single_total_bb_RD[:,i:(i+1)], res["new_p_binom"], res["new_taus"], relative_rdr_weight)
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


def aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, smooth_mat, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        idx = smooth_mat[i,:].nonzero()[1]
        # idx = np.append(idx, np.array([i]))
        tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                            np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"], res["new_alphas"], \
                                            np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"], res["new_taus"], relative_rdr_weight)
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

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrf_log_likelihood(nodepotential, single_X, single_base_nb_mean, single_total_bb_RD, res, pred, smooth_mat, adjacency_mat, assignment, spatial_weight):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_p_binom"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    #
    for i in trange(N):
        idx = smooth_mat[i,:].nonzero()[1] # smooth_mat can be identity matrix
        for c in range(n_clones):
            tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_v2( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
            #
            if nodepotential == "weighted_sum":
                if np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0) > 0 and np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) > 0:
                    ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                    single_llf[i,c] = ratio_nonzeros * np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:, 0] + res["log_gamma"][:,:,c], axis=0) ) + np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:, 0] + res["log_gamma"][:,:,c], axis=0) )
                else:
                    single_llf[i,c] = np.sum( scipy.special.logsumexp(tmp_log_emission_rdr[:,:,0] + res["log_gamma"][:,:,c], axis=0) ) + np.sum( scipy.special.logsumexp(tmp_log_emission_baf[:,:,0] + res["log_gamma"][:,:,c], axis=0) )
            else:
                if np.sum(single_base_nb_mean[:,idx] > 0) > 0 and np.sum(single_total_bb_RD[:,idx] > 0) > 0:
                    ratio_nonzeros = 1.0 * np.sum(np.sum(single_total_bb_RD[:,idx], axis=1) > 0) / np.sum(np.sum(single_base_nb_mean[:,idx], axis=1) > 0)
                    single_llf[i,c] = ratio_nonzeros * np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
                else:
                    single_llf[i,c] = np.sum(tmp_log_emission_rdr[pred[:,c], np.arange(n_obs), 0]) + np.sum(tmp_log_emission_baf[pred[:,c], np.arange(n_obs), 0])
    #
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(assignment[adjacency_mat[i,:].nonzero()[1]] == assignment[i]) )
    return total_llf


def hmrf_reassignment_compositehmm(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, relative_rdr_weight, spatial_weight):
    # basic dimension info
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_individual_states = int(len(res["new_p_binom"]) / 2.0)
    n_composite_states = int(len(res["state_tuples"]) / 2.0)
    
    # initialize result vector
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    # re-assign by HMRF
    for i in trange(N):
        # log emission probability of each composite state, matrix size (2*n_composite_states, n_obs)
        tmp_log_emission = compute_emission_probability_nb_betabinom_composite(single_X[:,:,i:(i+1)], res["state_tuples"], \
            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], single_total_bb_RD[:,i:(i+1)], \
            res["new_p_binom"], res["new_taus"], res["new_scalefactors"], relative_rdr_weight)
        for c in range(n_clones):
            single_llf[i,c] = np.sum(tmp_log_emission[pred[(c*n_obs):(c*n_obs+n_obs)], np.arange(n_obs)])
        # node potential
        w_node = single_llf[i,:]
        # edge potential
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i,:].nonzero()[1]:
            w_edge[new_assignment[j]] += 1
        # combine both potential for the new assignment
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
    
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def similarity_components_baf(baf_profiles, res, topk=10, threshold=0.05):
    n_clones = baf_profiles.shape[0]
    adj_baf_profiles = np.where(baf_profiles > 0.5, 1-baf_profiles, baf_profiles)
    G = nx.Graph()
    G.add_nodes_from( np.arange(n_clones) )
    for c1 in range(n_clones):
        for c2 in range(c1+1, n_clones):
            diff = np.sort(np.abs(baf_profiles[c1,:] - baf_profiles[c2,:]))[::-1][topk]
            adj_diff = np.sort(np.abs(adj_baf_profiles[c1,:] - adj_baf_profiles[c2,:]))[::-1][topk]
            if diff < 2*threshold and adj_diff < threshold:
                G.add_edge(c1, c2)
                print(c1, c2, diff)
    merging_groups = [cc for cc in nx.connected_components(G)]
    merging_groups.sort(key = lambda x:np.min(x))
    # clone assignment after merging
    map_clone_id = {}
    for i,x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    return merging_groups, merged_res


def similarity_components_rdrbaf(baf_profiles, rdr_profiles, res, topk=10, bafthreshold=0.05, rdrthreshold=0.1):
# def similarity_components_rdrbaf(baf_profiles, rdr_profiles, res, topk=10, bafthreshold=0.05, rdrthreshold=0.15):
    n_clones = baf_profiles.shape[0]
    adj_baf_profiles = np.where(baf_profiles > 0.5, 1-baf_profiles, baf_profiles)
    G = nx.Graph()
    G.add_nodes_from( np.arange(n_clones) )
    for c1 in range(n_clones):
        for c2 in range(c1+1, n_clones):
            baf_diff = np.sort(np.abs(baf_profiles[c1,:] - baf_profiles[c2,:]))[::-1][topk]
            baf_adj_diff = np.sort(np.abs(adj_baf_profiles[c1,:] - adj_baf_profiles[c2,:]))[::-1][topk]
            rdr_diff = np.sort(np.abs(rdr_profiles[c1,:] - rdr_profiles[c2,:]))[::-1][topk]
            if baf_diff < 2*bafthreshold and baf_adj_diff < bafthreshold and rdr_diff < rdrthreshold:
                G.add_edge(c1, c2)
    merging_groups = [cc for cc in nx.connected_components(G)]
    merging_groups.sort(key = lambda x:np.min(x))
    # clone assignment after merging
    map_clone_id = {}
    for i,x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    return merging_groups, merged_res


def similarity_components_baf_neymanpearson(X, base_nb_mean, total_bb_RD, res, sil_threshold=0, topk=10):
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    n_clones = len(np.unique(res["new_assignment"]))
    G = nx.Graph()
    G.add_nodes_from( np.arange(n_clones) )
    #
    def eval_neymanpearson(log_emission_baf_c1, pred_c1, log_emission_baf_c2, pred_c2, bidx, n_states, res, p):
        # Pearson residual vectors under the corresponding state
        llf_original = np.append(log_emission_baf_c1[pred_c1[bidx], bidx], log_emission_baf_c2[pred_c2[bidx], bidx]).reshape(-1,1)
        # Pearson residual vectors under the switched state
        if (res["new_p_binom"][p[0],0] > 0.5) == (res["new_p_binom"][p[1],0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
        llf_switch = np.append(log_emission_baf_c1[switch_pred_c1[bidx], bidx], log_emission_baf_c2[switch_pred_c2[bidx], bidx]).reshape(-1,1)
        # silhouette score
        return np.mean(llf_original) - np.mean(llf_switch)
    #
    log_emission_rdr, log_emission_baf = compute_emission_probability_nb_betabinom_v2(np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
        base_nb_mean.flatten("F").reshape(-1,1), res["new_log_mu"], res["new_alphas"], \
        total_bb_RD.flatten("F").reshape(-1,1), res["new_p_binom"], res["new_taus"])
    log_emission_rdr = log_emission_rdr.reshape((log_emission_rdr.shape[0], n_obs, n_clones), order="F")
    log_emission_baf = log_emission_baf.reshape((log_emission_baf.shape[0], n_obs, n_clones), order="F")
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2],-1))
    reshaped_pred_cnv = reshaped_pred % n_states
    for c1 in range(n_clones):
        for c2 in range(n_clones):
            unmergeable_bincount = 0
            unique_pair_states = [x for x in np.unique(reshaped_pred_cnv[np.array([c1,c2]), :], axis=1).T if x[0] != x[1]]
            for p in unique_pair_states:
                bidx = np.where( (reshaped_pred_cnv[c1,:]==p[0]) & (reshaped_pred_cnv[c2,:]==p[1]) )[0]
                t_neymanpearson = eval_neymanpearson(log_emission_baf[:,:,c1], reshaped_pred[c1,:], log_emission_baf[:,:,c2], reshaped_pred[c2,:], bidx, n_states, res, p)
                print(p, len(bidx), t_neymanpearson)
                if t_neymanpearson > sil_threshold:
                    unmergeable_bincount += len(bidx)
            if unmergeable_bincount < topk:
                G.add_edge(c1, c2)
    merging_groups = [cc for cc in nx.connected_components(G)]
    merging_groups.sort(key = lambda x:np.min(x))
    # clone assignment after merging
    map_clone_id = {}
    for i,x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    return NotImplemented


def similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, sil_threshold=0.1, topk=10):
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    n_clones = len(np.unique(res["new_assignment"]))
    G = nx.Graph()
    G.add_nodes_from( np.arange(n_clones) )
    #
    def eval_neymanpearson(log_emission_rdr_c1, log_emission_baf_c1, pred_c1, log_emission_rdr_c2, log_emission_baf_c2, pred_c2, bidx, n_states, res, p):
        # Pearson residual vectors under the corresponding state
        llf_original = np.append(log_emission_rdr_c1[pred_c1[bidx], bidx] + log_emission_baf_c1[pred_c1[bidx], bidx], \
            log_emission_rdr_c2[pred_c2[bidx], bidx] + log_emission_baf_c2[pred_c2[bidx], bidx]).reshape(-1,1)
        # Pearson residual vectors under the switched state
        if (res["new_p_binom"][p[0],0] > 0.5) == (res["new_p_binom"][p[1],0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
        llf_switch = np.append(log_emission_rdr_c1[switch_pred_c1[bidx], bidx] + log_emission_baf_c1[switch_pred_c1[bidx], bidx], \
            log_emission_rdr_c2[switch_pred_c2[bidx], bidx] + log_emission_baf_c2[switch_pred_c2[bidx], bidx]).reshape(-1,1)
        # silhouette score
        return np.mean(llf_original) - np.mean(llf_switch)
    #
    log_emission_rdr, log_emission_baf = compute_emission_probability_nb_betabinom_v2(np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
        base_nb_mean.flatten("F").reshape(-1,1), res["new_log_mu"], res["new_alphas"], \
        total_bb_RD.flatten("F").reshape(-1,1), res["new_p_binom"], res["new_taus"])
    log_emission_rdr = log_emission_rdr.reshape((log_emission_rdr.shape[0], n_obs, n_clones), order="F")
    log_emission_baf = log_emission_baf.reshape((log_emission_baf.shape[0], n_obs, n_clones), order="F")
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2],-1))
    reshaped_pred_cnv = reshaped_pred % n_states
    for c1 in range(n_clones):
        for c2 in range(n_clones):
            unmergeable_bincount = 0
            unique_pair_states = [x for x in np.unique(reshaped_pred_cnv[np.array([c1,c2]), :], axis=1).T if x[0] != x[1]]
            for p in unique_pair_states:
                bidx = np.where( (reshaped_pred_cnv[c1,:]==p[0]) & (reshaped_pred_cnv[c2,:]==p[1]) )[0]
                t_neymanpearson = eval_neymanpearson(log_emission_rdr[:,:,c1], log_emission_baf[:,:,c1], reshaped_pred[c1,:], log_emission_rdr[:,:,c2], log_emission_baf[:,:,c2], reshaped_pred[c2,:], bidx, n_states, res, p)
                print(p, len(bidx), t_neymanpearson)
                if t_neymanpearson > sil_threshold:
                    unmergeable_bincount += len(bidx)
            if unmergeable_bincount < topk:
                G.add_edge(c1, c2)
    merging_groups = [cc for cc in nx.connected_components(G)]
    merging_groups.sort(key = lambda x:np.min(x))
    # clone assignment after merging
    map_clone_id = {}
    for i,x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    return merged_res


def merge_by_minspots(assignment, res, min_spots_thresholds=50, single_tumor_prop=None, threshold=0.5):
    n_clones = len(np.unique(assignment))
    new_assignment = copy.copy(assignment)
    merging_groups = [[i] for i in range(n_clones)]
    if single_tumor_prop is None:
        tmp_single_tumor_prop = np.array([1] * len(assignment))
    else:
        tmp_single_tumor_prop = single_tumor_prop
    while np.min(np.bincount(new_assignment[tmp_single_tumor_prop > threshold])) < min_spots_thresholds:
        idx_min = np.argmin(np.bincount(new_assignment[tmp_single_tumor_prop > threshold]))
        idx_max = np.argmax(np.bincount(new_assignment[tmp_single_tumor_prop > threshold]))
        merging_groups = [ [i] for i in range(n_clones) if (i!=idx_min) and (i!=idx_max)] + [[idx_min, idx_max]]
        merging_groups.sort(key = lambda x:np.min(x))
        # clone assignment after merging
        map_clone_id = {}
        for i,x in enumerate(merging_groups):
            for z in x:
                map_clone_id[z] = i
        new_assignment = np.array([map_clone_id[x] for x in new_assignment])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    return merging_groups, merged_res


# def bic_merge_modelselection_concatenate(single_X, lengths, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, smooth_mat, adjacency_mat, \
#     res, params, t, fix_NB_dispersion, shared_NB_dispersion, fix_BB_dispersion, shared_BB_dispersion, max_iter, tol, relative_rdr_weight, spatial_weight):
#     """
#     Checking whether the "closest" two clones should be merged under BIC. The degrees of freedom in BIC include number of hidden states across all clones (either original clone partition or after merging).

#     Attributes
#     ----------
#     single_X : array, shape (n_observations, n_components, n_spots)
#         Observed expression UMI count and allele frequency UMI count per single spot.

#     single_base_nb_mean : array, shape (n_observations, n_spots)
#         Mean expression under diploid state.

#     single_total_bb_RD : array, array, shape (n_observations, n_spots)
#         SNP-covering reads for both REF and ALT across genes along genome.

#     res : dictionary
#         HMM parameters including "new_log_mu", "new_alphas", "new_p_binom", "new_taus", etc.

#     smooth_mat, adjacency_mat : sparse matrix, shape (n_spots, n_spots)
#         Matrix to indicate which spots to pool for enhancing signal, and which spots to consider for HMRF edge potential.

#     prev_assignment : array, shape (n_spots,)
#         Clone assignment of last iteration.

#     Returns
#     ----------
#     merge_performed : boolean
#         Indicator for whether merging is performed under BIC criteria. For more details on BIC of HMRF.
#         See https://www.proquest.com/docview/304538216?pq-origsite=gscholar&fromopenview=true p75 and https://ieeexplore.ieee.org/document/1227985 equation (20).

#     original_bic : float
#         BIC under the original model

#     merged_bic :float
#         BIC under the merged model

#     merged_res : None or dictonary
#         Dictonary will include keys 'new_log_mu', 'new_alphas', 'new_p_binom', 'new_taus', 'new_log_startprob', 'new_log_transmat', 'log_gamma', 'pred_cnv', 'llf', 'prev_assignment', 'new_assignment', 'total_llf'.    
#     """
#     N = single_X.shape[2]
#     n_obs = single_X.shape[0]
#     n_clones = int(len(res["pred_cnv"]) / n_obs)
#     n_states = res["new_p_binom"].shape[0]
#     original_bic = -2 * res["total_llf"] + (n_clones * n_obs + n_states * 2) * np.log(single_X.shape[0] * single_X.shape[2])
#     #
#     # determine the "closest" two clones
#     baf_profiles = np.array([ res["new_p_binom"][res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)], 0] for c in range(n_clones) ])
#     pdistances = scipy.spatial.distance.squareform( scipy.spatial.distance.pdist(baf_profiles) )
#     np.fill_diagonal(pdistances, np.max(pdistances)+1)
#     closest_clones = np.unravel_index(np.argmin(pdistances), pdistances.shape)
#     #
#     # re-estimating parameters under the merged clone pseudobulk
#     tmpmap_clone_label = {c:c if c <= max(closest_clones) else c-1 for c in range(n_clones)}
#     tmpmap_clone_label[max(closest_clones)] = min(closest_clones)
#     tmpprev_assignment = np.array([ tmpmap_clone_label[x] for x in res["new_assignment"] ])
#     tmpclone_index = [np.where(tmpprev_assignment == c)[0] for c in range(len(tmpmap_clone_label) - 1)]
#     # re-assigning spots to clones under re-estimated parameters and tmpprev_assignment
#     tmpX, tmpbase_nb_mean, tmptotal_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, tmpclone_index)
#     tmpres = pipeline_baum_welch(None, np.vstack([tmpX[:,0,:].flatten("F"), tmpX[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, tmpX.shape[2]), n_states, \
#         tmpbase_nb_mean.flatten("F").reshape(-1,1), tmptotal_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, tmpX.shape[2]), params=params, t=t, random_state=0, \
#         fix_NB_dispersion=True, shared_NB_dispersion=True, fix_BB_dispersion=True, shared_BB_dispersion=True, \
#         init_log_mu=res["new_log_mu"], init_p_binom=res["new_p_binom"], init_alphas=res["new_alphas"], init_taus=res["new_taus"], \
#         is_diag=True, relative_rdr_weight=relative_rdr_weight, max_iter=max_iter, tol=tol)
#     # re-assigning spots to clones under re-estimated parameters and tmpprev_assignment
#     tmppred = np.argmax(tmpres["log_gamma"], axis=0)
#     tmpnew_assignment, _, tmptotal_llf = aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, tmpres, tmppred, \
#         smooth_mat, adjacency_mat, tmpprev_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
#     # compute merged bic
#     merged_bic = -2 * tmptotal_llf + ((n_clones - 1) * n_obs + n_states * 2) * np.log(single_X.shape[0] * single_X.shape[2])
#     #
#     # decide whether to merge clone
#     if merged_bic < original_bic:
#         merge_performed = True
#         merged_res = {'new_log_mu':tmpres["new_log_mu"], 'new_alphas':tmpres["new_alphas"], 'new_p_binom':tmpres["new_p_binom"], 'new_taus':tmpres["new_taus"], 'new_log_startprob':tmpres["new_log_startprob"], \
#             'new_log_transmat':tmpres["new_log_transmat"], 'log_gamma':tmpres["log_gamma"], 'pred_cnv':tmppred%n_states, 'llf':tmpres["llf"], 'prev_assignment':tmpprev_assignment, 'new_assignment':tmpnew_assignment, 'total_llf':tmptotal_llf}
#     else:
#         merge_performed = False
#         merged_res = None
#     return merge_performed, merged_res, original_bic, merged_bic


# def silhouette_merge_modelselection_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, smooth_mat=None):
#     """
#     Checking whether the "closest" two clones should be merged under BIC. The degrees of freedom in BIC include number of hidden states across all clones (either original clone partition or after merging).

#     Attributes
#     ----------
#     single_X : array, shape (n_observations, n_components, n_spots)
#         Observed expression UMI count and allele frequency UMI count per single spot.

#     single_base_nb_mean : array, shape (n_observations, n_spots)
#         Mean expression under diploid state.

#     single_total_bb_RD : array, array, shape (n_observations, n_spots)
#         SNP-covering reads for both REF and ALT across genes along genome.

#     res : dictionary
#         HMM parameters including "new_log_mu", "new_alphas", "new_p_binom", "new_taus", etc.

#     smooth_mat, adjacency_mat : sparse matrix, shape (n_spots, n_spots)
#         Matrix to indicate which spots to pool for enhancing signal, and which spots to consider for HMRF edge potential.

#     prev_assignment : array, shape (n_spots,)
#         Clone assignment of last iteration.

#     Returns
#     ----------
#     merge_performed : boolean
#         Indicator for whether merging is performed under BIC criteria. For more details on BIC of HMRF.
#         See https://www.proquest.com/docview/304538216?pq-origsite=gscholar&fromopenview=true p75 and https://ieeexplore.ieee.org/document/1227985 equation (20).

#     original_bic : float
#         BIC under the original model

#     merged_bic :float
#         BIC under the merged model

#     merged_res : None or dictonary
#         Dictonary will include keys 'new_log_mu', 'new_alphas', 'new_p_binom', 'new_taus', 'new_log_startprob', 'new_log_transmat', 'log_gamma', 'pred_cnv', 'llf', 'prev_assignment', 'new_assignment', 'total_llf'.    
#     """
#     N = single_X.shape[2]
#     n_obs = single_X.shape[0]
#     n_clones = int(len(res["pred_cnv"]) / n_obs)
#     n_states = res["new_p_binom"].shape[0]
#     # convert concatenated HMM states to matrix of (n_obs, n_clones)
#     pred_mat = np.zeros((n_obs, n_clones), dtype=int)
#     for c in range(n_clones):
#         pred_mat[:, c] = res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)]
#     # unique composite states
#     uniq_pred_mat = np.unique(pred_mat, axis=0)

#     # enhance count signal by smooth_mat
#     if not smooth_mat is None:
#         tmpsingle_X = np.zeros(single_X.shape)
#         tmpsingle_X[:,0,:] = single_X[:,0,:] @ smooth_mat
#         tmpsingle_X[:,1,:] = single_X[:,1,:] @ smooth_mat
#         tmpsingle_base_nb_mean = single_base_nb_mean @ smooth_mat
#         tmpsingle_total_bb_RD = single_total_bb_RD @ smooth_mat
#         # keep only non-overlapping windows
#         idx_nonoverlapping = []
#         idx_overlap = set()
#         for i in range(N):
#             if not (i in idx_overlap):
#                 idx_nonoverlapping.append(i)
#                 idx_overlap = idx_overlap | set(list( smooth_mat[i,:].nonzero()[1] ))
#         idx_nonoverlapping = np.array(idx_nonoverlapping)
#         tmpsingle_X = tmpsingle_X[:, :, idx_nonoverlapping]
#         tmpsingle_base_nb_mean = tmpsingle_base_nb_mean[:, idx_nonoverlapping]
#         tmpsingle_total_bb_RD = tmpsingle_total_bb_RD[:, idx_nonoverlapping]
#         tmpnew_assignment = res["new_assignment"][idx_nonoverlapping]
#     else:
#         idx_nonoverlapping = np.arange(N)

#     # aggregate counts for each composite state for each spot
#     n_composite_states = uniq_pred_mat.shape[0]
#     per_state_counts = np.zeros((n_composite_states, len(idx_nonoverlapping)))
#     per_state_nb_mean = np.zeros((n_composite_states, len(idx_nonoverlapping)))
#     per_state_Ballele = np.zeros((n_composite_states, len(idx_nonoverlapping)))
#     per_state_total = np.zeros((n_composite_states, len(idx_nonoverlapping)))
#     for i,statevec in enumerate(uniq_pred_mat):
#         idx = np.where( np.sum(uniq_pred_mat - statevec.reshape(1,-1), axis=1) == n_clones )[0]
#         per_state_counts[i,:] = np.sum(tmpsingle_X[idx,0,:], axis=0)
#         per_state_nb_mean[i,:] = np.sum(tmpsingle_base_nb_mean[idx,:], axis=0)
#         for c in range(n_clones):
#             idx_spots = np.where(tmpnew_assignment == c)[0]
#             s = c * n_obs
#             this_phase = (np.argmax(res["log_gamma"][:, s+idx], axis=0) < n_states)
#             per_state_Ballele[i, idx_spots] = this_phase.dot(tmpsingle_X[idx, 1, :][:, idx_spots]) + (1-this_phase).dot(tmpsingle_total_bb_RD[idx,:][:,idx_spots] - tmpsingle_X[idx, 1, :][:, idx_spots])
#         per_state_total[i,:] = np.sum(tmpsingle_total_bb_RD[idx,:], axis=0)
    
#     # silhouette score
#     X = np.vstack([np.log1p(1.0 * per_state_counts / per_state_nb_mean), 1.0 * per_state_Ballele / per_state_total]).T
#     X[np.isnan(X)] = 0
#     score = silhouette_score(X, labels=tmpnew_assignment)
#     return score


def hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, \
    n_states, log_sitewise_transmat, coords=None, smooth_mat=None, adjacency_mat=None, max_iter_outer=5, nodepotential="max", params="stmp", t=1-1e-6, random_state=0, \
    init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, relative_rdr_weight=0.3, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    # pseudobulk
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
    # initialize HMM parameters by GMM
    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params, random_state=random_state, in_log_space=False, remove_baf_zero=False)
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
            res = pipeline_baum_welch(None, X, lengths, n_states, \
                              base_nb_mean, total_bb_RD, log_sitewise_transmat, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                              relative_rdr_weight=relative_rdr_weight, is_diag=is_diag, \
                              init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, \
                    adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            # elif nodepotential == "test_sum":
            #     new_assignment, single_llf, total_llf = aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
            #         smooth_mat, adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
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
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)

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


def hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states, log_sitewise_transmat, \
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
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
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
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), params=params, t=t, random_state=random_state, \
                            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                            is_diag=is_diag, relative_rdr_weight=relative_rdr_weight, \
                            init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    smooth_mat, adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, \
                    adjacency_mat, last_assignment, relative_rdr_weight, spatial_weight=spatial_weight)
            # elif nodepotential == "test_sum":
            #     new_assignment, single_llf, total_llf = aggr_hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
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
        if adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
