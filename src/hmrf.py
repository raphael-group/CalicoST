import logging
from turtle import reset
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


def infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, n_states, log_sitewise_transmat, \
    params, t, random_state, fix_NB_dispersion, shared_NB_dispersion, fix_BB_dispersion, shared_BB_dispersion, max_iter, tol):
    # pseudobulk HMM for phase_prob
    res = pipeline_baum_welch(None, np.sum(single_X, axis=2, keepdims=True), lengths, n_states, \
                              np.sum(single_base_nb_mean, axis=1, keepdims=True), np.sum(single_total_bb_RD, axis=1, keepdims=True), log_sitewise_transmat, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, consider_normal=False, \
                              shared_BB_dispersion_normal=True, is_diag=True, \
                              init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None, max_iter=max_iter, tol=tol)
    phase_prob = np.exp(scipy.special.logsumexp(res["log_gamma"][:n_states, :], axis=0))
    return phase_prob


def data_driven_initialize_initial_clone(single_X, single_total_bb_RD, phase_prob, n_states, n_clones, sorted_chr_pos, coords, random_state):
    ### arm-level BAF ###
    # smoothing based on adjacency
    centromere_file = "/u/congma/ragr-data/datasets/ref-genomes/centromeres/hg38.centromeres.txt"
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


def hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
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


def hmrf_reassignment_v2(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    phase_prob = np.exp(scipy.special.logsumexp(res["log_gamma"][:n_states, :], axis=0))
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ValueWarning)
        for i in trange(N):
            for c in range(n_clones):
                # estimate RDR dispersion and evaluate RDR probability
                idx_nonzero = np.where(single_base_nb_mean[:,i] > 0)[0]
                if len(idx_nonzero) > 0:
                    nb_mean = single_base_nb_mean[idx_nonzero, i] * np.exp(res["new_log_mu"][ res["pred_cnv"][idx_nonzero], c])
                    theta = theta_ml(single_X[idx_nonzero,0,i], nb_mean)
                    nb_std = np.sqrt(nb_mean + 1.0 / theta * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    single_llf[i,c] += np.sum(scipy.stats.nbinom.logpmf(single_X[idx_nonzero, 0, i], n, p))
                # estimate BAF dispersion and evaluate BAF probability
                idx_nonzero = np.where(single_total_bb_RD[:,i] > 0)[0]
                if len(idx_nonzero) > 0:
                    this_p_binom = res["new_p_binom"][res["pred_cnv"], c]
                    model = Weighted_BetaBinom_dispersiononly(endog=np.append(single_X[idx_nonzero, 1, i], single_total_bb_RD[idx_nonzero, i]-single_X[idx_nonzero, 1, i]), \
                        exog=np.ones(2*len(idx_nonzero)).reshape(-1,1), \
                        baf=np.append(this_p_binom[idx_nonzero], 1 - this_p_binom[idx_nonzero]), \
                        weights=np.append(phase_prob[idx_nonzero], 1-phase_prob[idx_nonzero]), \
                        exposure=np.append(single_total_bb_RD[idx_nonzero, i], single_total_bb_RD[idx_nonzero, i]))
                    modelfit = model.fit(disp=0, maxiter=500, xtol=1e-3, ftol=1e-3)
                    this_tau = modelfit.params[-1]
                    single_llf[i,c] += np.sum( scipy.stats.betabinom.logpmf(single_X[idx_nonzero,1,i], single_total_bb_RD[idx_nonzero,i], \
                        np.where(phase_prob[idx_nonzero] > 0.5, this_p_binom[idx_nonzero], 1-this_p_binom[idx_nonzero]) * this_tau, \
                        np.where(phase_prob[idx_nonzero] > 0.5, 1-this_p_binom[idx_nonzero], this_p_binom[idx_nonzero]) * this_tau) )
                    single_llf[i,c] += np.sum( scipy.stats.betabinom.logpmf(single_X[idx_nonzero,1,i], single_total_bb_RD[idx_nonzero,i], \
                        np.where(phase_prob[idx_nonzero] > 0.5, this_p_binom[idx_nonzero], 1-this_p_binom[idx_nonzero]) * res["new_taus"][0,c], \
                        np.where(phase_prob[idx_nonzero] > 0.5, 1-this_p_binom[idx_nonzero], this_p_binom[idx_nonzero]) * res["new_taus"][0,c]) )
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


def hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
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


def test_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        idx = adjacency_mat[i,:].nonzero()[1]
        idx = np.append(idx, np.array([i]))
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
                                                np.sum(single_base_nb_mean[:,idx], axis=1, keepdims=True), res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                np.sum(single_total_bb_RD[:,idx], axis=1, keepdims=True), res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
            single_llf[i,c] = np.sum(tmp_log_emission[pred, np.arange(n_obs), 0])
        w_node = single_llf[i,:]
        new_assignment[i] = np.argmax( w_node )

    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf


def hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
    # Note this is the old version without scalefactors
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = np.max(prev_assignment) + 1
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    for i in trange(N):
        tmp_log_emission = compute_emission_probability_nb_betabinom(single_X[:,:,i:(i+1)], \
                                            single_base_nb_mean[:,i:(i+1)], res["new_log_mu"], res["new_alphas"], \
                                            single_total_bb_RD[:,i:(i+1)], res["new_p_binom"], res["new_taus"])
        for c in range(n_clones):
            single_llf[i,c] = np.sum(tmp_log_emission[pred[(c*n_obs):(c*n_obs+n_obs)], np.arange(n_obs), 0])
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


def hmrf_reassignment_compositehmm(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight):
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
            res["new_p_binom"], res["new_taus"], res["new_scalefactors"])
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


def hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, \
    n_states, log_sitewise_transmat, coords=None, adjacency_mat=None, max_iter_outer=5, nodepotential="max", params="stmp", t=1-1e-6, random_state=0, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    consider_normal=False, shared_BB_dispersion_normal=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
    # spot adjacency matric
    assert not (coords is None and adjacency_mat is None)
    if adjacency_mat is None:
        adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    # pseudobulk
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
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
            res = pipeline_baum_welch(None, X, lengths, n_states, \
                              base_nb_mean, total_bb_RD, log_sitewise_transmat, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, consider_normal=consider_normal, \
                              shared_BB_dispersion_normal=shared_BB_dispersion_normal, is_diag=is_diag, \
                              init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
            pred = np.argmax(res["log_gamma"], axis=0)
            # clone assignmment
            if nodepotential == "max":
                new_assignment, single_llf, total_llf = hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
            elif nodepotential == "weighted_sum":
                new_assignment, single_llf, total_llf = hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
            elif nodepotential == "test_sum":
                new_assignment, single_llf, total_llf = test_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
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


def hmrf_concatenate_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, \
    n_states, log_sitewise_transmat, coords=None, max_iter_outer=5, params="stmp", t=1-1e-6, random_state=0, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    consider_normal=False, shared_BB_dispersion_normal=True, is_diag=True, \
    burn_in=5, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
    # spot adjacency matric
    assert not (coords is None)
    adjacency_mat = compute_adjacency_mat(coords, unit_xsquared, unit_ysquared)
    # pseudobulk
    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
    # initialize HMM parameters by GMM
    if ("m" in params) and ("p" in params):
        X_gmm = np.vstack([np.log(X[:,0,s]/base_nb_mean[:,s]) for s in range(X.shape[2])] + \
                   [X[:,1,s] / total_bb_RD[:,s] for s in range(X.shape[2])] ).T
    elif "m" in params:
        X_gmm = np.vstack([ np.log(X[:,0,s]/base_nb_mean[:,s]) for s in range(X.shape[2]) ]).T
    elif "p" in params:
        X_gmm = np.vstack([ X[:,1,s] / total_bb_RD[:,s] for s in range(X.shape[2]) ]).T
    X_gmm = X_gmm[np.sum(np.isnan(X_gmm), axis=1) == 0, :]
    gmm = GaussianMixture(n_components=n_states, max_iter=1, random_state=random_state).fit(X_gmm)
    # initialization parameters for HMM
    if ("m" in params) and ("p" in params):
        last_log_mu = gmm.means_[:,:X.shape[2]]
        last_p_binom = gmm.means_[:, X.shape[2]:]
    elif "m" in params:
        last_log_mu = gmm.means_
        last_p_binom = None
    elif "p" in params:
        last_log_mu = None
        last_p_binom = gmm.means_
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)
    for c,idx in enumerate(initial_clone_index):
        last_assignment[idx] = c
    # HMM
    for r in range(max_iter_outer):
        hmmmodel = hmm_sitewise(params=params, t=t)
        if not Path(f"{outdir}/round{r}_nstates{n_states}_{params}.npz").exists():
            if r < burn_in:
                res = pipeline_baum_welch(None, X, lengths, n_states, \
                              base_nb_mean, total_bb_RD, log_sitewise_transmat, params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                              fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, consider_normal=consider_normal, \
                              shared_BB_dispersion_normal=shared_BB_dispersion_normal, is_diag=is_diag, \
                              init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)

                pred = np.argmax(res["log_gamma"], axis=0)
                # clone assignmment
                new_assignment, single_llf, total_llf = hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
                res["prev_assignment"] = last_assignment
                res["new_assignment"] = new_assignment

                # save results
                np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)
                
            elif r == burn_in:
                res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), n_states, \
                              base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                              consider_normal=consider_normal, shared_BB_dispersion_normal=shared_BB_dispersion_normal, is_diag=is_diag, max_iter=max_iter, tol=tol)

                pred = np.argmax(res["log_gamma"], axis=0)
                # clone assignmment
                new_assignment, single_llf, total_llf = hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
                res["prev_assignment"] = last_assignment
                res["new_assignment"] = new_assignment
                # save results
                np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)
            else:
                res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), n_states, \
                              base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), params=params, t=t, random_state=random_state, \
                              fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                              consider_normal=consider_normal, shared_BB_dispersion_normal=shared_BB_dispersion_normal, is_diag=is_diag, \
                              init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=max_iter, tol=tol)
                pred = np.argmax(res["log_gamma"], axis=0)
                # clone assignmment
                new_assignment, single_llf, total_llf = hmrf_reassignment_concatenate(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                    adjacency_mat, last_assignment, spatial_weight=spatial_weight)
                res["prev_assignment"] = last_assignment
                res["new_assignment"] = new_assignment
                # save results
                np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)
        else:
            res = np.load(f"{outdir}/round{r}_nstates{n_states}_{params}.npz")

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)

        # update last parameter
        if r != burn_in:
            if "mp" in params:
                print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
            elif "m" in params:
                print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
            elif "p" in params:
                print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        if np.all( last_assignment == res["new_assignment"] ) and r > burn_in:
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]
