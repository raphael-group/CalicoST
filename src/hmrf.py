import logging
import numpy as np
from numba import njit
import scipy.special
import scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from tqdm import trange
import copy
from pathlib import Path
from hmm_NB_BB_phaseswitch import *
from utils_distribution_fitting import *


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


def bin_locus(single_X, single_base_nb_mean, single_total_bb_RD, n_states, pred_cnv, pred, binsize):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    # loop over snp locus to determine whether to merge to one bin
    multiply_vec = []
    binned_pred_cnv = []
    s = 0
    while s < n_obs:
        if np.all(pred_cnv[s:] == pred_cnv[s]):
            t = len(pred_cnv)
        else:
            t = s + np.where(pred_cnv[s:] != pred_cnv[s])[0][0]
        m = np.zeros(n_obs)
        m[s:min(s+binsize,t)] = 1
        multiply_vec.append(m)
        binned_pred_cnv.append( pred_cnv[s] )
        s = min(s+binsize,t)
    multiply_vec = np.vstack(multiply_vec)
    binned_pred_cnv = np.array(binned_pred_cnv)
    # phase single_X
    tmp_X = np.where(pred.reshape(-1,1) < n_states, single_X[:,1,:], single_total_bb_RD - single_X[:,1,:])
    # actually binning
    new_n_obs = multiply_vec.shape[0]
    binned_single_X = np.zeros((new_n_obs, 2, N))
    binned_single_X[:,0,:] = multiply_vec @ single_X[:,0,:]
    binned_single_X[:,1,:] = multiply_vec @ tmp_X
    binned_single_base_nb_mean = multiply_vec @ single_base_nb_mean
    binned_single_total_bb_RD = multiply_vec @ single_total_bb_RD
    return binned_single_X, binned_single_base_nb_mean, binned_single_total_bb_RD, binned_pred_cnv


def naive_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    single_llf = np.zeros((N, n_clones))

    for i in trange(N):
        for c in range(n_clones):
            tmp_log_emission = compute_emission_probability_nb_betabinom(single_X[:,:,i:(i+1)], \
                                                single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
                                                single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
            single_llf[i,c] = np.sum(tmp_log_emission[pred, np.arange(n_obs), 0])

    return np.argmax(single_llf, axis=1), single_llf


# def hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, adjacency_mat, prev_assignment, spatial_weight=1.0/6):
#     N = single_X.shape[2]
#     n_obs = single_X.shape[0]
#     n_clones = res["new_log_mu"].shape[1]
#     single_llf = np.zeros((N, n_clones))
#     new_assignment = copy.copy(prev_assignment)

#     for i in trange(N):
#         for c in range(n_clones):
#             tmp_log_emission = compute_emission_probability_nb_betabinom(single_X[:,:,i:(i+1)], \
#                                                 single_base_nb_mean[:,i:(i+1)], res["new_log_mu"][:,c:(c+1)], res["new_alphas"][:,c:(c+1)], \
#                                                 single_total_bb_RD[:,i:(i+1)], res["new_p_binom"][:,c:(c+1)], res["new_taus"][:,c:(c+1)])
#             single_llf[i,c] = np.sum(tmp_log_emission[pred, np.arange(n_obs), 0])
#         w_node = single_llf[i,:]
#         w_edge = np.zeros(n_clones)
#         for j in adjacency_mat[i,:].nonzero()[1]:
#             w_edge[new_assignment[j]] += 1
#         new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )

#     # compute total log likelihood log P(X | Z) + log P(Z)
#     total_llf = np.sum(single_llf[np.arange(N), new_assignment])
#     for i in range(N):
#         total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
#     return new_assignment, single_llf, total_llf


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


def regionwise_likelihood(single_X, single_base_nb_mean, single_total_bb_RD, res, min_length=300):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_p_binom"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    phase_prob = np.exp(scipy.special.logsumexp(res["log_gamma"][:n_states, :], axis=0))
    phase_indicator = (phase_prob > 0.5) # 1 means state 0:n_states; 0 means state n_states:2*n_states

    def get_regions(pred, min_length):
        all_regions = []
        s = 0
        t = s+1
        while s < len(pred):
            if np.all(pred[s:] == pred[s]):
                if len(pred) - s >= min_length:
                    all_regions.append( (s, len(pred)) )
                break
            t = s + np.where(pred[s:] != pred[s])[0][0]
            if t - s >= min_length:
                all_regions.append( (s,t) )
            s = t
        return all_regions

    regions = get_regions(res["pred_cnv"], min_length=min_length)
    collapsed_new_log_mu = res["new_log_mu"].flatten().reshape(-1,1)
    collased_new_alphas = res["new_alphas"].flatten().reshape(-1,1)
    collapsed_new_p_binom = res["new_p_binom"].flatten().reshape(-1,1)
    collapsed_new_taus = res["new_taus"].flatten().reshape(-1,1)

    collapsed_llf = np.zeros((N, len(regions), collapsed_new_p_binom.shape[0] ))
    for i in trange(N):
        for ind_r, r in enumerate(regions):
            tmp = compute_emission_probability_nb_betabinom(single_X[r[0]:r[1],:,i:(i+1)], \
                                        single_base_nb_mean[r[0]:r[1],i:(i+1)], collapsed_new_log_mu, collased_new_alphas, \
                                        single_total_bb_RD[r[0]:r[1],i:(i+1)], collapsed_new_p_binom, collapsed_new_taus)
            tmp2 = tmp[:collapsed_new_p_binom.shape[0], :]
            tmp2[:, ~phase_indicator[r[0]:r[1]]] = tmp[collapsed_new_p_binom.shape[0]:, ~phase_indicator[r[0]:r[1]]]
            tmp2 = np.sum(tmp2, axis=1)
            tmp2 = tmp2 - scipy.special.logsumexp(tmp2)
            collapsed_llf[i, ind_r, :] = tmp2.flatten()
    return collapsed_llf


def clustering_regionwise_likelihood(collapsed_llf, res, last_assignment):
    n_clones = res["new_p_binom"].shape[1]
    tmp = np.exp(collapsed_llf.reshape((collapsed_llf.shape[0], -1)))
    tmp = tmp / np.sum(tmp, axis=1, keepdims=True)
    precomputed_distance2 = scipy.spatial.distance.cdist(tmp, tmp, metric="jensenshannon")
    # hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clones, affinity="precomputed", linkage="complete").fit(precomputed_distance2)
    # reorder the cluster IDs to best match original one
    import networkx as nx
    B = nx.Graph()
    B.add_nodes_from([f"A{i}" for i in range(n_clones)], bipartite=0)
    B.add_nodes_from([f"B{i}" for i in range(n_clones)], bipartite=1)
    B.add_weighted_edges_from([(f"A{i}", f"B{j}", np.sum(np.logical_and(last_assignment==i, clustering.labels_==j))) for i in range(n_clones) for j in range(n_clones)])
    match = nx.bipartite.maximum_matching(B)
    map_labels = {int(k[1:]):int(v[1:]) for k,v in match.items() if k[0] == "B"}
    return np.array([map_labels[x] for x in clustering.labels_])


def hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, \
    n_states, log_sitewise_transmat, coords=None, max_iter_outer=5, params="stmp", t=1-1e-6, random_state=0, init_alphas=None, init_taus=None,\
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    consider_normal=False, shared_BB_dispersion_normal=True, \
    is_diag=True, max_iter=100, tol=1e-4, unit_xsquared=9, unit_ysquared=3, spatial_weight=1.0/6):
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
            ##### initialize with the parameters of last iteration #####
            new_log_mu, new_alphas, new_p_binom, new_taus, new_log_startprob, new_log_transmat = hmmmodel.run_baum_welch_nb_bb_sitewise(X, lengths, \
                n_states, base_nb_mean, total_bb_RD, log_sitewise_transmat, \
                fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                consider_normal=consider_normal, shared_BB_dispersion_normal=shared_BB_dispersion_normal, \
                is_diag=is_diag, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, \
                max_iter=max_iter, tol=tol)

            # compute posterior and prediction
            log_gamma = posterior_nb_bb_sitewise(X, lengths, \
                                            base_nb_mean, new_log_mu, new_alphas, \
                                            total_bb_RD, new_p_binom, new_taus, \
                                            new_log_startprob, new_log_transmat, log_sitewise_transmat)
            pred = np.argmax(log_gamma, axis=0)
            pred_cnv = pred % n_states
            
            # likelihood
            log_emission = compute_emission_probability_nb_betabinom(X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus)
            log_alpha = forward_lattice_sitewise(lengths, new_log_transmat, new_log_startprob, log_emission, log_sitewise_transmat)
            llf = np.sum(scipy.special.logsumexp(log_alpha[:,lengths-1], axis=0))

            # ##### initialize with gmm parameters #####
            # if r > 1:
            #     gmm_log_mu, gmm_p_binom = initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params)
            #     new_log_mu2, new_alphas2, new_p_binom2, new_taus2, new_log_startprob2, new_log_transmat2 = hmmmodel.run_baum_welch_nb_bb_sitewise(X, lengths, \
            #         n_states, base_nb_mean, total_bb_RD, log_sitewise_transmat, \
            #         fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
            #         fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
            #         consider_normal=consider_normal, shared_BB_dispersion_normal=shared_BB_dispersion_normal, \
            #         is_diag=is_diag, init_log_mu=gmm_log_mu, init_p_binom=gmm_p_binom, init_alphas=last_alphas, init_taus=last_taus, \
            #         max_iter=max_iter, tol=tol)

            #     # compute posterior and prediction
            #     log_gamma2 = posterior_nb_bb_sitewise(X, lengths, \
            #                                     base_nb_mean, new_log_mu2, new_alphas2, \
            #                                     total_bb_RD, new_p_binom2, new_taus2, \
            #                                     new_log_startprob2, new_log_transmat2, log_sitewise_transmat)
            #     pred2 = np.argmax(log_gamma2, axis=0)
            #     pred_cnv2 = pred2 % n_states
                
            #     # likelihood
            #     log_emission2 = compute_emission_probability_nb_betabinom(X, base_nb_mean, new_log_mu2, new_alphas2, total_bb_RD, new_p_binom2, new_taus2)
            #     log_alpha2 = forward_lattice_sitewise(lengths, new_log_transmat2, new_log_startprob2, log_emission2, log_sitewise_transmat)
            #     llf2 = np.sum(scipy.special.logsumexp(log_alpha2[:,lengths-1], axis=0))

            #     ##### compare the two initializations #####
            #     if llf2 > llf:
            #         print("picking GMM initialization parameters in HMM.")
            #         new_log_mu = new_log_mu2; new_alphas = new_alphas2; new_p_binom = new_p_binom2; new_taus = new_taus2
            #         new_log_startprob = new_log_startprob2; new_log_transmat = new_log_transmat2
            #         log_gamma = log_gamma2; pred = pred2; pred_cnv = pred_cnv2
            #         log_emission = log_emission2; log_alpha = log_alpha2; llf = llf2

            # clone assignmment
            res = {"new_log_mu":new_log_mu, "new_alphas":new_alphas, "new_p_binom":new_p_binom, "new_taus":new_taus, \
                "new_log_startprob":new_log_startprob, "new_log_transmat":new_log_transmat, "log_gamma":log_gamma, "pred_cnv":pred_cnv, "llf":llf, "prev_assignment":last_assignment}
            new_assignment, single_llf, total_llf = hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res, pred, \
                adjacency_mat, last_assignment, spatial_weight=spatial_weight)
            res["new_assignment"] = new_assignment

            # save results
            np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)

        else:
            res = np.load(f"{outdir}/round{r}_nstates{n_states}_{params}.npz")

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in np.sort(np.unique(res["new_assignment"]))]
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)

        # update last parameter
        if "mp" in params:
            print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        if np.all( last_assignment == res["new_assignment"] ):
            break
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]


def regionwise_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, \
    n_states, log_sitewise_transmat, max_iter_outer=5, params="stmp", t=1-1e-6, random_state=0, \
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True, \
    is_diag=True, max_iter=100, tol=1e-4):
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
    last_alphas = None
    last_taus = None
    last_assignment = np.zeros(single_X.shape[2], dtype=int)
    for c,idx in enumerate(initial_clone_index):
        last_assignment[idx] = c
    # HMM
    for r in range(max_iter_outer):
        hmmmodel = hmm_sitewise(params=params, t=t)
        if not Path(f"{outdir}/round{r}_nstates{n_states}_{params}.npz").exists():
            new_log_mu, new_alphas, new_p_binom, new_taus, new_log_startprob, new_log_transmat = hmmmodel.run_baum_welch_nb_bb_sitewise(X, lengths, \
                n_states, np.zeros(base_nb_mean.shape,dtype=int), total_bb_RD, log_sitewise_transmat, \
                fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, \
                fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                is_diag=is_diag, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, \
                max_iter=max_iter, tol=tol)

            # compute posterior and prediction
            log_gamma = posterior_nb_bb_sitewise(X, lengths, \
                                            base_nb_mean, new_log_mu, new_alphas, \
                                            total_bb_RD, new_p_binom, new_taus, \
                                            new_log_startprob, new_log_transmat, log_sitewise_transmat)
            pred = np.argmax(log_gamma, axis=0)
            pred_cnv = pred % n_states
            
            # likelihood
            log_emission = compute_emission_probability_nb_betabinom(X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus)
            log_alpha = forward_lattice_sitewise(lengths, new_log_transmat, new_log_startprob, log_emission, log_sitewise_transmat)
            llf = np.sum(scipy.special.logsumexp(log_alpha[:,lengths-1], axis=0))

            # clone assignmment
            res = {"new_log_mu":new_log_mu, "new_alphas":new_alphas, "new_p_binom":new_p_binom, "new_taus":new_taus, \
                "new_log_startprob":new_log_startprob, "new_log_transmat":new_log_transmat, "log_gamma":log_gamma, "pred_cnv":pred_cnv, "llf":llf}
            collapsed_llf = regionwise_likelihood(single_X, single_base_nb_mean, single_total_bb_RD, res, min_length=300)
            new_assignment = clustering_regionwise_likelihood(collapsed_llf, res, last_assignment)
            res["new_assignment"] = new_assignment

            # save results
            np.savez(f"{outdir}/round{r}_nstates{n_states}_{params}.npz", **res)

        else:
            res = np.load(f"{outdir}/round{r}_nstates{n_states}_{params}.npz")

        # regroup to pseudobulk
        clone_index = [np.where(res["new_assignment"] == c)[0] for c in range(len(initial_clone_index))]
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)

        # update last parameter
        if "mp" in params:
            print("outer iteration {}: difference between parameters = {}, {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])), np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        elif "m" in params:
            print("outer iteration {}: difference between NB parameters = {}".format( r, np.mean(np.abs(last_log_mu-res["new_log_mu"])) ))
        elif "p" in params:
            print("outer iteration {}: difference between BetaBinom parameters = {}".format( r, np.mean(np.abs(last_p_binom-res["new_p_binom"])) ))
        print("outer iteration {}: ARI between assignment = {}".format( r, adjusted_rand_score(last_assignment, res["new_assignment"]) ))
        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]