import numpy as np
import scipy.special
from sklearn.mixture import GaussianMixture
from tqdm import trange
import copy
from utils_distribution_fitting import *


############################################################
# M step related
############################################################

def update_emission_params_nb_sitewise(X_nb, log_gamma, base_nb_mean, alphas, \
    start_log_mu=None, fix_NB_dispersion=False, shared_NB_dispersion=False, min_log_rdr=-2, max_log_rdr=2):
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
    n_spots = X_nb.shape[1]
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        new_alphas = alphas
        for s in range(n_spots):
            idx_nonzero = np.where(base_nb_mean[:,s] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(X_nb[idx_nonzero,s], np.ones(len(idx_nonzero)).reshape(-1,1), \
                            family=sm.families.NegativeBinomial(alpha=alphas[i,s]), \
                            exposure=base_nb_mean[idx_nonzero,s], var_weights=gamma[i,idx_nonzero]+gamma[i+n_states,idx_nonzero])
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                # print(s, i, res.params)
                if not (start_log_mu is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.array([start_log_mu[i, s]]), xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0] if -model.loglike(res.params) < -model.loglike(res2.params) else res2.params[0]
    else:
        new_log_mu = np.zeros((n_states, n_spots))
        new_alphas = np.zeros((n_states, n_spots))
        if not shared_NB_dispersion:
            for s in range(n_spots):
                idx_nonzero = np.where(base_nb_mean[:,s] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial(X_nb[idx_nonzero,s], \
                                np.ones(len(idx_nonzero)).reshape(-1,1), \
                                weights=gamma[i,idx_nonzero]+gamma[i+n_states,idx_nonzero], exposure=base_nb_mean[idx_nonzero,s])
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_log_mu[i, s]], [alphas[i, s]]), xtol=1e-4, ftol=1e-4)
                        new_log_mu[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                        new_alphas[i, s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            for s in range(n_spots):
                idx_nonzero = np.where(base_nb_mean[:,s] > 0)[0]
                all_states_nb_mean = np.tile(base_nb_mean[idx_nonzero,s], n_states)
                all_states_y = np.tile(X_nb[idx_nonzero,s], n_states)
                all_states_weights = np.concatenate([gamma[i,idx_nonzero]+gamma[i+n_states,idx_nonzero] for i in range(n_states)])
                all_states_features = np.zeros((n_states*len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    all_states_features[(i*len(idx_nonzero)):((i+1)*len(idx_nonzero)), i] = 1
                model = Weighted_NegativeBinomial(all_states_y, all_states_features, weights=all_states_weights, exposure=all_states_nb_mean)
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[:,s] = res.params[:-1]
                new_alphas[:,s] = res.params[-1]
                if not (start_log_mu is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.append(start_log_mu[:,s], [alphas[0,s]]), xtol=1e-4, ftol=1e-4)
                    new_log_mu[:,s] = res.params[:-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[:-1]
                    new_alphas[:,s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr
    return new_log_mu, new_alphas


def update_emission_params_bb_sitewise(X_bb, log_gamma, total_bb_RD, taus, \
    start_p_binom=None, fix_BB_dispersion=False, shared_BB_dispersion=False, \
    percent_threshold=0.99, min_binom_prob=0.01, max_binom_prob=0.99):
    """
    Attributes
    ----------
    X_bb : array, shape (n_observations, n_spots)
        Observed allele frequency UMI count.

    log_gamma : array, (2*n_states, n_observations)
        Posterior probability of observing each state at each observation time.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.
    """
    n_spots = X_bb.shape[1]
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = np.ones((n_states, n_spots)) * 0.5
    new_taus = copy.copy(taus)
    if fix_BB_dispersion: 
        for s in np.arange(X_bb.shape[1]):
            idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
            for i in range(n_states):
                model = Weighted_BetaBinom_fixdispersion(np.append(X_bb[idx_nonzero,s], total_bb_RD[idx_nonzero,s]-X_bb[idx_nonzero,s]), \
                    np.ones(2*len(idx_nonzero)).reshape(-1,1), \
                    taus[i,s], \
                    weights=np.append(gamma[i,idx_nonzero], gamma[i+n_states,idx_nonzero]), \
                    exposure=np.append(total_bb_RD[idx_nonzero,s], total_bb_RD[idx_nonzero,s]) )
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_p_binom[i, s] = res.params[0]
                if not (start_p_binom is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.array(start_p_binom[i, s]), xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
    else:
        if not shared_BB_dispersion:
            for s in np.arange(X_bb.shape[1]):
                idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
                for i in range(n_states):
                    model = Weighted_BetaBinom(np.append(X_bb[idx_nonzero,s], total_bb_RD[idx_nonzero,s]-X_bb[idx_nonzero,s]), \
                        np.ones(2*len(idx_nonzero)).reshape(-1,1), \
                        weights=np.append(gamma[i,idx_nonzero], gamma[i+n_states,idx_nonzero]), \
                        exposure=np.append(total_bb_RD[idx_nonzero,s], total_bb_RD[idx_nonzero,s]) )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    new_taus[i, s] = res.params[-1]
                    if not (start_p_binom is None):
                        res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_p_binom[i, s]], [taus[i, s]]), xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                        new_taus[i, s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            for s in np.arange(X_bb.shape[1]):
                idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
                all_states_exposure = np.tile( np.append(total_bb_RD[idx_nonzero,s], total_bb_RD[idx_nonzero,s]), n_states)
                all_states_y = np.tile( np.append(X_bb[idx_nonzero,s], total_bb_RD[idx_nonzero,s]-X_bb[idx_nonzero,s]), n_states)
                all_states_weights = np.concatenate([ np.append(gamma[i,idx_nonzero], gamma[i+n_states,idx_nonzero]) for i in range(n_states) ])
                all_states_features = np.zeros((2*n_states*len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    all_states_features[(i*2*len(idx_nonzero)):((i+1)*2*len(idx_nonzero)), i] = 1
                model = Weighted_BetaBinom(all_states_y, all_states_features, weights=all_states_weights, exposure=all_states_exposure)
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_p_binom[:,s] = res.params[:-1]
                new_p_binom[new_p_binom[:,s] < min_binom_prob, s] = min_binom_prob
                new_p_binom[new_p_binom[:,s] > max_binom_prob, s] = max_binom_prob
                if res.params[-1] > 0:
                    new_taus[:, s] = res.params[-1]
                if not (start_p_binom is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.append(start_p_binom[:,s], [taus[0, s]]), xtol=1e-4, ftol=1e-4)
                    new_p_binom[:,s] = res.params[:-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[:-1]
                    new_p_binom[new_p_binom[:,s] < min_binom_prob, s] = min_binom_prob
                    new_p_binom[new_p_binom[:,s] > max_binom_prob, s] = max_binom_prob
                    if res2.params[-1] > 0:
                        new_taus[:,s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
    return new_p_binom, new_taus



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
            tmp_log_emission_rdr, tmp_log_emission_baf = compute_emission_probability_nb_betabinom_phaseswitch( np.sum(single_X[:,:,idx], axis=2, keepdims=True), \
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
            # w_edge[new_assignment[j]] += 1
            w_edge[new_assignment[j]] += adjacency_mat[i,j]
        # combine both potential for the new assignment
        new_assignment[i] = np.argmax( w_node + spatial_weight * w_edge )
    
    # compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum( spatial_weight * np.sum(new_assignment[adjacency_mat[i,:].nonzero()[1]] == new_assignment[i]) )
    return new_assignment, single_llf, total_llf



def allele_starch_combine_clones():
    res_combine = {"new_assignment":np.zeros(single_X.shape[2], dtype=int)}
    offset_clone = 0
    for bafc in range(n_baf_clones):
        prefix = f"clone{bafc}"
        allres = dict( np.load(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", allow_pickle=True) )
        r = allres["num_iterations"] - 1
        res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
            "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
            "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
            "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
            "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
        idx_spots = np.where(adata.obs.index.isin( allres["barcodes"] ))[0]
        n_obs = single_X.shape[0]
        if len(np.unique(res["new_assignment"])) == 1:
            n_merged_clones = 1
            c = res["new_assignment"][0]
            merged_res = copy.copy(res)
            merged_res["new_assignment"] = np.zeros(len(idx_spots), dtype=int)
            log_gamma = res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)].reshape((2*config["n_states"], n_obs, 1))
            pred_cnv = res["pred_cnv"][ (c*n_obs):(c*n_obs+n_obs) ].reshape((-1,1))
        else:
            if config["tumorprop_file"] is None:
                X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in range(n_clones_rdr)])
                tumor_prop = None
            else:
                X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in range(n_clones_rdr)], single_tumor_prop[idx_spots])
            merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, params="smp", tumor_prop=tumor_prop)
            print(f"part {bafc} merging_groups: {merging_groups}")
            #
            if config["tumorprop_file"] is None:
                merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], res, min_spots_thresholds=50)
            else:
                merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], res, min_spots_thresholds=50, single_tumor_prop=single_tumor_prop[idx_spots])
            # compute posterior using the newly merged pseudobulk
            n_merged_clones = len(merging_groups)
            tmp = copy.copy(merged_res["new_assignment"])
            if config["tumorprop_file"] is None:
                X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)])
                tumor_prop = None
            else:
                X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)], single_tumor_prop[idx_spots])
            #
            merged_res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config["n_states"], \
                    base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                    fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                    is_diag=True, init_log_mu=res["new_log_mu"], init_p_binom=res["new_p_binom"], init_alphas=res["new_alphas"], init_taus=res["new_taus"], max_iter=config["max_iter"], tol=config["tol"])
            merged_res["new_assignment"] = copy.copy(tmp)
            log_gamma = np.stack([ merged_res["log_gamma"][:,(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ], axis=-1)
            pred_cnv = np.vstack([ merged_res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ]).T

        # add to res_combine
        if len(res_combine) == 1:
            res_combine.update({"new_log_mu":np.hstack([ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([ merged_res["new_alphas"] ] * n_merged_clones), \
                "new_p_binom":np.hstack([ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([ merged_res["new_taus"] ] * n_merged_clones), \
                "log_gamma":log_gamma, "pred_cnv":pred_cnv})
        else:
            res_combine.update({"new_log_mu":np.hstack([res_combine["new_log_mu"]] + [ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([res_combine["new_alphas"]] + [ merged_res["new_alphas"] ] * n_merged_clones), \
                "new_p_binom":np.hstack([res_combine["new_p_binom"]] + [ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([res_combine["new_taus"]] + [ merged_res["new_taus"] ] * n_merged_clones), \
                "log_gamma":np.dstack([res_combine["log_gamma"], log_gamma ]), "pred_cnv":np.hstack([res_combine["pred_cnv"], pred_cnv])})
        res_combine["new_assignment"][idx_spots] = merged_res["new_assignment"] + offset_clone
        offset_clone += n_merged_clones
    # compute HMRF log likelihood
    total_llf = hmrf_log_likelihood(config["nodepotential"], single_X, single_base_nb_mean, single_total_bb_RD, res_combine, np.argmax(res_combine["log_gamma"],axis=0), smooth_mat, adjacency_mat, res_combine["new_assignment"], config["spatial_weight"])
    res_combine["total_llf"] = total_llf
    # save results
    np.savez(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", **res_combine)



def simplify_parameters(res, params="smp", bafthreshold=0.05, rdrthreshold=0.1):
    n_states = res["new_p_binom"].shape[0]
    G = nx.Graph()
    G.add_nodes_from( np.arange(n_states) )
    mAF = np.where(res["new_p_binom"].flatten() < 0.5, res["new_p_binom"].flatten(), 1-res["new_p_binom"].flatten())
    if "m" in params and "p" in params:
        tmp_edge_graph = (np.abs( res["new_log_mu"].flatten().reshape(-1,1) - res["new_log_mu"].flatten().reshape(1,-1) ) < rdrthreshold) & (np.abs( mAF.reshape(-1,1) - mAF.reshape(1,-1) ) < bafthreshold)
    elif "m" in params:
        tmp_edge_graph = (np.abs( res["new_log_mu"].flatten().reshape(-1,1) - res["new_log_mu"].flatten().reshape(1,-1) ) < rdrthreshold)
    else:
        tmp_edge_graph = (np.abs( mAF.reshape(-1,1) - mAF.reshape(1,-1) ) < bafthreshold)
    G.add_edges_from([ (i,j) for i in range(tmp_edge_graph.shape[0]) for j in range(tmp_edge_graph.shape[1]) if tmp_edge_graph[i,j] ])
    # maximal cliques
    cliques = []
    for x in nx.find_cliques(G):
        this_len = len(x)
        cliques.append( (x, this_len) )
    cliques.sort(key = lambda x:(-x[1]) )
    covered_states = set()
    merging_state_groups = []
    for c in cliques:
        if len(set(c[0]) & covered_states) == 0:
            merging_state_groups.append( list(c[0]) )
            covered_states = covered_states | set(c[0])
    for c in range(n_states):
        if not (c in covered_states):
            merging_state_groups.append( [c] )
            covered_states.add(c)
    merging_state_groups.sort(key = lambda x:np.min(x))
    # merged parameters
    simplied_res = {"new_log_mu":np.array([ np.mean(res["new_log_mu"].flatten()[idx]) for idx in merging_state_groups]).reshape(-1,1), \
        "new_p_binom":np.array([ np.mean(res["new_p_binom"].flatten()[idx]) for idx in merging_state_groups]).reshape(-1,1), \
        "new_alphas":np.array([ np.mean(res["new_alphas"].flatten()[idx]) for idx in merging_state_groups]).reshape(-1,1), \
        "new_taus":np.array([ np.mean(res["new_taus"].flatten()[idx]) for idx in merging_state_groups]).reshape(-1,1)}
    return simplied_res


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


def initialization_rdr_bybaf(n_states, X, base_nb_mean, total_bb_RD, params, prior_p_binom, random_state=None, in_log_space=True):
    tmp_log_mu, tmp_p_binom = initialization_by_gmm(n_states, X, base_nb_mean, total_bb_RD, params, random_state=random_state, in_log_space=in_log_space, min_binom_prob=0, max_binom_prob=1)
    prior_log_mu = np.zeros(prior_p_binom.shape)
    for i,x in enumerate(prior_p_binom):
        idx_nearest = np.argmin( scipy.spatial.distance.cdist(x.reshape(-1,1), tmp_p_binom) )
        prior_log_mu[i] = tmp_log_mu[idx_nearest]
    return prior_log_mu



def output_integer_CN():
    ##### infer integer copy #####
    res_combine = dict(np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))
    n_final_clone = len(np.unique(res_combine["new_assignment"]))
    medfix = ["", "_diploid", "_triploid", "_tetraploid"]
    for o,max_medploidy in enumerate([None, 2, 3, 4]):
        # A/B copy number per bin
        A_copy = np.zeros((n_final_clone, n_obs), dtype=int)
        B_copy = np.zeros((n_final_clone, n_obs), dtype=int)
        # A/B copy number per state
        state_A_copy = np.zeros((n_final_clone, config['n_states']), dtype=int)
        state_B_copy = np.zeros((n_final_clone, config['n_states']), dtype=int)

        df_genelevel_cnv = None
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==c)[0] for c in range(n_final_clone)])
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==c)[0] for c in range(n_final_clone)], single_tumor_prop)

        for s in range(n_final_clone):
            # adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1
            lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
            this_pred_cnv = res_combine["pred_cnv"][:,s]
            adjusted_log_mu = np.log( np.exp(res_combine["new_log_mu"][:,s]) / np.sum(np.exp(res_combine["new_log_mu"][this_pred_cnv,s]) * lambd) )
            if not max_medploidy is None:
                best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, max_medploidy=max_medploidy)
            else:
                best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv)
            print(f"max med ploidy = {max_medploidy}, clone {s}, integer copy inference loss = {_}")
            
            A_copy[s,:] = best_integer_copies[res_combine["pred_cnv"][:,s], 0]
            B_copy[s,:] = best_integer_copies[res_combine["pred_cnv"][:,s], 1]
            state_A_copy[s,:] = best_integer_copies[:,0]
            state_B_copy[s,:] = best_integer_copies[:,1]
            tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[res_combine["pred_cnv"][:,s], 0], best_integer_copies[res_combine["pred_cnv"][:,s], 1], x_gene_list)
            tmpdf.columns = [f"clone{s} A", f"clone{s} B"]
            if df_genelevel_cnv is None:
                df_genelevel_cnv = copy.copy(tmpdf)
            else:
                df_genelevel_cnv = df_genelevel_cnv.join(tmpdf)
        # output gene-level copy number
        df_genelevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_genelevel.tsv", header=True, index=True, sep="\t")
        # output segment-level copy number
        df_seglevel_cnv = pd.DataFrame({"CHR":[x[0] for x in sorted_chr_pos], "START":[x[1] for x in sorted_chr_pos], \
            "END":[ (sorted_chr_pos[i+1][1] if i+1 < len(sorted_chr_pos) and x[0]==sorted_chr_pos[i+1][0] else -1) for i,x in enumerate(sorted_chr_pos)] })
        for s in range(n_final_clone):
            df_seglevel_cnv[f"clone{s} A"] = A_copy[s,:]
            df_seglevel_cnv[f"clone{s} B"] = B_copy[s,:]
        df_seglevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_seglevel.tsv", header=True, index=False, sep="\t")
        # output per-state copy number
        df_state_cnv = {}
        for s in range(n_final_clone):
            df_state_cnv[f"clone{s} logmu"] = res_combine["new_log_mu"][:,s]
            df_state_cnv[f"clone{s} p"] = res_combine["new_p_binom"][:,s]
            df_state_cnv[f"clone{s} A"] = state_A_copy[s,:]
            df_state_cnv[f"clone{s} B"] = state_B_copy[s,:]
        df_state_cnv = pd.DataFrame.from_dict(df_state_cnv)
        df_state_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_perstate.tsv", header=True, index=False, sep="\t")
    
    ##### output clone label #####
    adata.obs["clone_label"] = res_combine["new_assignment"]
    if config["tumorprop_file"] is None:
        adata.obs[["clone_label"]].to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")
    else:
        adata.obs[["tumor_proportion", "clone_label"]].to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")


def set_bin_exp_to_zero():
    # remove bins for which RDR > MAX_RDR in any smoothed spot
    MAX_RDR = 15
    N_STEP = 2
    multi_step_smooth = copy.copy(smooth_mat)
    for _ in range(N_STEP):
        multi_step_smooth = (multi_step_smooth + multi_step_smooth @ smooth_mat)
    multi_step_smooth = (multi_step_smooth > 0).astype(int)
    rdr = (copy_single_X_rdr @ multi_step_smooth) / (copy_single_base_nb_mean @ multi_step_smooth)
    rdr[np.sum(copy_single_base_nb_mean,axis=1) == 0] = 0
    bidx_inconfident = np.where(~np.all(rdr <= MAX_RDR, axis=1))[0] 
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)
    copy_single_X_rdr[bidx_inconfident, :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
    copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
