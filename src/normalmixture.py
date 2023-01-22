import numpy as np
import scipy.special
from sklearn.mixture import GaussianMixture
from tqdm import trange
import copy
from utils_distribution_fitting import *


def compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, relative_rdr_weight=1.0):
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

    p_binom : array, shape (n_states, n_spots)
        BAF due to CNV. Mean of Beta Binomial distribution in HMM per state per spot.

    taus : array, shape (n_states, n_spots)
        Over-dispersion of Beta Binomial distribution in HMM per state per spot.
    
    Returns
    ----------
    log_emission : array, shape (2*n_states, n_obs, n_spots)
        Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = log_mu.shape[0]
    # initialize log_emission
    log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
    log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))
    for i in np.arange(n_states):
        for s in np.arange(n_spots):
            # expression from NB distribution: TBD!!!
            idx_nonzero = np.where(base_nb_mean[:,s] > 0)[0]
            if len(idx_nonzero) > 0:
                nb_mean = base_nb_mean[idx_nonzero,s] * (tumor_prop[s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[s])
                nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                n, p = convert_params(nb_mean, nb_std)
                log_emission_rdr[i, idx_nonzero, s] = relative_rdr_weight * scipy.stats.nbinom.logpmf(X[idx_nonzero, 0, s], n, p)
                log_emission_rdr[i + n_states, idx_nonzero, s] = log_emission_rdr[i, idx_nonzero, s]
            # AF from BetaBinom distribution
            idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
            if len(idx_nonzero) > 0:
                mix_p_A = p_binom[i, s] * tumor_prop[s] + 0.5 * (1 - tumor_prop[s])
                mix_p_B = (1 - p_binom[i, s]) * tumor_prop[s] + 0.5 * (1 - tumor_prop[s])
                log_emission_baf[i, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], mix_p_A * taus[i, s], mix_p_B * taus[i, s])
                log_emission_baf[i + n_states, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], mix_p_B * taus[i, s], mix_p_A * taus[i, s])
    return log_emission_rdr + log_emission_baf


def compute_emission_probability_nb_betabinom_mix_v2(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, relative_rdr_weight=1.0):
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

    p_binom : array, shape (n_states, n_spots)
        BAF due to CNV. Mean of Beta Binomial distribution in HMM per state per spot.

    taus : array, shape (n_states, n_spots)
        Over-dispersion of Beta Binomial distribution in HMM per state per spot.
    
    Returns
    ----------
    log_emission : array, shape (2*n_states, n_obs, n_spots)
        Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = log_mu.shape[0]
    # initialize log_emission
    log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
    log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))
    for i in np.arange(n_states):
        for s in np.arange(n_spots):
            # expression from NB distribution: TBD!!!
            idx_nonzero = np.where(base_nb_mean[:,s] > 0)[0]
            if len(idx_nonzero) > 0:
                nb_mean = base_nb_mean[idx_nonzero,s] * (tumor_prop[s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[s])
                nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                n, p = convert_params(nb_mean, nb_std)
                log_emission_rdr[i, idx_nonzero, s] = relative_rdr_weight * scipy.stats.nbinom.logpmf(X[idx_nonzero, 0, s], n, p)
                log_emission_rdr[i + n_states, idx_nonzero, s] = log_emission_rdr[i, idx_nonzero, s]
            # AF from BetaBinom distribution
            idx_nonzero = np.where(total_bb_RD[:,s] > 0)[0]
            if len(idx_nonzero) > 0:
                mix_p_A = p_binom[i, s] * tumor_prop[s] + 0.5 * (1 - tumor_prop[s])
                mix_p_B = (1 - p_binom[i, s]) * tumor_prop[s] + 0.5 * (1 - tumor_prop[s])
                log_emission_baf[i, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], mix_p_A * taus[i, s], mix_p_B * taus[i, s])
                log_emission_baf[i + n_states, idx_nonzero, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero,1,s], total_bb_RD[idx_nonzero,s], mix_p_B * taus[i, s], mix_p_A * taus[i, s])
    return log_emission_rdr, log_emission_baf


############################################################
# M step related
############################################################

def update_emission_params_nb_sitewise_uniqvalues_mix(unique_values, mapping_matrices, log_gamma, base_nb_mean, alphas, tumor_prop, \
    start_log_mu=None, fix_NB_dispersion=False, shared_NB_dispersion=False, min_log_rdr=-2, max_log_rdr=2):
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
    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_log_mu = copy.copy(start_log_mu) if not start_log_mu is None else np.zeros((n_states, n_spots))
    new_alphas = copy.copy(alphas)
    # expression signal by NB distribution
    if fix_NB_dispersion:
        new_log_mu = np.zeros((n_states, n_spots))
        for s in range(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
            for i in range(n_states):
                model = sm.GLM(unique_values[s][idx_nonzero,0], np.ones(len(idx_nonzero)).reshape(-1,1), \
                            family=sm.families.NegativeBinomial(alpha=alphas[i,s]), \
                            exposure=unique_values[s][idx_nonzero,1], var_weights=tmp[i,idx_nonzero]+tmp[i+n_states,idx_nonzero])
                res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                new_log_mu[i, s] = res.params[0]
                if not (start_log_mu is None):
                    res2 = model.fit(disp=0, maxiter=1500, start_params=np.array([start_log_mu[i, s]]), xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0] if -model.loglike(res.params) < -model.loglike(res2.params) else res2.params[0]
    else:
        if not shared_NB_dispersion:
            for s in range(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
                for i in range(n_states):
                    model = Weighted_NegativeBinomial_mix(unique_values[s][idx_nonzero,0], \
                                np.ones(len(idx_nonzero)).reshape(-1,1), \
                                weights=tmp[i,idx_nonzero]+tmp[i+n_states,idx_nonzero], exposure=unique_values[s][idx_nonzero,1], \
                                tumor_prop=tumor_prop[s])
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_log_mu[i, s] = res.params[0]
                    new_alphas[i, s] = res.params[-1]
                    if not (start_log_mu is None):
                        res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_log_mu[i, s]], [alphas[i, s]]), xtol=1e-4, ftol=1e-4)
                        new_log_mu[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                        new_alphas[i, s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in range(n_spots):
                idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
                this_exposure = np.tile(unique_values[s][idx_nonzero,1], n_states)
                this_y = np.tile(unique_values[s][idx_nonzero,0], n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate([ tmp[i,idx_nonzero] + tmp[i+n_states,idx_nonzero] for i in range(n_states) ])
                this_features = np.zeros((n_states*len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[(i*len(idx_nonzero)):((i+1)*len(idx_nonzero)), i] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array([ i for i in range(this_features.shape[1]) if np.sum(this_weights[this_features[:,i]==1]) >= 0.1 ])
                idx_row_posweight = np.concatenate([ np.where(this_features[:,k]==1)[0] for k in idx_state_posweight ])
                y.append( this_y[idx_row_posweight] )
                exposure.append( this_exposure[idx_row_posweight] )
                weights.append( this_weights[idx_row_posweight] )
                features.append( this_features[idx_row_posweight, :][:, idx_state_posweight] )
                state_posweights.append( idx_state_posweight )
                tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_NegativeBinomial_mix(y, features, weights=weights, exposure=exposure, tumor_prop=tp)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s,idx_state_posweight in enumerate(state_posweights):
                l1 = int( np.sum([len(x) for x in state_posweights[:s]]) )
                l2 = int( np.sum([len(x) for x in state_posweights[:(s+1)]]) )
                new_log_mu[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_alphas[:,:] = res.params[-1]
            if not (start_log_mu is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=np.concatenate([start_log_mu[idx_state_posweight,s] for s,idx_state_posweight in enumerate(state_posweights)] + [np.ones(1) * alphas[0,s]]), xtol=1e-4, ftol=1e-4)
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s,idx_state_posweight in enumerate(state_posweights):
                        l1 = int( np.sum([len(x) for x in state_posweights[:s]]) )
                        l2 = int( np.sum([len(x) for x in state_posweights[:(s+1)]]) )
                        new_log_mu[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_alphas[:,:] = res2.params[-1]
    new_log_mu[new_log_mu > max_log_rdr] = max_log_rdr
    new_log_mu[new_log_mu < min_log_rdr] = min_log_rdr
    return new_log_mu, new_alphas


def update_emission_params_bb_sitewise_uniqvalues_mix(unique_values, mapping_matrices, log_gamma, total_bb_RD, taus, tumor_prop, \
    start_p_binom=None, fix_BB_dispersion=False, shared_BB_dispersion=False, \
    percent_threshold=0.99, min_binom_prob=0.01, max_binom_prob=0.99):
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
    n_spots = len(unique_values)
    n_states = int(log_gamma.shape[0] / 2)
    gamma = np.exp(log_gamma)
    # initialization
    new_p_binom = copy.copy(start_p_binom) if not start_p_binom is None else np.ones((n_states, n_spots)) * 0.5
    new_taus = copy.copy(taus)
    if fix_BB_dispersion:
        for s in np.arange(n_spots):
            tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
            idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
            for i in range(n_states):
                # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                if np.sum(tmp[i,idx_nonzero]) + np.sum(tmp[i+n_states,idx_nonzero]) >= 0.1:
                    model = Weighted_BetaBinom_fixdispersion_mix(np.append(unique_values[s][idx_nonzero,0], unique_values[s][idx_nonzero,1]-unique_values[s][idx_nonzero,0]), \
                        np.ones(2*len(idx_nonzero)).reshape(-1,1), \
                        taus[i,s], \
                        weights=np.append(tmp[i,idx_nonzero], tmp[i+n_states,idx_nonzero]), \
                        exposure=np.append(unique_values[s][idx_nonzero,1], unique_values[s][idx_nonzero,1]), \
                        tumor_prop=tumor_prop[s] )
                    res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                    new_p_binom[i, s] = res.params[0]
                    if not (start_p_binom is None):
                        res2 = model.fit(disp=0, maxiter=1500, start_params=np.array(start_p_binom[i, s]), xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
    else:
        if not shared_BB_dispersion:
            for s in np.arange(n_spots):
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
                for i in range(n_states):
                    # only optimize for BAF only when the posterior probability >= 0.1 (at least 1 SNP is under this state)
                    if np.sum(tmp[i,idx_nonzero]) + np.sum(tmp[i+n_states,idx_nonzero]) >= 0.1:
                        model = Weighted_BetaBinom_mix(np.append(unique_values[s][idx_nonzero,0], unique_values[s][idx_nonzero,1]-unique_values[s][idx_nonzero,0]), \
                            np.ones(2*len(idx_nonzero)).reshape(-1,1), \
                            weights=np.append(tmp[i,idx_nonzero], tmp[i+n_states,idx_nonzero]), \
                            exposure=np.append(unique_values[s][idx_nonzero,1], unique_values[s][idx_nonzero,1]),\
                            tumor_prop=tumor_prop )
                        res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
                        new_p_binom[i, s] = res.params[0]
                        new_taus[i, s] = res.params[-1]
                        if not (start_p_binom is None):
                            res2 = model.fit(disp=0, maxiter=1500, start_params=np.append([start_p_binom[i, s]], [taus[i, s]]), xtol=1e-4, ftol=1e-4)
                            new_p_binom[i, s] = res.params[0] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[0]
                            new_taus[i, s] = res.params[-1] if model.nloglikeobs(res.params) < model.nloglikeobs(res2.params) else res2.params[-1]
        else:
            exposure = []
            y = []
            weights = []
            features = []
            state_posweights = []
            tp = []
            for s in np.arange(n_spots):
                idx_nonzero = np.where(unique_values[s][:,1] > 0)[0]
                this_exposure = np.tile( np.append(unique_values[s][idx_nonzero,1], unique_values[s][idx_nonzero,1]), n_states)
                this_y = np.tile( np.append(unique_values[s][idx_nonzero,0], unique_values[s][idx_nonzero,1]-unique_values[s][idx_nonzero,0]), n_states)
                tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).A
                this_weights = np.concatenate([ np.append(tmp[i,idx_nonzero], tmp[i+n_states,idx_nonzero]) for i in range(n_states) ])
                this_features = np.zeros((2*n_states*len(idx_nonzero), n_states))
                for i in np.arange(n_states):
                    this_features[(i*2*len(idx_nonzero)):((i+1)*2*len(idx_nonzero)), i] = 1
                # only optimize for states where at least 1 SNP belongs to
                idx_state_posweight = np.array([ i for i in range(this_features.shape[1]) if np.sum(this_weights[this_features[:,i]==1]) >= 0.1 ])
                idx_row_posweight = np.concatenate([ np.where(this_features[:,k]==1)[0] for k in idx_state_posweight ])
                y.append( this_y[idx_row_posweight] )
                exposure.append( this_exposure[idx_row_posweight] )
                weights.append( this_weights[idx_row_posweight] )
                features.append( this_features[idx_row_posweight, :][:, idx_state_posweight] )
                state_posweights.append( idx_state_posweight )
                tp.append( tumor_prop[s] * np.ones(len(idx_row_posweight)) )
            exposure = np.concatenate(exposure)
            y = np.concatenate(y)
            weights = np.concatenate(weights)
            features = scipy.linalg.block_diag(*features)
            tp = np.concatenate(tp)
            model = Weighted_BetaBinom_mix(y, features, weights=weights, exposure=exposure, tumor_prop=tp)
            res = model.fit(disp=0, maxiter=1500, xtol=1e-4, ftol=1e-4)
            for s,idx_state_posweight in enumerate(state_posweights):
                l1 = int( np.sum([len(x) for x in state_posweights[:s]]) )
                l2 = int( np.sum([len(x) for x in state_posweights[:(s+1)]]) )
                new_p_binom[idx_state_posweight, s] = res.params[l1:l2]
            if res.params[-1] > 0:
                new_taus[:,:] = res.params[-1]
            if not (start_p_binom is None):
                res2 = model.fit(disp=0, maxiter=1500, start_params=np.concatenate([start_p_binom[idx_state_posweight,s] for s,idx_state_posweight in enumerate(state_posweights)] + [np.ones(1) * taus[0,s]]), xtol=1e-4, ftol=1e-4)
                if model.nloglikeobs(res2.params) < model.nloglikeobs(res.params):
                    for s,idx_state_posweight in enumerate(state_posweights):
                        l1 = int( np.sum([len(x) for x in state_posweights[:s]]) )
                        l2 = int( np.sum([len(x) for x in state_posweights[:(s+1)]]) )
                        new_p_binom[idx_state_posweight, s] = res2.params[l1:l2]
                    if res2.params[-1] > 0:
                        new_taus[:,:] = res2.params[-1]
    return new_p_binom, new_taus
