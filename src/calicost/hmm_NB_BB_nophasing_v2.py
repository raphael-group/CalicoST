import logging
import numpy as np
import line_profiler
from numba import njit
from scipy.stats import norm, multivariate_normal, poisson
import scipy.special
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.mixture import GaussianMixture
from tqdm import trange
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import copy
from calicost.utils_distribution_fitting import *
from calicost.utils_hmm import *
from calicost.utils_tumor import get_tumor_weight
from calicost.utils_thread_emission import thread_betabinom, thread_nbinom
import networkx as nx

"""
Joint NB-BB HMM that accounts for tumor/normal genome proportions. Tumor genome proportion is weighted by mu in BB distribution.
"""

############################################################
# whole inference
############################################################

class hmm_nophasing_v2(object):
    def __init__(self, params="stmp", t=1-1e-4):
        """
        Attributes
        ----------
        params : str
            Codes for parameters that need to be updated. The corresponding parameter can only be updated if it is included in this argument. "s" for start probability; "t" for transition probability; "m" for Negative Binomial RDR signal; "p" for Beta Binomial BAF signal.

        t : float
            Determine initial self transition probability to be 1-t.
        """
        self.params = params
        self.t = t

    @staticmethod
    def compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus):
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
        log_emission : array, shape (n_states, n_obs, n_spots)
            Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        n_states = log_mu.shape[0]
        # initialize log_emission
        log_emission_rdr = np.zeros((n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((n_states, n_obs, n_spots))
        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                # expression from NB distribution
                idx_nonzero_rdr = np.where(base_nb_mean[:,s] > 0)[0]
                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr,s] * np.exp(log_mu[i, s])
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(X[idx_nonzero_rdr, 0, s], n, p)
                # AF from BetaBinom distribution
                idx_nonzero_baf = np.where(total_bb_RD[:,s] > 0)[0]
                if len(idx_nonzero_baf) > 0:
                    log_emission_baf[i, idx_nonzero_baf, s] = scipy.stats.betabinom.logpmf(X[idx_nonzero_baf,1,s], total_bb_RD[idx_nonzero_baf,s], p_binom[i, s] * taus[i, s], (1-p_binom[i, s]) * taus[i, s])
        return log_emission_rdr, log_emission_baf

    @staticmethod
    def compute_emission_probability_nb_betabinom_v2(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus):
        n_states = log_mu.shape[0]
        (n_obs, n_comp, n_spots) = X.shape

        # NB (n_states, n_obs, n_spots) == (7, 4_248, 1)                                                                                                                                                             
        log_emission_rdr = np.zeros(shape=(n_states, n_obs, n_spots), dtype=float)

        # NB nb_mean, nb_std: (segments, spots) * (states, spots) = (states, segments, spots) == (7, 4248, 1)                                                                                                       
        nb_mean = np.exp(log_mu)[:, None, :] * base_nb_mean[None, :, :]
        nb_var = nb_mean + alphas[:, None, :] * nb_mean**2

        kk = np.tile(X[:, 0, :], (n_states, 1, 1))
        nn, pp = convert_params_var(nb_mean, nb_var)

        idx = np.tile(base_nb_mean > 0., (n_states, 1, 1))
        log_emission_rdr[idx] = scipy.stats.nbinom(kk[idx], nn[idx], pp[idx])

        # NB BAF                                                                                                                                                                                                    
        log_emission_baf = np.zeros(shape=(n_states, n_obs, n_spots), dtype=float)

        kk = np.tile(X[:, 1, :], (n_states, 1, 1))
        nn = np.tile(total_bb_RD[:, :], (n_states, 1, 1))

        # NB (states, spots)                                                                                                                                                                                        
        aa = np.tile((p_binom * taus)[:, None, :], (1, n_obs, 1))
        bb = np.tile(((1. - p_binom) * taus)[:, None, :], (1, n_obs, 1))

        idx = np.tile(total_bb_RD > 0., (n_states, 1, 1))
        log_emission_baf[idx] = scipy.stats.betabinom(kk[idx], nn[idx], aa[idx], bb[idx])

        return log_emission_rdr, log_emission_baf

    @staticmethod
    def compute_emission_probability_nb_betabinom_mix_v1(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, **kwargs):
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
        log_emission : array, shape (n_states, n_obs, n_spots)
            Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        n_states = log_mu.shape[0]
        
        # initialize log_emission
        log_emission_rdr = np.zeros((n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((n_states, n_obs, n_spots))
        
        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                # expression from NB distribution
                idx_nonzero_rdr = np.where(base_nb_mean[:,s] > 0)[0]
                if len(idx_nonzero_rdr) > 0:
                    # nb_mean = base_nb_mean[idx_nonzero_rdr,s] * (tumor_prop[s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[s])
                    nb_mean = base_nb_mean[idx_nonzero_rdr,s] * (tumor_prop[idx_nonzero_rdr,s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[idx_nonzero_rdr,s])
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(X[idx_nonzero_rdr, 0, s], n, p)
                # AF from BetaBinom distribution
                if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
                    this_weighted_tp = []
                    for c in range(len(kwargs["sample_length"])):
                        range_s = np.sum(kwargs["sample_length"][:c])
                        range_t = np.sum(kwargs["sample_length"][:(c+1)])
                        this_weighted_tp.append( tumor_prop[range_s:range_t,s] * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c,s]) / (tumor_prop[range_s:range_t,s] * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c,s]) + 1 - tumor_prop[range_s:range_t,s]) )
                    this_weighted_tp = np.concatenate(this_weighted_tp)
                else:
                    this_weighted_tp = tumor_prop[:,s]
                idx_nonzero_baf = np.where(total_bb_RD[:,s] > 0)[0]
                if len(idx_nonzero_baf) > 0:
                    mix_p_A = p_binom[i, s] * this_weighted_tp[idx_nonzero_baf] + 0.5 * (1 - this_weighted_tp[idx_nonzero_baf])
                    mix_p_B = (1 - p_binom[i, s]) * this_weighted_tp[idx_nonzero_baf] + 0.5 * (1 - this_weighted_tp[idx_nonzero_baf])
                    log_emission_baf[i, idx_nonzero_baf, s] += scipy.stats.betabinom.logpmf(X[idx_nonzero_baf,1,s], total_bb_RD[idx_nonzero_baf,s], mix_p_A * taus[i, s], mix_p_B * taus[i, s])
        return log_emission_rdr, log_emission_baf

    @staticmethod
    @line_profiler.profile
    def compute_emission_probability_nb(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, **kwargs):
        n_states = log_mu.shape[0]
        n_obs, n_comp, n_spots = X.shape
        
        # NB (n_states, n_obs, n_spots) == (7, 4248, 1)
        log_emission_rdr = np.zeros(shape=(n_states, n_obs, n_spots), dtype=float)

        assert base_nb_mean.shape == (n_obs, n_spots)
        assert tumor_prop.shape == (n_obs, n_spots)
        assert log_mu.shape == (n_states, n_spots)
        
        # NB nb_mean, nb_std: (segments, spots) * (states, spots) = (states, segments, spots) == (7, 4248, 1)
        nb_mean = base_nb_mean[None, :, :] * (tumor_prop[None, :, :] * np.exp(log_mu)[:, None, :] + 1. - tumor_prop[None, :, :])
        nb_var = nb_mean + alphas[:, None, :] * nb_mean**2
        
        kk = np.tile(X[:, 0, :], (n_states, 1, 1))
        nn, pp = convert_params_var(nb_mean, nb_var)
        
        idx = np.tile(base_nb_mean > 0., (n_states, 1, 1))
        log_emission_rdr[idx] = scipy.stats.nbinom.logpmf(kk[idx], nn[idx], pp[idx])

        return log_emission_rdr

    @staticmethod
    @line_profiler.profile
    def compute_emission_probability_bb_mix(X, base_nb_mean, total_bb_RD, p_binom, taus, tumor_prop, tumor_weight=None):
        n_states = p_binom.shape[0]
        n_obs, n_comp, n_spots = X.shape

        assert base_nb_mean.shape == (n_obs, n_spots)
        assert tumor_prop.shape == (n_obs, n_spots)
        assert p_binom.shape == (n_states, n_spots)

        log_emission_baf = np.zeros((n_states, n_obs, n_spots))

        if tumor_weight is None:
            tumor_weight = tumor_prop
        
        mix_p_A = p_binom[:, None, :] * tumor_weight + 0.5 * (1. - tumor_weight)
        mix_p_B = (1. - p_binom)[:, None, :] * tumor_weight + 0.5 * (1. - tumor_weight)

        aa = mix_p_A * taus[:, None, :]
        bb = mix_p_B * taus[:, None, :]

        kk = np.tile(X[:, 1, :], (n_states, 1, 1))
        nn = np.tile(total_bb_RD[:, :], (n_states, 1, 1))

        idx = np.tile(total_bb_RD > 0., (n_states, 1, 1))

        log_emission_baf[idx] = scipy.stats.betabinom.logpmf(kk[idx], nn[idx], aa[idx], bb[idx])                                                                                                                            
        return log_emission_baf
    
    @staticmethod
    @line_profiler.profile
    def compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, **kwargs):
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

        tumor_prop: array, shape (n_obs, n_spots)
            Tumor proportion
        
        Returns
        ----------
        log_emission : array, shape (n_states, n_obs, n_spots)
            Log emission probability for each gene each spot (or sample) under each state. There is a common bag of states across all spots.
        """
        n_states = log_mu.shape[0]
        n_obs, n_comp, n_spots = X.shape
        
        # NB (n_states, n_obs, n_spots) == (7, 4248, 1)
        log_emission_rdr = np.zeros(shape=(n_states, n_obs, n_spots), dtype=float)

        assert base_nb_mean.shape == (n_obs, n_spots)
        assert tumor_prop.shape == (n_obs, n_spots)
        assert log_mu.shape == (n_states, n_spots)
        
        # NB nb_mean, nb_std: (segments, spots) * (states, spots) = (states, segments, spots) == (7, 4248, 1)
        nb_mean = base_nb_mean[None, :, :] * (tumor_prop[None, :, :] * np.exp(log_mu)[:, None, :] + 1. - tumor_prop[None, :, :])
        nb_var = nb_mean + alphas[:, None, :] * nb_mean**2
        
        kk = np.tile(X[:, 0, :], (n_states, 1, 1))
        nn, pp = convert_params_var(nb_mean, nb_var)

        idx = np.tile(base_nb_mean > 0., (n_states, 1, 1))

        # log_emission_rdr[idx] = scipy.stats.nbinom.logpmf(kk[idx], nn[idx], pp[idx])
        log_emission_rdr[idx] = thread_nbinom(kk[idx], nn[idx], pp[idx]) 
        
        if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
            sample_lengths = kwargs["sample_length"]
            logmu_shift = kwargs["logmu_shift"]

            # TODO HACK ask Cong.
            logmu_shift = np.tile(logmu_shift, (1, n_spots))
            
            # NB see ../src/calicost/utils_tumor.py
            tumor_weight = get_tumor_weight(sample_lengths, tumor_prop, log_mu, logmu_shift)
        else:
            tumor_weight = np.tile(tumor_prop, (n_states, 1, 1))

        # NB initialize log_emission
        log_emission_baf = np.zeros((n_states, n_obs, n_spots))
        
        mix_p_A = p_binom[:, None, :] * tumor_weight + 0.5 * (1. - tumor_weight)
        mix_p_B = (1. - p_binom)[:, None, :] * tumor_weight + 0.5 * (1. - tumor_weight)

        aa = mix_p_A * taus[:, None, :]
        bb = mix_p_B * taus[:, None, :]
        
        kk = np.tile(X[:, 1, :], (n_states, 1, 1))
        nn = np.tile(total_bb_RD[:, :], (n_states, 1, 1))

        idx = np.tile(total_bb_RD > 0., (n_states, 1, 1))
        
        # log_emission_baf[idx] = scipy.stats.betabinom.logpmf(kk[idx], nn[idx], aa[idx], bb[idx])
        log_emission_baf[idx] = thread_betabinom(kk[idx], nn[idx], aa[idx], bb[idx])
        
        return log_emission_rdr, log_emission_baf
    
    @staticmethod
    @njit 
    def forward_lattice(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
        '''
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.
        Input
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: n_states * n_observations * n_spots. Log probability.
        Output
            log_alpha: size n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
        '''
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]
        assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
        assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
        # initialize log_alpha
        log_alpha = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0
        for le in lengths:
            # start prob
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities. 
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_alpha[:, cumlen] = log_startprob + np_sum_ax_squeeze(log_emission[:, cumlen, :], axis=1)
            for t in np.arange(1, le):
                for j in np.arange(log_emission.shape[0]):
                    for i in np.arange(log_emission.shape[0]):
                        buf[i] = log_alpha[i, (cumlen + t - 1)] + log_transmat[i, j]
                    log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(log_emission[j, (cumlen + t), :])
            cumlen += le
        return log_alpha
    #
    @staticmethod
    @njit 
    def backward_lattice(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat):
        '''
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: n_states * n_observations * n_spots. Log probability.
        Output
            log_beta: size 2*n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
        '''
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]
        assert np.sum(lengths) == n_obs, "Sum of lengths must be equal to the first dimension of X!"
        assert len(log_startprob) == n_states, "Length of startprob_ must be equal to the first dimension of log_transmat!"
        # initialize log_beta
        log_beta = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0
        for le in lengths:
            # start prob
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities. 
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_beta[:, (cumlen + le - 1)] = 0
            for t in np.arange(le-2, -1, -1):
                for i in np.arange(log_emission.shape[0]):
                    for j in np.arange(log_emission.shape[0]):
                        buf[j] = log_beta[j, (cumlen + t + 1)] + log_transmat[i, j] + np.sum(log_emission[j, (cumlen + t + 1), :])
                    log_beta[i, (cumlen + t)] = mylogsumexp(buf)
            cumlen += le
        return log_beta

    #
    def run_baum_welch_nb_bb(self, X, lengths, n_states, base_nb_mean, total_bb_RD, log_sitewise_transmat=None, tumor_prop=None, \
        fix_NB_dispersion=False, shared_NB_dispersion=False, fix_BB_dispersion=False, shared_BB_dispersion=False, \
        is_diag=False, init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None, max_iter=100, tol=1e-4, **kwargs):
        '''
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            base_nb_mean: size of n_observations * n_spots.
            In NB-BetaBinom model, n_components = 2
        Intermediate
            log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
            alpha: size of n_states. Dispersioon parameter of each HMM state.
        '''
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        assert n_comp == 2
        # initialize NB logmean shift and BetaBinom prob
        log_mu = np.vstack([np.linspace(-0.1, 0.1, n_states) for r in range(n_spots)]).T if init_log_mu is None else init_log_mu
        p_binom = np.vstack([np.linspace(0.05, 0.45, n_states) for r in range(n_spots)]).T if init_p_binom is None else init_p_binom
        # initialize (inverse of) dispersion param in NB and BetaBinom
        alphas = 0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        taus = 30 * np.ones((n_states, n_spots)) if init_taus is None else init_taus
        # initialize start probability and emission probability
        log_startprob = np.log( np.ones(n_states) / n_states )
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1-self.t) / (n_states-1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1,1))
        # initialize log_gamma
        log_gamma = kwargs["log_gamma"] if "log_gamma" in kwargs else None
        # a trick to speed up BetaBinom optimization: taking only unique values of (B allele count, total SNP covering read count)
        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(X[:,0,:], base_nb_mean)
        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(X[:,1,:], total_bb_RD)
        # EM algorithm
        for r in trange(max_iter):
            # E step
            if tumor_prop is None:
                log_emission_rdr, log_emission_baf = hmm_nophasing_v2.compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus)
                log_emission = log_emission_rdr + log_emission_baf
            else:
                # compute mu as adjusted RDR
                if ((not log_gamma is None) or (r > 0)) and ("m" in self.params):
                    logmu_shift = []
                    for c in range(len(kwargs["sample_length"])):
                        this_pred_cnv = np.argmax(log_gamma[:,np.sum(kwargs["sample_length"][:c]):np.sum(kwargs["sample_length"][:(c+1)])], axis=0)%n_states
                        logmu_shift.append( scipy.special.logsumexp(log_mu[this_pred_cnv,:] + np.log(kwargs["lambd"]).reshape(-1,1), axis=0) )
                    logmu_shift = np.vstack(logmu_shift)
                    log_emission_rdr, log_emission_baf = hmm_nophasing_v2.compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, logmu_shift=logmu_shift, sample_length=kwargs["sample_length"])
                else:
                    log_emission_rdr, log_emission_baf = hmm_nophasing_v2.compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop)
                log_emission = log_emission_rdr + log_emission_baf
            log_alpha = hmm_nophasing_v2.forward_lattice(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
            log_beta = hmm_nophasing_v2.backward_lattice(lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat)
            log_gamma = compute_posterior_obs(log_alpha, log_beta)
            log_xi = compute_posterior_transition_nophasing(log_alpha, log_beta, log_transmat, log_emission)
            # M step
            if "s" in self.params:
                new_log_startprob = update_startprob_nophasing(lengths, log_gamma)
                new_log_startprob = new_log_startprob.flatten()
            else:
                new_log_startprob = log_startprob
            if "t" in self.params:
                new_log_transmat = update_transition_nophasing(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat
            if "m" in self.params:
                if tumor_prop is None:
                    new_log_mu, new_alphas = update_emission_params_nb_nophasing_uniqvalues(unique_values_nb, mapping_matrices_nb, log_gamma, alphas, start_log_mu=log_mu, \
                        fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion)
                else:
                    new_log_mu, new_alphas = update_emission_params_nb_nophasing_uniqvalues_mix(unique_values_nb, mapping_matrices_nb, log_gamma, alphas, tumor_prop, start_log_mu=log_mu, \
                        fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion)
            else:
                new_log_mu = log_mu
                new_alphas = alphas
            if "p" in self.params:
                if tumor_prop is None:
                    new_p_binom, new_taus = update_emission_params_bb_nophasing_uniqvalues(unique_values_bb, mapping_matrices_bb, log_gamma, taus, start_p_binom=p_binom, \
                        fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion)
                else:
                    # compute mu as adjusted RDR
                    if ("m" in self.params):
                        mu = []
                        for c in range(len(kwargs["sample_length"])):
                            this_pred_cnv = np.argmax(log_gamma[:,np.sum(kwargs["sample_length"][:c]):np.sum(kwargs["sample_length"][:(c+1)])], axis=0)%n_states
                            mu.append( np.exp(new_log_mu[this_pred_cnv,:]) / np.sum(np.exp(new_log_mu[this_pred_cnv,:]) * kwargs["lambd"].reshape(-1,1), axis=0, keepdims=True) )
                        mu = np.vstack(mu)
                        weighted_tp = (tumor_prop * mu) / (tumor_prop * mu + 1 - tumor_prop)
                    else:
                        weighted_tp = tumor_prop
                    new_p_binom, new_taus = update_emission_params_bb_nophasing_uniqvalues_mix(unique_values_bb, mapping_matrices_bb, log_gamma, taus, weighted_tp, start_p_binom=p_binom, \
                        fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion)
            else:
                new_p_binom = p_binom
                new_taus = taus
            # check convergence
            print( np.mean(np.abs( np.exp(new_log_startprob) - np.exp(log_startprob) )), \
                np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )), \
                np.mean(np.abs(new_log_mu - log_mu)),\
                np.mean(np.abs(new_p_binom - p_binom)) )
            print( np.hstack([new_log_mu, new_p_binom]) )
            if np.mean(np.abs( np.exp(new_log_transmat) - np.exp(log_transmat) )) < tol and \
                np.mean(np.abs(new_log_mu - log_mu)) < tol and np.mean(np.abs(new_p_binom - p_binom)) < tol:
                break
            log_startprob = new_log_startprob
            log_transmat = new_log_transmat
            log_mu = new_log_mu
            alphas = new_alphas
            p_binom = new_p_binom
            taus = new_taus
        return new_log_mu, new_alphas, new_p_binom, new_taus, new_log_startprob, new_log_transmat, log_gamma


