import logging
import numpy as np
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
from calicost.utils_hmm import *
from calicost.utils_distribution_fitting import *
from calicost.hmm_NB_BB_nophasing import *
from calicost.hmm_NB_BB_nophasing_v2 import *
import networkx as nx

logger = logging.getLogger(__name__)

############################################################
# whole inference
############################################################


class hmm_sitewise(object):
    def __init__(self, params="stmp", t=1 - 1e-4):
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
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
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

        logger.info(
            "Computing emission probability for negative binomial & beta binomial (sitewise) with n_spots and n_states = {n_spots} and {n_states}."
        )

        log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))
        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                # expression from NB distribution
                idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0)[0]
                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr, s] * np.exp(log_mu[i, s])
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(
                        X[idx_nonzero_rdr, 0, s], n, p
                    )
                    log_emission_rdr[i + n_states, idx_nonzero_rdr, s] = (
                        log_emission_rdr[i, idx_nonzero_rdr, s]
                    )
                # AF from BetaBinom distribution
                idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]
                if len(idx_nonzero_baf) > 0:
                    log_emission_baf[i, idx_nonzero_baf, s] = (
                        scipy.stats.betabinom.logpmf(
                            X[idx_nonzero_baf, 1, s],
                            total_bb_RD[idx_nonzero_baf, s],
                            p_binom[i, s] * taus[i, s],
                            (1 - p_binom[i, s]) * taus[i, s],
                        )
                    )
                    log_emission_baf[i + n_states, idx_nonzero_baf, s] = (
                        scipy.stats.betabinom.logpmf(
                            X[idx_nonzero_baf, 1, s],
                            total_bb_RD[idx_nonzero_baf, s],
                            (1 - p_binom[i, s]) * taus[i, s],
                            p_binom[i, s] * taus[i, s],
                        )
                    )

        logger.info(
            "Computed emission probability for negative binomial & beta binomial (sitewise)."
        )

        return log_emission_rdr, log_emission_baf

    @staticmethod
    def compute_emission_probability_nb_betabinom_mix(
        X,
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        tumor_prop,
        **kwargs,
    ):
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
        logger.info(
            "Computing emission probability for *mixed* negative binomial & beta binomial (sitewise)."
        )

        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        n_states = log_mu.shape[0]
        # initialize log_emission
        log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))
        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                # expression from NB distribution
                idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0)[0]
                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr, s] * (
                        tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[i, s])
                        + 1
                        - tumor_prop[idx_nonzero_rdr, s]
                    )
                    nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)
                    n, p = convert_params(nb_mean, nb_std)
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(
                        X[idx_nonzero_rdr, 0, s], n, p
                    )
                    log_emission_rdr[i + n_states, idx_nonzero_rdr, s] = (
                        log_emission_rdr[i, idx_nonzero_rdr, s]
                    )
                # AF from BetaBinom distribution
                idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]
                if len(idx_nonzero_baf) > 0:
                    mix_p_A = p_binom[i, s] * tumor_prop[idx_nonzero_baf, s] + 0.5 * (
                        1 - tumor_prop[idx_nonzero_baf, s]
                    )
                    mix_p_B = (1 - p_binom[i, s]) * tumor_prop[
                        idx_nonzero_baf, s
                    ] + 0.5 * (1 - tumor_prop[idx_nonzero_baf, s])
                    log_emission_baf[
                        i, idx_nonzero_baf, s
                    ] += scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        mix_p_A * taus[i, s],
                        mix_p_B * taus[i, s],
                    )
                    log_emission_baf[
                        i + n_states, idx_nonzero_baf, s
                    ] += scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        mix_p_B * taus[i, s],
                        mix_p_A * taus[i, s],
                    )

        logger.info(
            "Computed emission probability for *mixed* negative binomial & beta binomial (sitewise)."
        )

        return log_emission_rdr, log_emission_baf

    @staticmethod
    @njit
    def forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        """
        Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
        Input
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: 2*n_states * n_observations * n_spots. Log probability.
            log_sitewise_transmat: n_observations, the log transition probability of phase switch.
        Output
            log_alpha: size 2n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = int(np.ceil(log_emission.shape[0] / 2))
        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"
        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"
        log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
        # initialize log_alpha
        log_alpha = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0
        for le in lengths:
            # start prob
            combined_log_startprob = np.log(0.5) + np.append(
                log_startprob, log_startprob
            )
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_alpha[:, cumlen] = combined_log_startprob + np_sum_ax_squeeze(
                log_emission[:, cumlen, :], axis=1
            )
            for t in np.arange(1, le):
                phases_switch_mat = np.array(
                    [
                        [
                            log_sitewise_self_transmat[cumlen + t - 1],
                            log_sitewise_transmat[cumlen + t - 1],
                        ],
                        [
                            log_sitewise_transmat[cumlen + t - 1],
                            log_sitewise_self_transmat[cumlen + t - 1],
                        ],
                    ]
                )
                combined_transmat = np.kron(
                    np.exp(phases_switch_mat), np.exp(log_transmat)
                )
                combined_transmat = np.log(combined_transmat)
                for j in np.arange(log_emission.shape[0]):
                    for i in np.arange(log_emission.shape[0]):
                        buf[i] = (
                            log_alpha[i, (cumlen + t - 1)] + combined_transmat[i, j]
                        )
                    log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(
                        log_emission[j, (cumlen + t), :]
                    )
            cumlen += le
        return log_alpha

    @staticmethod
    @njit
    def backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        """
        Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: 2*n_states * n_observations * n_spots. Log probability.
            log_sitewise_transmat: n_observations, the log transition probability of phase switch.
        Output
            log_beta: size 2*n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = int(np.ceil(log_emission.shape[0] / 2))
        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"
        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"
        log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
        # initialize log_beta
        log_beta = np.zeros((log_emission.shape[0], n_obs))
        buf = np.zeros(log_emission.shape[0])
        cumlen = 0
        for le in lengths:
            # start prob
            # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
            # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
            log_beta[:, (cumlen + le - 1)] = 0
            for t in np.arange(le - 2, -1, -1):
                phases_switch_mat = np.array(
                    [
                        [
                            log_sitewise_self_transmat[cumlen + t],
                            log_sitewise_transmat[cumlen + t],
                        ],
                        [
                            log_sitewise_transmat[cumlen + t],
                            log_sitewise_self_transmat[cumlen + t],
                        ],
                    ]
                )
                combined_transmat = np.kron(
                    np.exp(phases_switch_mat), np.exp(log_transmat)
                )
                combined_transmat = np.log(combined_transmat)
                for i in np.arange(log_emission.shape[0]):
                    for j in np.arange(log_emission.shape[0]):
                        buf[j] = (
                            log_beta[j, (cumlen + t + 1)]
                            + combined_transmat[i, j]
                            + np.sum(log_emission[j, (cumlen + t + 1), :])
                        )
                    log_beta[i, (cumlen + t)] = mylogsumexp(buf)
            cumlen += le
        return log_beta

    def run_baum_welch_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=100,
        tol=1e-4,
    ):
        """
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            base_nb_mean: size of n_observations * n_spots.
            In NB-BetaBinom model, n_components = 2
        Intermediate
            log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
            alpha: size of n_states. Dispersioon parameter of each HMM state.
        """
        n_obs = X.shape[0]
        n_comp = X.shape[1]
        n_spots = X.shape[2]
        assert n_comp == 2

        logger.info(
            "Initialize Baum Welch NB logmean shift, BetaBinom prob and dispersion param inverse (sitewise)."
        )

        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for r in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )
        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for r in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )

        # NB initialize (inverse of) dispersion param in NB and BetaBinom
        alphas = (
            0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )
        taus = 30 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        # NB initialize start probability and emission probability
        log_startprob = np.log(np.ones(n_states) / n_states)
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))

        # NB a trick to speed up BetaBinom optimization: taking only unique values of
        #   (B allele count, total SNP covering read count)
        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(
            X[:, 0, :], base_nb_mean
        )
        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(
            X[:, 1, :], total_bb_RD
        )

        for r in trange(max_iter, desc="EM algorithm (sitewise)"):
            logger.info(
                f"Calculating E-step (sitewise) for iteration {r} of {max_iter}."
            )

            if tumor_prop is None:
                log_emission_rdr, log_emission_baf = (
                    hmm_sitewise.compute_emission_probability_nb_betabinom(
                        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                    )
                )
                log_emission = log_emission_rdr + log_emission_baf
            else:
                log_emission_rdr, log_emission_baf = (
                    hmm_sitewise.compute_emission_probability_nb_betabinom_mix(
                        X,
                        base_nb_mean,
                        log_mu,
                        alphas,
                        total_bb_RD,
                        p_binom,
                        taus,
                        tumor_prop,
                    )
                )
                log_emission = log_emission_rdr + log_emission_baf

            log_alpha = hmm_sitewise.forward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_beta = hmm_sitewise.backward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_gamma = compute_posterior_obs(log_alpha, log_beta)

            log_xi = compute_posterior_transition_sitewise(
                log_alpha, log_beta, log_transmat, log_emission
            )

            logger.info(
                f"Calculating M-step (sitewise) for iteration {r} of {max_iter}."
            )

            if "s" in self.params:
                new_log_startprob = update_startprob_sitewise(lengths, log_gamma)
                new_log_startprob = new_log_startprob.flatten()
            else:
                new_log_startprob = log_startprob
            if "t" in self.params:
                new_log_transmat = update_transition_sitewise(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat
            if "m" in self.params:
                # new_log_mu, new_alphas = update_emission_params_nb_sitewise(X[:,0,:], log_gamma, base_nb_mean, alphas, start_log_mu=log_mu, \
                #     fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion)
                if tumor_prop is None:
                    new_log_mu, new_alphas = (
                        update_emission_params_nb_sitewise_uniqvalues(
                            unique_values_nb,
                            mapping_matrices_nb,
                            log_gamma,
                            base_nb_mean,
                            alphas,
                            start_log_mu=log_mu,
                            fix_NB_dispersion=fix_NB_dispersion,
                            shared_NB_dispersion=shared_NB_dispersion,
                        )
                    )
                else:
                    new_log_mu, new_alphas = (
                        update_emission_params_nb_sitewise_uniqvalues_mix(
                            unique_values_nb,
                            mapping_matrices_nb,
                            log_gamma,
                            base_nb_mean,
                            alphas,
                            tumor_prop,
                            start_log_mu=log_mu,
                            fix_NB_dispersion=fix_NB_dispersion,
                            shared_NB_dispersion=shared_NB_dispersion,
                        )
                    )
            else:
                new_log_mu = log_mu
                new_alphas = alphas
            if "p" in self.params:
                if tumor_prop is None:
                    new_p_binom, new_taus = (
                        update_emission_params_bb_sitewise_uniqvalues(
                            unique_values_bb,
                            mapping_matrices_bb,
                            log_gamma,
                            total_bb_RD,
                            taus,
                            start_p_binom=p_binom,
                            fix_BB_dispersion=fix_BB_dispersion,
                            shared_BB_dispersion=shared_BB_dispersion,
                        )
                    )
                else:
                    new_p_binom, new_taus = (
                        update_emission_params_bb_sitewise_uniqvalues_mix(
                            unique_values_bb,
                            mapping_matrices_bb,
                            log_gamma,
                            total_bb_RD,
                            taus,
                            tumor_prop,
                            start_p_binom=p_binom,
                            fix_BB_dispersion=fix_BB_dispersion,
                            shared_BB_dispersion=shared_BB_dispersion,
                        )
                    )
            else:
                new_p_binom = p_binom
                new_taus = taus

            logger.info(
                f"EM convergence metrics (sitewise): {np.mean(np.abs(np.exp(new_log_startprob) - np.exp(log_startprob)))}, {np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat)))}, {np.mean(np.abs(new_log_mu - log_mu))}, {np.mean(np.abs(new_p_binom - p_binom))}"
            )

            # logger.info(np.hstack([new_log_mu, new_p_binom]))

            if (
                np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))) < tol
                and np.mean(np.abs(new_log_mu - log_mu)) < tol
                and np.mean(np.abs(new_p_binom - p_binom)) < tol
            ):
                break
            
            log_startprob = new_log_startprob
            log_transmat = new_log_transmat
            log_mu = new_log_mu
            alphas = new_alphas
            p_binom = new_p_binom
            taus = new_taus

        logger.info("Computed Baum-Welch (sitewise).")

        logger.info(f"Fitted (mu, p):\n{np.hstack([new_log_mu, new_p_binom])}")
        logger.info(f"Fitted (alphas, taus):\n{np.hstack([new_alphas, new_taus])}")
        
        return (
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            new_log_startprob,
            new_log_transmat,
            log_gamma,
        )


def posterior_nb_bb_sitewise(
    X,
    lengths,
    base_nb_mean,
    log_mu,
    alphas,
    total_bb_RD,
    p_binom,
    taus,
    log_startprob,
    log_transmat,
    log_sitewise_transmat,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    lengths : array, shape (n_chromosomes,)
        Number genes (or bins) per chromosome, the sum of this vector should be equal to n_observations.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    log_mu : array, shape (n_states, n_spots)
        Log read depth shift of each CNV states.

    alphas : array, shape (n_states, n_spots)
        Inverse of dispersion in NB distribution of each state.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.

    p_binom : array, shape (n_states, n_spots)
        MAF of each CNV states.

    taus : array, shape (n_states, n_spots)
        Inverse of dispersion of Beta-Binomial distribution of each state.

    log_startprob : array, shape (n_states,)
        Log of start probability.

    log_transmat : array, shape (n_states, n_states)
        Log of transition probability across states.

    log_sitewise_transmat : array, shape (n_observations)
        Log of phase switch probability of each gene (or bin).
    """
    log_emission_rdr, log_emission_baf = (
        hmm_sitewise.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    )
    log_emission = log_emission_rdr + log_emission_baf
    log_alpha = hmm_sitewise.forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    )
    log_beta = hmm_sitewise.backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    )
    log_gamma = compute_posterior_obs(log_alpha, log_beta)
    return log_gamma


def loglikelihood_nb_bb_sitewise(
    X,
    lengths,
    base_nb_mean,
    log_mu,
    alphas,
    total_bb_RD,
    p_binom,
    taus,
    log_startprob,
    log_transmat,
    log_sitewise_transmat,
):
    """
    Attributes
    ----------
    X : array, shape (n_observations, n_components, n_spots)
        Observed expression UMI count and allele frequency UMI count.

    lengths : array, shape (n_chromosomes,)
        Number genes (or bins) per chromosome, the sum of this vector should be equal to n_observations.

    base_nb_mean : array, shape (n_observations, n_spots)
        Mean expression under diploid state.

    log_mu : array, shape (n_states, n_spots)
        Log read depth shift of each CNV states.

    alphas : array, shape (n_states, n_spots)
        Inverse of dispersion in NB distribution of each state.

    total_bb_RD : array, shape (n_observations, n_spots)
        SNP-covering reads for both REF and ALT across genes along genome.

    p_binom : array, shape (n_states, n_spots)
        MAF of each CNV states.

    taus : array, shape (n_states, n_spots)
        Inverse of dispersion of Beta-Binomial distribution of each state.

    log_startprob : array, shape (n_states,)
        Log of start probability.

    log_transmat : array, shape (n_states, n_states)
        Log of transition probability across states.

    log_sitewise_transmat : array, shape (n_observations)
        Log of phase switch probability of each gene (or bin).
    """
    log_emission_rdr, log_emission_baf = (
        hmm_sitewise.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    )
    log_emission = log_emission_rdr + log_emission_baf
    log_alpha = hmm_sitewise.forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    )
    return (
        np.sum(scipy.special.logsumexp(log_alpha[:, np.cumsum(lengths) - 1], axis=0)),
        log_alpha,
    )


def viterbi_nb_bb_sitewise(
    X,
    lengths,
    base_nb_mean,
    log_mu,
    alphas,
    total_bb_RD,
    p_binom,
    taus,
    log_startprob,
    log_transmat,
    log_sitewise_transmat,
):
    """
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            exposures: size of n_observations * n_spots.
            base_prob: size of n_observations. The expression probability derived from normal spots.
            log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
            alpha: size of n_states. Dispersioon parameter of each HMM state.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
        Output
    #        log_prob: a scalar.
            labels: size of n_observations.
        Intermediate
            log_emission: n_states * n_observations * n_spots. Log probability.
            log_v: n_states * n_observations per chromosome. Log of viterbi DP table. v[i,t] = max_{q_1, ..., q_{t-1}} P(o_1, q_1, ..., o_{t-1}, q_{t-1}, o_t, q_t=i | lambda).
    """
    n_obs = X.shape[0]
    n_comp = X.shape[1]
    n_spots = X.shape[2]
    n_states = log_transmat.shape[0]
    log_sitewise_self_transmat = np.log(1 - np.exp(log_sitewise_transmat))
    log_emission_rdr, log_emission_baf = (
        hmm_sitewise.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    )
    log_emission = log_emission_rdr + log_emission_baf
    # initialize viterbi DP table and backtracking table
    labels = np.array([])
    merged_labels = np.array([])
    cumlen = 0
    for le in lengths:
        log_v = np.zeros((2 * n_states, le))
        bt = np.zeros((2 * n_states, le))
        for t in np.arange(le):
            if cumlen == 0 and t == 0:
                log_v[:, 0] = (
                    np.mean(log_emission[:, 0, :], axis=1)
                    + np.append(log_startprob, log_startprob)
                    + np.log(0.5)
                )
                continue
            for i in np.arange(2 * n_states):
                if t > 0:
                    tmp = (
                        log_v[:, (t - 1)]
                        + np.append(
                            log_transmat[:, i - n_states * int(i / n_states)],
                            log_transmat[:, i - n_states * int(i / n_states)],
                        )
                        + np.sum(log_emission[i, (cumlen + t), :])
                    )
                else:
                    tmp = np.append(
                        log_startprob[i - n_states * int(i / n_states)],
                        log_startprob[i - n_states * int(i / n_states)],
                    ) + np.sum(log_emission[i, (cumlen + t), :])
                bt[i, t] = np.argmax(tmp)
                log_v[i, t] = np.max(tmp)
        # backtracking to get the sequence
        chr_labels = [np.argmax(log_v[:, -1])]

        if cumlen == 0:
            for t2 in np.arange(le - 1, 0, -1):
                chr_labels.append(int(bt[chr_labels[-1], t2]))
        else:
            for t2 in np.arange(le - 2, -1, -1):
                chr_labels.append(int(bt[chr_labels[-1], t2]))

        chr_labels = np.array(chr_labels[::-1]).astype(int)
        # merge two phases
        chr_merged_labels = copy.copy(chr_labels)
        chr_merged_labels[chr_merged_labels >= n_states] = (
            chr_merged_labels[chr_merged_labels >= n_states] - n_states
        )

        if cumlen == 0:
            labels = chr_labels
            merged_labels = chr_merged_labels
        else:
            labels = np.append(labels, chr_labels)
            merged_labels = np.append(merged_labels, chr_merged_labels)

        cumlen += le
    return labels, merged_labels


def pipeline_baum_welch(
    output_prefix,
    X,
    lengths,
    n_states,
    base_nb_mean,
    total_bb_RD,
    log_sitewise_transmat,
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    params="smp",
    t=1 - 1e-6,
    random_state=0,
    in_log_space=True,
    only_minor=False,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    **kwargs,
):
    """
    tumor_prop : array, (n_obs, n_spots)
        Probability of sequencing a tumor read. (tumor cell proportion weighted by ploidy)

    """
    # initialization
    n_spots = X.shape[2]
    if ((init_log_mu is None) and ("m" in params)) or (
        (init_p_binom is None) and ("p" in params)
    ):
        tmp_log_mu, tmp_p_binom = initialization_by_gmm(
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=in_log_space,
            only_minor=only_minor,
        )
        if (init_log_mu is None) and ("m" in params):
            init_log_mu = tmp_log_mu
        if (init_p_binom is None) and ("p" in params):
            init_p_binom = tmp_p_binom

    logger.info(f"Initial mu:\n{init_log_mu}")
    logger.info(f"Initial p:\n{init_p_binom}")

    hmmmodel = hmmclass(params=params, t=t)
    remain_kwargs = {
        k: v for k, v in kwargs.items() if k in ["lambd", "sample_length", "log_gamma"]
    }
    (
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        new_log_startprob,
        new_log_transmat,
        log_gamma,
    ) = hmmmodel.run_baum_welch_nb_bb(
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        tumor_prop,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        init_alphas=init_alphas,
        init_taus=init_taus,
        max_iter=max_iter,
        tol=tol,
        **remain_kwargs,
    )

    # likelihood
    if tumor_prop is None:
        log_emission_rdr, log_emission_baf = (
            hmmclass.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                new_log_mu,
                new_alphas,
                total_bb_RD,
                new_p_binom,
                new_taus,
            )
        )
        log_emission = log_emission_rdr + log_emission_baf
    else:
        if ("m" in params) and ("sample_length" in kwargs):
            logmu_shift = []
            for c in range(len(kwargs["sample_length"])):
                this_pred_cnv = (
                    np.argmax(
                        log_gamma[
                            :,
                            np.sum(kwargs["sample_length"][:c]) : np.sum(
                                kwargs["sample_length"][: (c + 1)]
                            ),
                        ],
                        axis=0,
                    )
                    % n_states
                )
                logmu_shift.append(
                    scipy.special.logsumexp(
                        new_log_mu[this_pred_cnv, :]
                        + np.log(kwargs["lambd"]).reshape(-1, 1),
                        axis=0,
                    )
                )
            logmu_shift = np.vstack(logmu_shift)
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    X,
                    base_nb_mean,
                    new_log_mu,
                    new_alphas,
                    total_bb_RD,
                    new_p_binom,
                    new_taus,
                    tumor_prop,
                    logmu_shift=logmu_shift,
                    sample_length=kwargs["sample_length"],
                )
            )
        else:
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    X,
                    base_nb_mean,
                    new_log_mu,
                    new_alphas,
                    total_bb_RD,
                    new_p_binom,
                    new_taus,
                    tumor_prop,
                )
            )
        # log_emission_rdr, log_emission_baf = hmmclass.compute_emission_probability_nb_betabinom_mix(X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus, tumor_prop)
        log_emission = log_emission_rdr + log_emission_baf
    log_alpha = hmmclass.forward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )
    llf = np.sum(scipy.special.logsumexp(log_alpha[:, np.cumsum(lengths) - 1], axis=0))

    log_beta = hmmclass.backward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )
    log_gamma = compute_posterior_obs(log_alpha, log_beta)
    pred = np.argmax(log_gamma, axis=0)
    pred_cnv = pred % n_states

    # save results
    if not output_prefix is None:
        tmp = np.log10(1 - t)
        np.savez(
            f"{output_prefix}_nstates{n_states}_{params}_{tmp:.0f}_seed{random_state}.npz",
            new_log_mu=new_log_mu,
            new_alphas=new_alphas,
            new_p_binom=new_p_binom,
            new_taus=new_taus,
            new_log_startprob=new_log_startprob,
            new_log_transmat=new_log_transmat,
            log_gamma=log_gamma,
            pred_cnv=pred_cnv,
            llf=llf,
        )
    else:
        res = {
            "new_log_mu": new_log_mu,
            "new_alphas": new_alphas,
            "new_p_binom": new_p_binom,
            "new_taus": new_taus,
            "new_log_startprob": new_log_startprob,
            "new_log_transmat": new_log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": pred_cnv,
            "llf": llf,
        }
        return res


def eval_neymanpearson_bafonly(
    log_emission_baf_c1, pred_c1, log_emission_baf_c2, pred_c2, bidx, n_states, res, p
):
    assert (
        log_emission_baf_c1.shape[0] == n_states
        or log_emission_baf_c1.shape[0] == 2 * n_states
    )
    # likelihood under the corresponding state
    llf_original = np.append(
        log_emission_baf_c1[pred_c1[bidx], bidx],
        log_emission_baf_c2[pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # likelihood under the switched state
    if log_emission_baf_c1.shape[0] == 2 * n_states:
        if (res["new_p_binom"][p[0], 0] > 0.5) == (res["new_p_binom"][p[1], 0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
    else:
        switch_pred_c1 = pred_c2
        switch_pred_c2 = pred_c1
    llf_switch = np.append(
        log_emission_baf_c1[switch_pred_c1[bidx], bidx],
        log_emission_baf_c2[switch_pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # log likelihood difference
    return np.mean(llf_original) - np.mean(llf_switch)


def eval_neymanpearson_rdrbaf(
    log_emission_rdr_c1,
    log_emission_baf_c1,
    pred_c1,
    log_emission_rdr_c2,
    log_emission_baf_c2,
    pred_c2,
    bidx,
    n_states,
    res,
    p,
):
    assert (
        log_emission_baf_c1.shape[0] == n_states
        or log_emission_baf_c1.shape[0] == 2 * n_states
    )
    # likelihood under the corresponding state
    llf_original = np.append(
        log_emission_rdr_c1[pred_c1[bidx], bidx]
        + log_emission_baf_c1[pred_c1[bidx], bidx],
        log_emission_rdr_c2[pred_c2[bidx], bidx]
        + log_emission_baf_c2[pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # likelihood under the switched state
    if log_emission_baf_c1.shape[0] == 2 * n_states:
        if (res["new_p_binom"][p[0], 0] > 0.5) == (res["new_p_binom"][p[1], 0] > 0.5):
            switch_pred_c1 = n_states * (pred_c1 >= n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 >= n_states) + (pred_c1 % n_states)
        else:
            switch_pred_c1 = n_states * (pred_c1 < n_states) + (pred_c2 % n_states)
            switch_pred_c2 = n_states * (pred_c2 < n_states) + (pred_c1 % n_states)
    else:
        switch_pred_c1 = pred_c2
        switch_pred_c2 = pred_c1
    llf_switch = np.append(
        log_emission_rdr_c1[switch_pred_c1[bidx], bidx]
        + log_emission_baf_c1[switch_pred_c1[bidx], bidx],
        log_emission_rdr_c2[switch_pred_c2[bidx], bidx]
        + log_emission_baf_c2[switch_pred_c2[bidx], bidx],
    ).reshape(-1, 1)
    # log likelihood difference
    return np.mean(llf_original) - np.mean(llf_switch)


def compute_neymanpearson_stats(
    X, base_nb_mean, total_bb_RD, res, params, tumor_prop, hmmclass
):
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    n_clones = X.shape[2]
    lambd = np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean)
    #
    if tumor_prop is None:
        log_emission_rdr, log_emission_baf = (
            hmmclass.compute_emission_probability_nb_betabinom(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                base_nb_mean.flatten("F").reshape(-1, 1),
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD.flatten("F").reshape(-1, 1),
                res["new_p_binom"],
                res["new_taus"],
            )
        )
    else:
        if "m" in params:
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
                        res["new_log_mu"][this_pred_cnv, :]
                        + np.log(lambd).reshape(-1, 1),
                        axis=0,
                    )
                )
            logmu_shift = np.vstack(logmu_shift)
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                    logmu_shift=logmu_shift,
                    sample_length=np.ones(n_clones, dtype=int) * n_obs,
                )
            )
        else:
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                )
            )
    log_emission_rdr = log_emission_rdr.reshape(
        (log_emission_rdr.shape[0], n_obs, n_clones), order="F"
    )
    log_emission_baf = log_emission_baf.reshape(
        (log_emission_baf.shape[0], n_obs, n_clones), order="F"
    )
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))
    reshaped_pred_cnv = reshaped_pred % n_states
    all_test_statistics = {
        (c1, c2): [] for c1 in range(n_clones) for c2 in range(c1 + 1, n_clones)
    }
    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            # unmergeable_bincount = 0
            unique_pair_states = [
                x
                for x in np.unique(reshaped_pred_cnv[np.array([c1, c2]), :], axis=1).T
                if x[0] != x[1]
            ]
            list_t_neymanpearson = []
            for p in unique_pair_states:
                bidx = np.where(
                    (reshaped_pred_cnv[c1, :] == p[0])
                    & (reshaped_pred_cnv[c2, :] == p[1])
                )[0]
                if "m" in params and "p" in params:
                    t_neymanpearson = eval_neymanpearson_rdrbaf(
                        log_emission_rdr[:, :, c1],
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_rdr[:, :, c2],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )
                elif "p" in params:
                    t_neymanpearson = eval_neymanpearson_bafonly(
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )
                all_test_statistics[(c1, c2)].append((p[0], p[1], t_neymanpearson))

    return all_test_statistics


def similarity_components_rdrbaf_neymanpearson(
    X,
    base_nb_mean,
    total_bb_RD,
    res,
    threshold=2.0,
    minlength=10,
    topk=10,
    params="smp",
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    **kwargs,
):
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    n_clones = X.shape[2]

    logger.info(
        "Computing similarity_components_rdrbaf_neymanpearson for (n_obs, n_states, n_clones) = ({n_obs}, {n_states}, {n_clones})."
    )

    G = nx.Graph()
    G.add_nodes_from(np.arange(n_clones))

    lambd = np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean)

    if tumor_prop is None:
        log_emission_rdr, log_emission_baf = (
            hmmclass.compute_emission_probability_nb_betabinom(
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                base_nb_mean.flatten("F").reshape(-1, 1),
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD.flatten("F").reshape(-1, 1),
                res["new_p_binom"],
                res["new_taus"],
            )
        )
    else:
        if "m" in params:
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
                        res["new_log_mu"][this_pred_cnv, :]
                        + np.log(lambd).reshape(-1, 1),
                        axis=0,
                    )
                )
            logmu_shift = np.vstack(logmu_shift)
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                    logmu_shift=logmu_shift,
                    sample_length=np.ones(n_clones, dtype=int) * n_obs,
                )
            )
        else:
            log_emission_rdr, log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    res["new_log_mu"],
                    res["new_alphas"],
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    res["new_p_binom"],
                    res["new_taus"],
                    tumor_prop,
                )
            )
    log_emission_rdr = log_emission_rdr.reshape(
        (log_emission_rdr.shape[0], n_obs, n_clones), order="F"
    )
    log_emission_baf = log_emission_baf.reshape(
        (log_emission_baf.shape[0], n_obs, n_clones), order="F"
    )
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))
    reshaped_pred_cnv = reshaped_pred % n_states

    all_test_statistics = []

    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            unique_pair_states = [
                x
                for x in np.unique(reshaped_pred_cnv[np.array([c1, c2]), :], axis=1).T
                if x[0] != x[1]
            ]
            list_t_neymanpearson = []
            for p in unique_pair_states:
                bidx = np.where(
                    (reshaped_pred_cnv[c1, :] == p[0])
                    & (reshaped_pred_cnv[c2, :] == p[1])
                )[0]

                if "m" in params and "p" in params:
                    t_neymanpearson = eval_neymanpearson_rdrbaf(
                        log_emission_rdr[:, :, c1],
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_rdr[:, :, c2],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )
                elif "p" in params:
                    t_neymanpearson = eval_neymanpearson_bafonly(
                        log_emission_baf[:, :, c1],
                        reshaped_pred[c1, :],
                        log_emission_baf[:, :, c2],
                        reshaped_pred[c2, :],
                        bidx,
                        n_states,
                        res,
                        p,
                    )

                # TODO
                logger.info(f"{c1}, {c2}, {p}, {len(bidx)}, {t_neymanpearson}")

                all_test_statistics.append([c1, c2, p, t_neymanpearson])

                if len(bidx) >= minlength:
                    list_t_neymanpearson.append(t_neymanpearson)
            if (
                len(list_t_neymanpearson) == 0
                or np.max(list_t_neymanpearson) < threshold
            ):
                max_v = (
                    np.max(list_t_neymanpearson)
                    if len(list_t_neymanpearson) > 0
                    else 1e-3
                )
                G.add_weighted_edges_from([(c1, c2, max_v)])

    logger.info("Computing Maximal cliques.")

    cliques = []

    for x in nx.find_cliques(G):
        this_len = len(x)
        this_weights = (
            np.sum([G.get_edge_data(a, b)["weight"] for a in x for b in x if a != b])
            / 2
        )
        cliques.append((x, this_len, this_weights))

    cliques.sort(key=lambda x: (-x[1], x[2]))

    covered_nodes = set()
    merging_groups = []

    for c in cliques:
        if len(set(c[0]) & covered_nodes) == 0:
            merging_groups.append(list(c[0]))
            covered_nodes = covered_nodes | set(c[0])

    for c in range(n_clones):
        if not (c in covered_nodes):
            merging_groups.append([c])
            covered_nodes.add(c)

    merging_groups.sort(key=lambda x: np.min(x))

    # NB clone assignment after merging
    map_clone_id = {}

    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i

    new_assignment = np.array([map_clone_id[x] for x in res["new_assignment"]])
    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
    merged_res["pred_cnv"] = np.concatenate(
        [
            res["pred_cnv"][(c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )
    merged_res["log_gamma"] = np.hstack(
        [
            res["log_gamma"][:, (c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )

    logger.info("Computed similarity_components_rdrbaf_neymanpearson.")

    return merging_groups, merged_res


def combine_similar_states_across_clones(
    X,
    base_nb_mean,
    total_bb_RD,
    res,
    params="smp",
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    merge_threshold=0.1,
    **kwargs,
):
    n_clones = X.shape[2]
    n_obs = X.shape[0]
    n_states = res["new_p_binom"].shape[0]
    reshaped_pred = np.argmax(res["log_gamma"], axis=0).reshape((X.shape[2], -1))
    reshaped_pred_cnv = reshaped_pred % n_states

    all_test_statistics = compute_neymanpearson_stats(
        X, base_nb_mean, total_bb_RD, res, params, tumor_prop, hmmclass
    )

    # NB make the pair of states consistent between clone c1 and clone c2 if their t_neymanpearson test statistics is small
    for c1 in range(n_clones):
        for c2 in range(c1 + 1, n_clones):
            list_t_neymanpearson = all_test_statistics[(c1, c2)]
            for p1, p2, t_neymanpearson in list_t_neymanpearson:
                if t_neymanpearson < merge_threshold:
                    c_keep = (
                        c1
                        if np.sum(total_bb_RD[:, c1]) > np.sum(total_bb_RD[:, c2])
                        else c2
                    )
                    c_change = c2 if c_keep == c1 else c1
                    bidx = np.where(
                        (reshaped_pred_cnv[c1, :] == p1)
                        & (reshaped_pred_cnv[c2, :] == p2)
                    )[0]
                    res["pred_cnv"][(c_change * n_obs) : (c_change * n_obs + n_obs)][
                        bidx
                    ] = res["pred_cnv"][(c_keep * n_obs) : (c_keep * n_obs + n_obs)][
                        bidx
                    ]
                    print(
                        f"Merging states {[p1,p2]} in clone {c1} and clone {c2}. NP statistics = {t_neymanpearson}"
                    )
    return res
