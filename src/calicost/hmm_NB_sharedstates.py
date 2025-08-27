import functools
import inspect
import logging

import numpy as np
import scipy
from scipy import linalg, special
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state
from hmmlearn.hmm import BaseHMM
import statsmodels
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


def convert_params(mean, std):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = mean / std**2
    n = mean * p / (1.0 - p)
    return n, p


class Weighted_NegativeBinomial(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, seed=0, **kwds):
        super(Weighted_NegativeBinomial, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.seed = seed

    #
    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)
        n, p = convert_params(nb_mean, nb_std)
        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf

    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("alpha")

        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        return super(Weighted_NegativeBinomial, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )


class ConstrainedNBHMM(BaseHMM):
    """
    HMM model with NB emission probability and constraint of all cells have the shared hidden state vector.
    A degenerative case is to use pseudobulk UMI count matrix of size G genes by 1 cell.

    Attributes
    ----------
    base_nb_mean : array, shape (n_genes, n_cells)
        Mean expression under diploid state.

    startprob_ : array, shape (n_components)
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    log_mu : array, shape (n_components)
        Shift in log of expression due to CNV. Each CNV states (components) has it's own shift value.

    params : str
        "s" for start probability, "t" for transition probability, "m" for log of expression shift due to CNV, "a" for inverse dispersion of NB distribution.

    Examples
    ----------
    base_nb_mean = eta.reshape(-1,1) * np.sum(totalUMI)
    hmmmodel = ConstrainedNBHMM(n_components=3)
    X = np.vstack( [np.sum(count,axis=0), base_nb_mean] ).T
    hmmmodel.fit( X )
    hmmmodel.predict( X )
    """

    def __init__(
        self,
        n_components=1,
        shared_dispersion=False,
        startprob_prior=1.0,
        transmat_prior=1.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stma",
        init_params="",
    ):
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            params=params,
            verbose=verbose,
            init_params=init_params,
        )
        self.shared_dispersion = shared_dispersion
        # initialize CNV's effect
        self.log_mu = np.linspace(-0.1, 0.1, self.n_components)
        # initialize inverse of dispersion
        self.alphas = np.array([0.01] * self.n_components)
        # self.alphas = 0.01 * np.ones(s(self.n_components, self.n_genes))
        # initialize start probability and transition probability
        self.startprob_ = np.ones(self.n_components) / self.n_components
        t = 0.9
        self.transmat_ = (
            np.ones((self.n_components, self.n_components))
            * (1 - t)
            / (self.n_components - 1)
        )
        np.fill_diagonal(self.transmat_, t)

    #
    def _compute_log_likelihood(self, X):
        """
        Compute log likelihood of X.

        Attributes
        ----------
        X : array_like, shape (n_genes, 2*n_cells)
            First (n_genes, n_cells) is the observed UMI count matrix; second (n_genes, n_cells) is base_nb_mean.

        Returns
        -------
        lpr : array_like, shape (n_genes, n_components)
            Array containing the log probabilities of each data point in X.
        """
        n_genes = X.shape[0]
        n_cells = int(X.shape[1] / 2)
        base_nb_mean = X[:, n_cells:]
        log_prob = np.zeros((n_genes, n_cells, self.n_components))
        for i in range(self.n_components):
            nb_mean = base_nb_mean * np.exp(self.log_mu[i])
            nb_std = np.sqrt(nb_mean + self.alphas[i] * nb_mean**2)
            # nb_std = np.sqrt(nb_mean + self.alphas[i,:].reshape(-1,1) * nb_mean**2)
            n, p = convert_params(nb_mean, nb_std)
            log_prob[:, :, i] = scipy.stats.nbinom.logpmf(X[:, :n_cells], n, p)
        return log_prob.mean(axis=1)

    #
    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        return stats

    #
    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )
        """
        Update sufficient statistics from a given sample.
        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~.BaseHMM._initialize_sufficient_statistics`.
        X : array, shape (n_genes, n_cells)
            Sample sequence.
        lattice : array, shape (n_genes, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm
        posteriors : array, shape (n_genes, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.
        fwdlattice, bwdlattice : array, shape (n_genes, n_components)
            forward and backward probabilities.
        """
        if "m" in self.params or "a" in self.params:
            stats["post"] = posteriors
            stats["obs"] = X
        if "t" in self.params:
            # for each ij, recover sum_t xi_ij from the inferred transition matrix
            bothlattice = fwdlattice + bwdlattice
            loggamma = (bothlattice.T - logsumexp(bothlattice, axis=1)).T

            # denominator for each ij is the sum of gammas over i
            denoms = np.sum(np.exp(loggamma), axis=0)
            # transpose to perform row-wise multiplication
            stats["denoms"] = denoms

    #
    def _do_mstep(self, stats):
        n_genes = stats["obs"].shape[0]
        n_cells = int(stats["obs"].shape[1] / 2)
        base_nb_mean = stats["obs"][:, n_cells:]
        super()._do_mstep(stats)
        if "m" in self.params and "a" in self.params:
            # NB regression fit dispersion and CNV's effect simultaneously
            if not self.shared_dispersion:
                for i in range(self.n_components):
                    model = Weighted_NegativeBinomial(
                        stats["obs"][:, :n_cells].flatten(),
                        np.ones(n_genes * n_cells).reshape(-1, 1),
                        weights=np.repeat(stats["post"][:, i], n_cells),
                        exposure=base_nb_mean.flatten(),
                    )
                    res = model.fit(disp=0, maxiter=500)
                    self.log_mu[i] = res.params[0]
                    self.alphas[i] = res.params[-1]
                    # self.alphas[i,:] = res.params[-1]
            else:
                all_states_nb_mean = np.tile(base_nb_mean.flatten(), self.n_components)
                all_states_y = np.tile(
                    stats["obs"][:, :n_cells].flatten(), self.n_components
                )
                all_states_weights = np.concatenate(
                    [
                        np.repeat(stats["post"][:, i], n_cells)
                        for i in range(self.n_components)
                    ]
                )
                all_states_features = np.zeros(
                    (self.n_components * n_genes * n_cells, self.n_components)
                )
                for i in np.arange(self.n_components):
                    all_states_features[
                        (i * n_genes * n_cells) : ((i + 1) * n_genes * n_cells), i
                    ] = 1
                model = Weighted_NegativeBinomial(
                    all_states_y,
                    all_states_features,
                    weights=all_states_weights,
                    exposure=all_states_nb_mean,
                )
                res = model.fit(disp=0, maxiter=500)
                self.log_mu = res.params[:-1]
                self.alphas[:] = res.params[-1]
                # self.alphas[:,:] = res.params[-1]
                # print(res.params)
        elif "m" in self.params:
            # NB regression fit CNV's effect only
            for i in range(self.n_components):
                model = sm.GLM(
                    stats["obs"].flatten(),
                    np.ones(self.n_genes * self.n_cells).reshape(-1, 1),
                    family=sm.families.NegativeBinomial(alpha=self.alphas[i]),
                    exposure=base_nb_mean.flatten(),
                )
                # model = sm.GLM(stats['obs'][:, :n_cells].flatten(), np.ones(n_genes*n_cells).reshape(-1,1), \
                #             family=sm.families.NegativeBinomial(alpha=np.repeat(self.alphas[i], n_cells)), \
                #             exposure=base_nb_mean.flatten(), var_weights=np.repeat(stats['post'][:,i], n_cells))
                res = model.fit(disp=0, maxiter=500)
                self.log_mu[i] = res.params[0]
        if "t" in self.params:
            # following copied from Matt's code
            denoms = stats["denoms"]
            x = (self.transmat_.T * denoms).T

            # numerator is the sum of ii elements
            num = np.sum(np.diag(x))
            # denominator is the sum of all elements
            denom = np.sum(x)

            # (this is the same as sum_i gamma_i)
            # assert np.isclose(denom, np.sum(denoms))

            stats["diag"] = num / denom
            self.transmat_ = self.form_transition_matrix(stats["diag"])

    #
    def form_transition_matrix(self, diag):
        tol = 1e-10
        diag = np.clip(diag, tol, 1 - tol)

        offdiag = (1 - diag) / (self.n_components - 1)
        transmat_ = np.diag([diag - offdiag] * self.n_components)
        transmat_ += offdiag
        # assert np.all(transmat_ > 0), (diag, offdiag, transmat_)
        return transmat_
