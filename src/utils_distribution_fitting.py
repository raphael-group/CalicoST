import functools
import inspect
import logging

import numpy as np
import scipy
from scipy import linalg, special
from scipy.special import logsumexp
from scipy.stats import betabinom
from sklearn import cluster
from sklearn.utils import check_random_state
import statsmodels
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


def convert_params(mean, std):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = mean/std**2
    n = mean*p/(1.0 - p)
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
        self.exog_names.append('alpha')

        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)
        
        return super(Weighted_NegativeBinomial, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinom(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, **kwds):
        super(Weighted_BetaBinom, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1 - self.exog @ params[:-1]) * params[-1]
        llf = betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.5 * np.ones(self.nparams), 1)
        
        return super(Weighted_BetaBinom, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinom_fixdispersion(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, **kwds):
        super(Weighted_BetaBinom_fixdispersion, self).__init__(endog, exog, **kwds)
        self.tau = tau
        self.weights = weights
        self.exposure = exposure
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params) * self.tau
        b = (1 - self.exog @ params) * self.tau
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.1 * np.ones(self.nparams)
        
        return super(Weighted_BetaBinom_fixdispersion, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
                                               

class Weighted_BetaBinomMixture(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, fixed_mixture_prop=0.5, **kwds):
        super(Weighted_BetaBinomMixture, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.fixed_mixture_prop = fixed_mixture_prop
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1 - self.exog @ params[:-1]) * params[-1]
        llf1 = betabinom.logpmf(self.endog, self.exposure, a, b) + np.log(self.fixed_mixture_prop)
        llf2 = betabinom.logpmf(self.endog, self.exposure, b, a) + np.log(self.fixed_mixture_prop)
        combined_llf = logsumexp( np.stack([llf1, llf2]), axis=0 )
        neg_sum_llf = -combined_llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.01 * np.ones(self.nparams), 1)
        
        return super(Weighted_BetaBinomMixture, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinomMixture_fixdispersion(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, fixed_mixture_prop=0.5, **kwds):
        super(Weighted_BetaBinomMixture_fixdispersion, self).__init__(endog, exog, **kwds)
        self.tau = tau
        self.weights = weights
        self.exposure = exposure
        self.fixed_mixture_prop = fixed_mixture_prop
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params) * self.tau
        b = (1 - self.exog @ params) * self.tau
        llf1 = betabinom.logpmf(self.endog, self.exposure, a, b) + np.log(self.fixed_mixture_prop)
        llf2 = betabinom.logpmf(self.endog, self.exposure, b, a) + np.log(self.fixed_mixture_prop)
        combined_llf = logsumexp( np.stack([llf1, llf2]), axis=0 )
        neg_sum_llf = -combined_llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.01 * np.ones(self.nparams)
        
        return super(Weighted_BetaBinomMixture_fixdispersion, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)