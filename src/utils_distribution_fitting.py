import functools
import inspect
import logging

import numpy as np
import scipy
from scipy import linalg, special
from scipy.special import logsumexp, loggamma
import scipy.integrate
import scipy.stats
from numba import jit, njit
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


class Weighted_NegativeBinomial_sepphi(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, numphi, seed=0, **kwds):
        super(Weighted_NegativeBinomial_sepphi, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
        self.numphi = numphi
    #
    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-self.numphi]) * self.exposure
        nb_std = np.sqrt(nb_mean + self.exog @ params[-self.numphi:] * nb_mean**2)
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
                start_params = np.append(0.1 * np.ones(self.nparams - self.numphi), 0.01 * np.ones(self.numphi))

        return super(Weighted_NegativeBinomial_sepphi, self).fit(start_params=start_params,
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
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1)
        return super(Weighted_BetaBinom, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinom_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_mix, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1] * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * params[-1]
        b = ((1 - self.exog @ params[:-1]) * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * params[-1]
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1)
        return super(Weighted_BetaBinom_mix, self).fit(start_params=start_params,
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


class Weighted_BetaBinom_fixdispersion_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_fixdispersion_mix, self).__init__(endog, exog, **kwds)
        self.tau = tau
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop
    #
    def nloglikeobs(self, params):
        a = (self.exog @ params * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * self.tau
        b = ((1 - self.exog @ params) * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * self.tau
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
        
        return super(Weighted_BetaBinom_fixdispersion_mix, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinom_dispersiononly(GenericLikelihoodModel):
    def __init__(self, endog, exog, baf, weights, exposure, **kwds):
        super(Weighted_BetaBinom_dispersiononly, self).__init__(endog, exog, **kwds)
        self.baf = baf
        self.weights = weights
        self.exposure = exposure
    #
    def nloglikeobs(self, params):
        a = self.baf * params[-1]
        b = (1 - self.baf) * params[-1]
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.ones(1)
        return super(Weighted_BetaBinom_dispersiononly, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
                                               

# class Weighted_BetaBinomMixture_dispersiononly(GenericLikelihoodModel):
#     def __init__(self, endog, exog, baf, weights, exposure, **kwds):
#         super(Weighted_BetaBinomMixture_dispersiononly, self).__init__(endog, exog, **kwds)
#         self.baf = baf
#         self.weights = weights
#         self.exposure = exposure
#     #
#     def nloglikeobs(self, params):
#         a = self.baf * params[-1]
#         b = (1 - self.baf) * params[-1]
#         llf1 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
#         llf2 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, b, a)
#         combined_llf = logsumexp( np.stack([llf1, llf2]), axis=0 )
#         neg_sum_llf = -combined_llf.dot(self.weights)
#         return neg_sum_llf
#     #
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
#         self.exog_names.append("tau")
#         if start_params is None:
#             if hasattr(self, 'start_params'):
#                 start_params = self.start_params
#             else:
#                 start_params = np.ones(1)
#         return super(Weighted_BetaBinomMixture_dispersiononly, self).fit(start_params=start_params,
#                                                maxiter=maxiter, maxfun=maxfun,
#                                                **kwds)


# class Weighted_BetaBinomMixture(GenericLikelihoodModel):
#     def __init__(self, endog, exog, weights, exposure, fixed_mixture_prop=0.5, **kwds):
#         super(Weighted_BetaBinomMixture, self).__init__(endog, exog, **kwds)
#         self.weights = weights
#         self.exposure = exposure
#         self.fixed_mixture_prop = fixed_mixture_prop
#     #
#     def nloglikeobs(self, params):
#         a = (self.exog @ params[:-1]) * params[-1]
#         b = (1 - self.exog @ params[:-1]) * params[-1]
#         llf1 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b) + np.log(self.fixed_mixture_prop)
#         llf2 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, b, a) + np.log(self.fixed_mixture_prop)
#         combined_llf = logsumexp( np.stack([llf1, llf2]), axis=0 )
#         neg_sum_llf = -combined_llf.dot(self.weights)
#         return neg_sum_llf
#     #
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
#         self.exog_names.append("tau")
#         if start_params is None:
#             if hasattr(self, 'start_params'):
#                 start_params = self.start_params
#             else:
#                 start_params = np.append(0.01 * np.ones(self.nparams), 1)
        
#         return super(Weighted_BetaBinomMixture, self).fit(start_params=start_params,
#                                                maxiter=maxiter, maxfun=maxfun,
#                                                **kwds)


# class Weighted_BetaBinomMixture_fixdispersion(GenericLikelihoodModel):
#     def __init__(self, endog, exog, tau, weights, exposure, fixed_mixture_prop=0.5, **kwds):
#         super(Weighted_BetaBinomMixture_fixdispersion, self).__init__(endog, exog, **kwds)
#         self.tau = tau
#         self.weights = weights
#         self.exposure = exposure
#         self.fixed_mixture_prop = fixed_mixture_prop
#     #
#     def nloglikeobs(self, params):
#         a = (self.exog @ params) * self.tau
#         b = (1 - self.exog @ params) * self.tau
#         llf1 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b) + np.log(self.fixed_mixture_prop)
#         llf2 = scipy.stats.betabinom.logpmf(self.endog, self.exposure, b, a) + np.log(self.fixed_mixture_prop)
#         combined_llf = logsumexp( np.stack([llf1, llf2]), axis=0 )
#         neg_sum_llf = -combined_llf.dot(self.weights)
#         return neg_sum_llf
#     #
#     def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
#         if start_params is None:
#             if hasattr(self, 'start_params'):
#                 start_params = self.start_params
#             else:
#                 start_params = 0.01 * np.ones(self.nparams)
        
#         return super(Weighted_BetaBinomMixture_fixdispersion, self).fit(start_params=start_params,
#                                                maxiter=maxiter, maxfun=maxfun,
#                                                **kwds)


"""
@njit
def poilog_maxf_single(x, mu, sig):
    z=0
    d=100
    while d>0.00001:
        if x-1-np.exp(z)-1/sig*(z-mu)>0:
            z=z+d
        else:
            z=z-d
        d=d/2
    return z

@njit
def poilog_maxf(x, mu, sig):
    assert len(x) == len(mu)
    z = np.zeros(len(x))
    for i in range(len(mu)):
        d=100
        while d>0.00001:
            if x[i]-1-np.exp(z[i])-1/sig*(z[i]-mu[i])>0:
                z[i] = z[i] + d
            else:
                z[i] = z[i] - d
            d = d/2
    return z


@njit
def poilog_upper_single(x, m, mu, sig):
    mf = (x-1)*m-np.exp(m)-0.5/sig*((m-mu)*(m-mu))
    z = m+20
    d = 10
    while d>0.000001:
        if (x-1)*z-np.exp(z)-0.5/sig*((z-mu)*(z-mu))-mf+np.log(1000000) > 0:
            z=z+d
        else:
            z=z-d
        d=d/2
    return z


@njit
def poilog_upper(x, m, mu, sig):
    mf = (x-1)*m-np.exp(m)-0.5/sig*((m-mu)*(m-mu))
    z = m+20
    assert len(x) == len(mu)
    for i in range(len(x)):
        d = 10
        while d>0.000001:
            if (x[i]-1)*z[i]-np.exp(z[i])-0.5/sig*((z[i]-mu[i])*(z[i]-mu[i]))-mf[i]+np.log(1000000) > 0:
                z[i] = z[i] + d
            else:
                z[i] = z[i] - d
            d=d/2
    return z


@njit
def poilog_lower_single(x, m, mu, sig):
    mf = (x-1)*m-np.exp(m)-0.5/sig*((m-mu)*(m-mu))
    z = m-20
    d = 10
    while d>0.000001:
        if (x-1)*z-np.exp(z)-0.5/sig*((z-mu)*(z-mu))-mf+np.log(1000000) > 0:
            z=z-d
        else:
            z=z+d
        d=d/2
    return z


@njit
def poilog_lower(x, m, mu, sig):
    mf = (x-1)*m-np.exp(m)-0.5/sig*((m-mu)*(m-mu))
    z = m-20
    assert len(x) == len(mu)
    for i in range(len(x)):
        d = 10
        while d>0.000001:
            if (x[i]-1)*z[i]-np.exp(z[i])-0.5/sig*((z[i]-mu[i])*(z[i]-mu[i]))-mf[i]+np.log(1000000) > 0:
                z[i] = z[i] - d
            else:
                z[i] = z[i] + d
            d=d/2
    return z


def poilog_pmf(x, mu, sig):
    m = poilog_maxf(x, mu, sig)
    a = poilog_lower(x, m, mu, sig)
    b = poilog_upper(x, m, mu, sig)
    fac = loggamma(x+1)
    # integration
    result = np.zeros(len(x))
    for i in range(len(x)):
        def my_f_vec(z):
            return np.exp(z*x[i] - np.exp(z)- 0.5/sig * ((z-mu[i])*(z-mu[i])) - fac[i])
        result[i] = scipy.integrate.quad_vec(my_f_vec, a[i], b[i])[0]
    val = result*(1.0 / np.sqrt(2* np.pi * sig))
    return val


import pypoilog
class Weighted_PoiLog(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, seed=0, **kwds):
        super(Weighted_PoiLog, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
    #
    def nloglikeobs(self, params):
        # the following cutoff for sigma is copied from poilog R code
        if params[-1] < (-372):
            params[-1] = -372
        if params[-1] >    354:
            params[-1] =  354
        mu = self.exog @ params[:-1] + np.log(self.exposure)
        sig2 = params[-1]
        llf = np.log(pypoilog.poilog_pmf_vec3(self.endog.astype(int), mu, sig2))
        neg_sum_llf = -llf.dot(self.weights)
        # print(params, neg_sum_llf)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('alpha')
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                log_y = np.log(self.endog + 1e-4) - np.log(self.exposure + 1e-4)
                xtx = self.exog.T @ (self.weights.reshape(-1,1) * self.exog)
                tmp_start = np.linalg.inv(xtx) @ (self.weights.reshape(-1,1) * self.exog).T @ log_y
                start_params = np.append(tmp_start, 1)
                print(f"start params: {start_params}")
        return super(Weighted_PoiLog, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_PoiLog_fixedsig(GenericLikelihoodModel):
    def __init__(self, endog, exog, sig2, weights, exposure, seed=0, **kwds):
        super(Weighted_PoiLog_fixedsig, self).__init__(endog, exog, **kwds)
        self.sig2 = sig2
        # the following cutoff for sigma is copied from poilog R code
        if self.sig2[-1] < (-372):
            self.sig2[-1] = -372
        if self.sig2[-1] >    354:
            self.sig2[-1] =  354
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
    #
    def nloglikeobs(self, params):
        mu = self.exog @ params + np.log(self.exposure)
        llf = np.log(pypoilog.poilog_pmf_vec3(self.endog.astype(int), mu, self.sig2))
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('alpha')
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                log_y = np.log(self.endog + 1e-4) - np.log(self.exposure + 1e-4)
                xtx = self.exog.T @ (self.weights.reshape(-1,1) * self.exog)
                start_params = np.linalg.inv(xtx) @ (self.weights.reshape(-1,1) * self.exog).T @ log_y
        return super(Weighted_PoiLog_fixedsig, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
"""