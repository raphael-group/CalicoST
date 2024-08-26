import contextlib
import functools
import inspect
import logging
import os
import sys
import time

import numpy as np
import scipy
import scipy.integrate
import scipy.stats
import statsmodels
import statsmodels.api as sm
from numba import jit, njit
from scipy import linalg, special
from scipy.special import loggamma, logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state
from statsmodels.base.model import GenericLikelihoodModel
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

num_threads = "2"

logger.info(f"Setting number of threads for MKL/BLAS/LAPACK/OMP to {num_threads}.")

os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["OMP_NUM_THREADS"] = num_threads

def convert_params(mean, std):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = mean / std**2
    n = mean * p / (1.0 - p)

    return n, p


@contextlib.contextmanager
def save_stdout(fpath):
    original = sys.stdout
    
    with open(fpath, "w") as ff:
        sys.stdout = ff
        try:
            yield

        # NB teardown
        finally:
            sys.stdout = original

class WeightedModel(GenericLikelihoodModel, ABC):
    """
    An ABC for defined emission models.

    Attributes                                                                                                                                                                                                                                                                                     
    ----------                                                                                                                                                                                                                                                                                     
    endog : array, (n_samples,)                                                                                                                                                                                                                                                                    
        Y values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    exog : array, (n_samples, n_features)                                                                                                                                                                                                                                                          
        Design matrix.                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                   
    weights : array, (n_samples,)                                                                                                                                                                                                                                                                  
        Sample weights.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    exposure : array, (n_samples,)                                                                                                                                                                                                                                                                
        Multiplication constant outside the exponential term. In scRNA-seq or SRT data, this term is the total UMI count per cell/spot.
    """
    def __init__(self, endog, exog, weights, exposure, tumor_prop=None, seed=0, **kwds):
        super(WeightedModel, self).__init__(endog, exog, **kwds)
        
        self.weights = weights
        self.exposure = exposure

        # NB Weight_BetaBinomial does not specify seed
        self.seed = seed
        self.tumor_prop = tumor_prop

        self.__post_init__()
                
        logger.info(f"Initializing {self.__class__.__name__} model for endog.shape = {endog.shape}.")
        
    @abstractmethod
    def nloglikeobs(self, params):
        pass

    @abstractmethod
    def get_default_start_params(self):
        pass

    @abstractmethod
    def get_ext_param_name(self):
        pass

    @classmethod
    @abstractmethod
    def get_ninstance(cls):
        pass
    
    @abstractmethod
    def __post_init__(self):
        # NB will increment the instance count for each derived class.
        pass
    
    def callback(self, params):
        nloglike = self.nloglikeobs(params)

        print(f"{params} {nloglike};")
    
    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwds):
        ext_param_name = self.get_ext_param_name()

        self.exog_names.append(ext_param_name)
        
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
                start_params_str = "existing"
            else:
                start_params = self.default_start_params()
                start_params_str = "default"
        else:
            start_params_str = "input"

        logger.info(f"Starting {self.__class__.__name__} optimization @ ({start_params_str}) {start_params}.")
        
        start = time.time()

        result = super(Weighted_Model, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            skip_hessian=True,
            callback=self.callback,
            full_output=True,
            retall=True,
            disp=False,
            **kwds
        )

        # NB specific to nm (Nelder-Mead) optimization.                                                                                                                                                                                                                                            
        niter = result.mle_retvals["iterations"]

        logger.info(f"Finished {self.__class__.__name__} optimization in {time.time() - start:.2f} seconds, with {niter} iterations.")

        return result

class Weighted_NegativeBinomial(WeightedModel):
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]), where exog is the design matrix, and params[-1] is 1 / overdispersion.
    This function fits the NB params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)
    """
    ninstance = 0
    
    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)
        
        n, p = convert_params(nb_mean, nb_std)

        return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)

    def get_default_start_params(self):
        return np.append(0.1 * np.ones(self.exog.shape[1]), 0.01)

    def get_ext_param_name():
        return "alpha"

    def __post_init__(self):
        Weighted_NegativeBinomial.ninstance += 1
        
    @classmethod
    def get_ninstance(cls):
        return cls.ninstance
        
class Weighted_NegativeBinomial_mix(WeightedModel):
    ninstance = 0
    
    def nloglikeobs(self, params):
        nb_mean = self.exposure * (
            self.tumor_prop * np.exp(self.exog @ params[:-1]) + 1 - self.tumor_prop
        )
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)

        n, p = convert_params(nb_mean, nb_std)

        return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)
    
    def get_default_start_params(self):
        return np.append(0.1 * np.ones(self.nparams), 0.01)

    def get_ext_param_name(self):
        return "alpha"

    def __post_init__(self):
        assert self.tumor_prop is not None, "Tumor proportion must be defined."
        
class Weighted_BetaBinom(WeightedModel):
    """
    Beta-binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)
    """
    ninstance = 0
    
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1. - self.exog @ params[:-1]) * params[-1]

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(self.weights)

    def get_default_start_params(self):
        return np.append(
            0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
        )

    def get_ext_param_name():
        return "tau"
    
class Weighted_BetaBinom_mix(WeightedModel_mix):
    def nloglikeobs(self, params):
        a = (
            self.exog @ params[:-1] * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]

        b = (
            (1 - self.exog @ params[:-1]) * self.tumor_prop
            + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(self.weights)

    def get_default_start_params(self):
        return np.append(
            0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1
        )
    
    def get_ext_param_name():
        return "tau"

    def	__post_init__(self):
        assert self.tumor_prop is not None, "Tumor proportion must be defined."
    
class Weighted_BetaBinom_fixdispersion(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, **kwds):
        super(Weighted_BetaBinom_fixdispersion, self).__init__(endog, exog, **kwds)

        logger.info(f"Initializing Weighted_BetaBinom_fixdispersion model for endog.shape = {endog.shape}.")

        self.tau = tau
        self.weights = weights
        self.exposure = exposure

    def nloglikeobs(self, params):
        a = (self.exog @ params) * self.tau
        b = (1 - self.exog @ params) * self.tau

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(self.weights)

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwds):
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
                start_params_str = "existing"
            else:
                start_params = 0.1 * np.ones(self.nparams)
                start_params_str = "default"
        else:
            start_params_str = "input"
                
        logger.info(f"Starting Weighted_BetaBinom_fixdispersion optimization @ ({start_params_str}) {start_params}.")

        start = time.time()

        result = super(Weighted_BetaBinom_fixdispersion, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            skip_hessian=True,
            callback=None,
            full_output=True,
            retall=False,
            **kwds
        )

        # NB specific to nm (Nelder-Mead) optimization.
        niter = result.mle_retvals["iterations"]

        logger.info(f"Finished Weighted_BetaBinom_fixdispersion optimization in {time.time() - start:.2f} seconds, with {niter} iterations.")

        return result


class Weighted_BetaBinom_fixdispersion_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_fixdispersion_mix, self).__init__(endog, exog, **kwds)

        logger.info(f"Initializing Weighted_BetaBinom_fixdispersion_mix model for endog.shape = {endog.shape}.")

        self.tau = tau
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop

    def nloglikeobs(self, params):
        a = (
            self.exog @ params * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * self.tau

        b = (
            (1 - self.exog @ params) * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * self.tau

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(self.weights)

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwds):
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
                start_params_str = "existing"
            else:
                start_params = 0.1 * np.ones(self.nparams)
                start_params_str = "default"
        else:
            start_params_str = "input"
                
        logger.info(f"Starting Weighted_BetaBinom_fixdispersion_mix optimization @ ({start_params_str}) {start_params}.")

        start = time.time()

        result = super(Weighted_BetaBinom_fixdispersion_mix, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            skip_hessian=True,
            callback=None,
            full_output=True,
            retall=False,
            **kwds
        )

        # NB specific to nm (Nelder-Mead) optimization.
        niter = result.mle_retvals["iterations"]

        logger.info(f"Finished Weighted_BetaBinom_fixdispersion_mix optimization in {time.time() - start:.2f} seconds, with {niter} iterations.")

        return result
