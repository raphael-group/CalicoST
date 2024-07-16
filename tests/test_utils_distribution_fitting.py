import calicostem
import line_profiler
import numpy as np
import scipy
import pytest
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import (
    hmrfmix_reassignment_posterior_concatenate_emission,
    hmrfmix_reassignment_posterior_concatenate_emission_v1,
)
from calicost.utils_distribution_fitting import Weighted_NegativeBinomial, Weighted_BetaBinom
from scipy.sparse import csr_matrix
from scipy.stats import betabinom
from sklearn.preprocessing import OneHotEncoder


def test_Weighted_BetaBinom(benchmark):
    np.random.seed(314)
    
    nclass, len_exog = 5, 1_000
    
    encoder = OneHotEncoder()

    aa = np.random.randint(low=0, high=1_000, size=5)
    bb = np.random.randint(low=0, high=1_000, size=5)

    state = np.random.randint(low=0, high=nclass, size=len_exog)
    exog = encoder.fit_transform(state.reshape(-1, 1)).toarray()

    exposure = np.random.randint(low=0, high=25, size=len_exog)    
    endog = np.array([scipy.stats.betabinom.rvs(xp, aa[ss], bb[ss]) for ss, xp in zip(state, exposure)])

    weights = np.random.uniform(size=len_exog)
    
    bb = Weighted_BetaBinom(endog, exog, weights, exposure)

    def call():
        return bb.fit()

    result = benchmark(call)
    
    print(result.params)
