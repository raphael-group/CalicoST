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
    
    tau = 200.
    
    ps = 0.5 + 0.2 * np.random.uniform(size=nclass)

    aa = tau * ps
    bb = tau * (1. - ps)
    
    state = np.random.randint(low=0, high=nclass, size=len_exog)
    exog = OneHotEncoder().fit_transform(state.reshape(-1, 1)).toarray()

    exposure = np.random.randint(low=10, high=25, size=len_exog)   
    endog = np.array([scipy.stats.betabinom.rvs(xp, aa[ss], bb[ss]) for ss, xp in zip(state, exposure)])

    weights = 0.5 + 0.1 * np.random.uniform(size=len_exog)
    
    bb = Weighted_BetaBinom(endog, exog, weights, exposure)

    params = np.concatenate([aa, np.array([tau])])
    
    def call():
        return bb.nloglikeobs(params)

    result = call()
    # result = benchmark(call)

    print(result)
    
    # NB regression testing
    # exp = 327.28956683765364
    
    # assert np.allclose(result.params.sum(), exp)
    
