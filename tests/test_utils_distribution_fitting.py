import calicostem
import line_profiler
import numpy as np
import pytest
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import (
    hmrfmix_reassignment_posterior_concatenate_emission,
    hmrfmix_reassignment_posterior_concatenate_emission_v1,
)
from scipy.sparse import csr_matrix

def test_
