import pytest
import numpy as np
from scipy.sparse import csr_matrix
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import hmrfmix_reassignment_posterior_concatenate_emission_v1
from calicost.hmrf import hmrfmix_reassignment_posterior_concatenate_emission_v2


def get_spatial_data():
    np.random.seed(314)
    
    # TODO HACK
    root = "/Users/mw9568/runs/CalicoSTdata/HT225C1_joint"
    
    inkwargs = np.load(f"{root}/kwargs.npz")
    res = np.load(f"{root}/res.npz")
    single_base_nb_mean = np.load(f"{root}/single_base_nb_mean.npy")
    single_tumor_prop = np.load(f"{root}/single_tumor_prop.npy")
    single_X = np.load(f"{root}/single_X.npy")
    single_total_bb_RD = np.load(f"{root}/single_total_bb_RD.npy")
    smooth_mat = np.load(f"{root}/smooth_mat.npz")

    smooth_mat = csr_matrix(
        (smooth_mat["data"], smooth_mat["indices"], smooth_mat["indptr"]),
        shape=smooth_mat["shape"],
    )

    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = 5
    n_states = 7
    
    kwargs = {}
    kwargs["logmu_shift"] = inkwargs["logmu_shift"]
    kwargs["sample_length"] = inkwargs["sample_length"]

    kwargs["logmu_shift"] = np.tile(inkwargs["logmu_shift"], (1, N))
            
    new_log_mu = 20.0 + 5 * np.random.uniform(size=N)
    new_log_mu = np.tile(new_log_mu, (n_states, 1))

    new_alphas = np.ones_like(new_log_mu)

    # TODO HACK
    new_p_binom = np.random.uniform(size=N)
    new_p_binom = np.tile(new_p_binom, (n_states, 1))

    new_taus = np.ones_like(new_p_binom)

    hmm = hmm_nophasing_v2()

    exp = (
        hmrfmix_reassignment_posterior_concatenate_emission_v1(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            smooth_mat,
            hmm,
            kwargs["logmu_shift"],
            kwargs["sample_length"],
        )
    )

    return (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        exp,
    )


@pytest.fixture
def spatial_data():
    return get_spatial_data()

@pytest.mark.skip(reason="This test is not currently needed")
def test_get_spatial_data():
    (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        exp,
    ) = get_spatial_data()

    # TODO
    # print(kwargs["sample_length"].shape)
    # print(kwargs["logmu_shift"].shape)
    
@pytest.mark.skip(reason="REMOVE SKIP")
def test_spatial_data_v1(benchmark, spatial_data):
    (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        exp,
    ) = spatial_data

    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = 5
    n_states = 7

    def benchmark_v1():
        hmrfmix_reassignment_posterior_concatenate_emission_v1(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            smooth_mat,
            hmm,
            kwargs["logmu_shift"],
            kwargs["sample_length"],
            dry_run=True,
        )

    benchmark.pedantic(benchmark_v1, iterations=1, rounds=1)
    
def test_spatial_data_v2(benchmark, spatial_data):
    (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        exp,
    ) = spatial_data
    """
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = 5
    n_states = 7

    def benchmark_v2():    
        tmp_log_emission_rdr2, tmp_log_emission_baf2 = (
            hmrfmix_reassignment_posterior_concatenate_emission_v2(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                single_tumor_prop,
                new_log_mu,
                new_alphas,
                new_p_binom,
                new_taus,
                smooth_mat,
                hmm,
                kwargs["logmu_shift"],
                kwargs["sample_length"],
            )
        )

        return tmp_log_emission_rdr2, tmp_log_emission_baf2

    # tmp_log_emission_rdr, tmp_log_emission_baf = benchmark_v2() 
    tmp_log_emission_rdr, tmp_log_emission_baf = benchmark.pedantic(benchmark_v2, iterations=1, rounds=1)
    
    assert np.allclose(tmp_log_emission_rdr, exp[0])
    assert np.allclose(tmp_log_emission_baf, exp[1])
    """
