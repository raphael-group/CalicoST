import core
import pytest
import line_profiler
import numpy as np
from scipy.sparse import csr_matrix
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import hmrfmix_reassignment_posterior_concatenate_emission_v1
from calicost.hmrf import hmrfmix_reassignment_posterior_concatenate_emission_v2
from calicost.utils_tumor import get_tumor_weight

ITERATIONS = ROUNDS = 1


def get_raw_spatial_data():
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

    kwargs = {}
    kwargs["logmu_shift"] = inkwargs["logmu_shift"]
    kwargs["sample_length"] = inkwargs["sample_length"]

    return (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    )


@pytest.mark.skip(reason="This test is currently not needed")
def test_get_raw_spatial_data():
    (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    ) = get_raw_spatial_data()

    logmu_shift = kwargs["logmu_shift"]
    sample_length = kwargs["sample_length"]

    n_obs, n_comp, n_spots = single_X.shape
    n_clones = len(kwargs["sample_length"])

    assert single_base_nb_mean.shape == (n_obs, n_spots)
    assert single_tumor_prop.shape == (n_spots,)
    assert single_X.shape == (n_obs, n_comp, n_spots)
    assert single_total_bb_RD.shape == (n_obs, n_spots)
    assert smooth_mat.shape == (n_spots, n_spots)
    assert sample_length.shape == (n_clones,)
    assert np.all(sample_length == n_obs)

    # TODO HACK last 1?
    assert logmu_shift.shape == (n_clones, 1)

    # NB expect (will fail):
    assert logmu_shift.shape == (n_clones, n_spots)


def get_spatial_data():
    np.random.seed(314)

    # TODO HACK
    # see https://github.com/raphael-group/CalicoST/blob/4696325d5ca103d0d72ea2d471c60d1d753b097b/src/calicost/hmrf.py#L765
    n_states = 3

    (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    ) = get_raw_spatial_data()

    # NB usually n_spots, or one spot / clone.
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = len(kwargs["sample_length"])

    # TODO
    new_log_mu = np.log(2.0 + 2.0 * np.random.uniform(size=N))
    new_log_mu = np.tile(new_log_mu, (n_states, 1))

    new_alphas = 0.01 * np.ones_like(new_log_mu, dtype=float)

    new_p_binom = np.random.uniform(size=N)
    new_p_binom = np.tile(new_p_binom, (n_states, 1))

    new_taus = np.ones_like(new_p_binom)

    hmm = hmm_nophasing_v2()

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
    )


@pytest.fixture
def spatial_data():
    return get_spatial_data()


@pytest.mark.skip(reason="This test is currently not needed")
def test_get_spatial_data(spatial_data):
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
    ) = spatial_data

    # NB usually n_spots, or one spot / clone.
    n_spots = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_states = new_log_mu.shape[0]
    n_clones = len(kwargs["sample_length"])

    assert new_log_mu.shape == (n_states, n_spots)
    assert new_log_mu.shape == new_alphas.shape
    assert new_p_binom.shape == new_p_binom.shape
    assert new_taus == new_taus.shape


@pytest.mark.skip(reason="This test is currently not needed")
def test_hmrfmix_reassignment_posterior_concatenate_emission_v1(
    benchmark, spatial_data
):
    """
    pytest -s test_hmrf.py::test_hmrfmix_reassignment_posterior_concatenate_emission_v1

    Tests the original loop version of the HMRF emission calc.  Calls the underlying
    hmm.emission calc.
    """
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
    ) = spatial_data

    def benchmark_v1():
        # See emacs +764 ../src/calicost/hmrf.py
        #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py
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

    benchmark.pedantic(benchmark_v1, iterations=ITERATIONS, rounds=ROUNDS)


@pytest.mark.skip(reason="This test is currently not needed")
def test_hmrfmix_reassignment_posterior_concatenate_emission_v2(
    benchmark, spatial_data
):
    """
    pytest -s test_hmrf.py::test_hmrfmix_reassignment_posterior_concatenate_emission_v2

    Tests the new loop version of the HMRF emission calc.  Calls the underlying
    hmm.emission calc.
    """
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
    ) = spatial_data

    def benchmark_v2():
        # See emacs +812 ../src/calicost/hmrf.py
        #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py
        return hmrfmix_reassignment_posterior_concatenate_emission_v2(
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

    tmp_log_emission_rdr, tmp_log_emission_baf = benchmark.pedantic(
        benchmark_v2, iterations=ITERATIONS, rounds=ROUNDS
    )

    # See emacs +764 ../src/calicost/hmrf.py                                                                                                                                                                     
    #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py                                                                                                                                                   
    exp = hmrfmix_reassignment_posterior_concatenate_emission_v1(
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
    
    assert np.allclose(tmp_log_emission_rdr, exp[0])
    assert np.allclose(tmp_log_emission_baf, exp[1])


def test_compute_emission_probability_nb_mix_exp(benchmark, spatial_data):
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
    ) = spatial_data

    n_obs, _, n_spots = single_X.shape

    def get_exp():
        return hmm.compute_emission_probability_nb_mix(
            single_X,
            single_base_nb_mean,
            new_log_mu,
            new_alphas,
            single_total_bb_RD,
            new_p_binom,
            new_taus,
            np.tile(single_tumor_prop, (n_obs, 1)),
        )

    log_emission_rdr = benchmark(get_exp)


def test_compute_emission_probability_bb_mix_exp(benchmark, spatial_data):
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
    ) = spatial_data

    n_obs, _, n_spots = single_X.shape

    sample_lengths = kwargs["sample_length"]
    logmu_shift = kwargs["logmu_shift"]

    single_tumor_prop = np.tile(single_tumor_prop, (n_obs, 1))

    # TODO HACK ask Cong.
    logmu_shift = np.tile(logmu_shift, (1, n_spots))
    tumor_weight = get_tumor_weight(sample_lengths, single_tumor_prop, new_log_mu, logmu_shift)

    # tumor_weight=tumor_weight
    def get_exp():
        return hmm.compute_emission_probability_bb_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            new_p_binom,
            new_taus,
            single_tumor_prop,
            tumor_weight = tumor_weight
        )

    log_emission_baf = benchmark(get_exp)


def test_compute_emission_probability_bb_mix(benchmark, spatial_data):
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
    ) = spatial_data

    n_obs, _, n_spots = single_X.shape

    sample_lengths = kwargs["sample_length"]
    logmu_shift = kwargs["logmu_shift"]

    single_tumor_prop = np.tile(single_tumor_prop, (n_obs, 1))

    # TODO HACK ask Cong.
    logmu_shift = np.tile(logmu_shift, (1, n_spots))
    # tumor_weight = get_tumor_weight(sample_lengths, single_tumor_prop, new_log_mu, logmu_shift)

    def get_result():
        return core.compute_emission_probability_bb_mix(
            single_X[:, 1, :],
            single_base_nb_mean,
            single_total_bb_RD.astype(float),
            new_p_binom,
            new_taus,
            single_tumor_prop,
        )

    # TODO tumor_weight=tumor_weight
    exp = hmm.compute_emission_probability_bb_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        new_p_binom,
        new_taus,
        single_tumor_prop,
    )

    log_emission_baf = benchmark(get_result)

    good = np.isclose(log_emission_baf, exp, atol=1.0e-6, equal_nan=True)
    mean = np.mean(good)

    print()
    print(mean)
    print(np.nanmin(log_emission_baf), log_emission_baf[0, 0, :])
    print(np.nanmin(exp), exp[0, 0, :])
    
    # NB TODO Rust NaNs matched to 0.0s
    assert mean >= 0.9998
    

def test_compute_emission_probability_nb_mix(benchmark, spatial_data):
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
    ) = spatial_data

    n_obs, _, n_spots = single_X.shape

    def benchmark_v3():
        return core.compute_emission_probability_nb_mix(
            single_X[:, 0, :],
            single_base_nb_mean,
            np.tile(single_tumor_prop, (n_obs, 1)),
            new_log_mu,
            new_alphas,
        )

    exp = hmm.compute_emission_probability_nb_mix(
        single_X,
        single_base_nb_mean,
        new_log_mu,
        new_alphas,
        single_total_bb_RD,
        new_p_binom,
        new_taus,
        np.tile(single_tumor_prop, (n_obs, 1)),
    )

    """
    log_emission_rdr = benchmark.pedantic(
        benchmark_v3, iterations=ITERATIONS, rounds=ROUNDS
    )
    """

    log_emission_rdr = benchmark(benchmark_v3)

    good = np.isclose(log_emission_rdr, exp, atol=1.0e-6, equal_nan=True)
    mean = np.mean(good)

    # NB TODO Rust NaNs matched to 0.0s
    assert mean >= 0.99997


@line_profiler.profile
def profile(iterations=ITERATIONS):
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

    for _ in range(iterations):
        tmp_log_emission_rdr, tmp_log_emission_baf = (
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


if __name__ == "__main__":
    profile(iterations=1)
