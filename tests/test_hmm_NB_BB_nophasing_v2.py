import pytest
import numpy as np
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2


@pytest.fixture
def mock_data():
    (n_states, n_obs, n_spots) = (7, 4_248, 1)

    X = 10 * np.ones(shape=(n_obs, 2, n_spots))
    base_nb_mean = 3 * np.ones(shape=(n_obs, n_spots))
    log_mu = np.ones(shape=(n_states, n_spots))
    alphas = np.ones(shape=(n_states, n_spots))
    total_bb_RD = 10 * np.ones(shape=(n_obs, n_spots))
    p_binom = np.random.uniform(size=(n_states, n_spots))
    taus = np.ones(shape=(n_states, n_spots))
    tumor_prop = np.random.uniform(size=(n_obs, n_spots))

    return X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop

def test_compute_emission_probability_nb_betabinom_array_speed(benchmark, mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data

    def get_result():
        return hmm.compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus,
    )

    result_rdr, result_baf = benchmark(get_result)

def test_compute_emission_probability_nb_betabinom_speed(benchmark, mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data

    def get_exp():
        return hmm.compute_emission_probability_nb_betabinom_v1(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus,
    )

    exp_rdr, exp_baf = benchmark(get_exp)

def test_compute_emission_probability_nb_betabinom_mix_equality(mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data


    exp_rdr, exp_baf = hmm.compute_emission_probability_nb_betabinom_v1(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus,
    )

    result_rdr, result_baf = hmm.compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus,
    )

    np.testing.assert_allclose(exp_rdr, result_rdr)

    assert np.all(exp_baf == result_baf)
    
def test_compute_emission_probability_nb_betabinom_mix_array_speed(benchmark, mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data

    def get_result():
        return hmm.compute_emission_probability_nb_betabinom_mix_array(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
    )

    result_rdr, result_baf = benchmark(get_result)

def test_compute_emission_probability_nb_betabinom_mix_speed(benchmark, mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data

    def get_exp():
        return hmm.compute_emission_probability_nb_betabinom_mix(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
    )
    
    exp_rdr, exp_baf = benchmark(get_exp)

def test_compute_emission_probability_nb_betabinom_mix_equality(mock_data):
    hmm = hmm_nophasing_v2()
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop = mock_data

    
    exp_rdr, exp_baf = hmm.compute_emission_probability_nb_betabinom_mix(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
    )
    
    result_rdr, result_baf = hmm.compute_emission_probability_nb_betabinom_mix_array(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
    )

    np.testing.assert_allclose(exp_rdr, result_rdr)
    
    assert np.all(exp_baf == result_baf)
