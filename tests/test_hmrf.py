import pytest
import numpy as np
from scipy.sparse import csr_matrix
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import solve_edges


@pytest.fixture
def mock_data():
    ii = 4
    n_clones = 5

    adjacency_mat = np.random.randint(0, 10, size=(10, 10))
    new_assignment = np.random.randint(0, n_clones, size=10)

    return ii, n_clones, adjacency_mat, new_assignment


@pytest.fixture
def mock_spatial_data():
    (n_spots, n_obs, n_clones, n_states) = (13_344, 2_282, 5, 7)

    X = 10 * np.ones(shape=(n_obs, 2, n_spots))
    base_nb_mean = 3 * np.ones(shape=(n_obs, n_spots))
    log_mu = np.ones(shape=(n_states, n_spots))
    alphas = np.ones(shape=(n_states, n_spots))
    total_bb_RD = 10 * np.ones(shape=(n_obs, n_spots))
    p_binom = np.random.uniform(size=(n_states, n_spots))
    taus = np.ones(shape=(n_states, n_spots))
    tumor_prop = np.random.uniform(size=(n_obs, n_spots))

    return (
        (n_spots, n_obs, n_clones, n_states),
        X,
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        tumor_prop,
    )


def get_spatial_data():
    # TODO HACK
    root = "/Users/mw9568/runs/CalicoSTdata/HT225C1_joint"

    np.random.seed(314)

    kwargs = np.load(f"{root}/kwargs.npz")
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
    
    kwargs_dict = {}
    kwargs_dict["logmu_shift"] = kwargs["logmu_shift"]
    kwargs_dict["sample_length"] = kwargs["sample_length"]

    kwargs_dict["logmu_shift"] = np.tile(kwargs_dict["logmu_shift"], (1, N))
            
    new_log_mu = 20.0 + 5 * np.random.uniform(size=N)
    new_log_mu = np.tile(new_log_mu, (n_states, 1))

    new_alphas = np.ones_like(new_log_mu)

    # TODO HACK
    new_p_binom = np.random.uniform(size=N)
    new_p_binom = np.tile(new_p_binom, (n_states, 1))

    new_taus = np.ones_like(new_p_binom)

    hmm = hmm_nophasing_v2()

    return (
        kwargs_dict,
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


def test_spatial_data(spatial_data):
    # spatial_data = get_spatial_data()

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

    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = 5
    n_states = 7

    posterior = np.zeros((N, n_clones), dtype=float)

    # NB removes 27 entries from smooth_mat, from 3 bad spots.
    smooth_mat = smooth_mat.tolil()
    
    bad_tumor_prop = np.isnan(single_tumor_prop).astype(int)
    bad_tumor_prop_idx = np.argwhere(bad_tumor_prop).squeeze()
    
    for idx in bad_tumor_prop_idx:
        smooth_mat[idx, :] = 0

        # DEPRECATE
        # smooth_mat[:, idx] = 0

    smooth_mat = smooth_mat.tocsr()

    # DEPRECATE
    # print(smooth_mat.size)

    smooth_baseline = single_base_nb_mean @ smooth_mat
    smooth_rd = single_total_bb_RD @ smooth_mat

    # NB smooth_RD == (2407, 17792)
    smooth_xrd = single_X[:, 0, :] @ smooth_mat
    smooth_xbaf = single_X[:, 1, :] @ smooth_mat

    # NB smooth_X == (2407, 2, 17792)
    smooth_X = np.stack([smooth_xrd, smooth_xbaf], axis=1)

    # NB single_tumor_prop == (17792, 1)
    norm = np.sum(smooth_mat, axis=0)

    smooth_tumor_prop = np.expand_dims(single_tumor_prop @ smooth_mat, -1).T
    smooth_tumor_prop /= norm

    smooth_tumor_prop = np.tile(smooth_tumor_prop, (n_obs, 1))
    
    tmp_log_emission_rdr_array, tmp_log_emission_baf_array = (
        hmm.compute_emission_probability_nb_betabinom_mix(
            smooth_X,
            smooth_baseline,
            new_log_mu,
            new_alphas,
            smooth_rd,
            new_p_binom,
            new_taus,
            smooth_tumor_prop,
            **kwargs,
        )
    )

    np.save("tmp_log_emission_rdr.npy", tmp_log_emission_rdr_array)
    np.save("tmp_log_emission_baf.npy", tmp_log_emission_baf_array)
    
    array_tmp_log_emission_rdr = np.load("tmp_log_emission_rdr.npy")
    array_tmp_log_emission_baf = np.load("tmp_log_emission_baf.npy")
    
    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]

        agg_adj_single_X = np.sum(single_X[:, :, idx], axis=2, keepdims=True)
        agg_adj_base_nb = np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True)
        agg_adj_bb_RD = np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True)
        idx_mean_single_tumor_prop = np.ones((n_obs, 1)) * np.mean(
            single_tumor_prop[idx]
        )

        assert np.allclose(smooth_X[:, :, i], agg_adj_single_X[:, :, 0])
        assert np.allclose(smooth_baseline[:, i], agg_adj_base_nb[:, 0])
        assert np.allclose(smooth_rd[:, i], agg_adj_bb_RD[:, 0])
        assert np.allclose(smooth_tumor_prop[:, i], idx_mean_single_tumor_prop)
        
        spot_kwargs = kwargs.copy()
        spot_kwargs["logmu_shift"] = np.expand_dims(kwargs["logmu_shift"][:, i], -1)
        
        tmp_log_emission_rdr, tmp_log_emission_baf = (
            hmm.compute_emission_probability_nb_betabinom_mix(
                agg_adj_single_X,
                agg_adj_base_nb,
                np.expand_dims(new_log_mu[:, i], -1),
                np.expand_dims(new_alphas[:, i], -1),
                agg_adj_bb_RD,
                np.expand_dims(new_p_binom[:, i], -1),
                np.expand_dims(new_taus[:, i], -1),
                idx_mean_single_tumor_prop,
                **spot_kwargs,
            )
        )

        assert np.allclose(array_tmp_log_emission_rdr[:,:,i], tmp_log_emission_rdr[:,:,0])
        assert np.allclose(array_tmp_log_emission_baf[:,:,i], tmp_log_emission_baf[:,:,0])

        # print()
        # print(array_tmp_log_emission_baf[0, :, i])
        # print(tmp_log_emission_baf[0, :, 0])

        if i == 1000:
            break

    print("Done.")


def test_spatial_data_new_benchmark(benchmark):
    benchmark(test_spatial_data_new)


def test_hmrfmix_reassignment_posterior_concatenate(mock_spatial_data):
    (
        (n_spots, n_obs, n_clones, n_states),
        X,
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        tumor_prop,
    ) = mock_spatial_data

    single_base_nb_mean_sum = np.sum(base_nb_mean)

    lambd = np.sum(base_nb_mean, axis=1) / single_base_nb_mean_sum
    log_lambd = np.log(lambd).reshape(-1, 1)

    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])

    data = np.array([1, 2, 3, 4, 5, 6])

    # shape=(n_spots, n_spots)
    shape = (10, 10)

    smooth = csr_matrix((data, (row, col)), shape=shape)

    # NB construct row and col from smooth.indptr and smooth.indices
    row = np.repeat(np.arange(smooth.shape[0]), np.diff(smooth.indptr))
    col = smooth.indices

    # TODO filter for zeros in smooth mat?

    # isin = [~np.isnan(tumor_prop[i, j]) for i, j in zip(row, col)]
    isin = [tumor_prop[i, j] < 0.5 for i, j in zip(row, col)]

    print(np.mean(isin))

    # NB ones where
    agg_spots = csr_matrix(
        (np.ones_like(smooth.data)[isin], (row[isin], col[isin])), shape=shape
    )

    # broadcast the last direction of X along a new axis and sum along this new axis.

    # result = X[:,:,:,None] * np.ones(n_spots)

    # np.sum(single_X[:,:,idx], axis=2, keepdims=True)

    # print(f"\n{result.shape}")


def test_edges_old(benchmark, mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data

    def get_exp():
        return solve_edges(
            ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=False
        )

    exp = benchmark(get_exp)


def test_edges_new(benchmark, mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data

    def get_result():
        return solve_edges(
            ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=True
        )

    result = benchmark(get_result)


def test_edges_equality(mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data

    exp = solve_edges(
        ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=False
    )
    result = solve_edges(
        ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=True
    )

    assert np.all(exp == result)
