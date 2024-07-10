import pytest
import numpy as np
from scipy.sparse import csr_matrix
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2

def get_spatial_data():
    np.random.seed(314)
    
    # TODO HACK
    root = "/Users/mw9568/runs/CalicoSTdata/HT225C1_joint"
    
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
    """
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
    """
    tmp_log_emission_rdr_array, tmp_log_emission_baf_array = (
        hmm.compute_emission_probability_nb_betabinom_mix(
            single_X,
            single_base_nb_mean,
            new_log_mu,
            new_alphas,
            single_total_bb_RD,
            new_p_binom,
            new_taus,
            single_tumor_prop,
            **kwargs,
        )
    )

    np.save("tmp_log_emission_rdr2.npy", tmp_log_emission_rdr_array)
    np.save("tmp_log_emission_baf2.npy", tmp_log_emission_baf_array)
    
    array_tmp_log_emission_rdr = np.load("tmp_log_emission_rdr2.npy")
    array_tmp_log_emission_baf = np.load("tmp_log_emission_baf2.npy")
    """
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
    """
    print("Done.")
