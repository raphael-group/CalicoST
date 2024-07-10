from numba import njit
import numpy as np

def get_tumor_weight(sample_lengths, tumor_prop, log_mu, logmu_shift):
    n_obs, n_spots = tumor_prop.shape
    n_state = log_mu.shape[0]
        
    weighted_tumor_prop = []
    
    for c in range(len(sample_lengths)):
        range_s = np.sum(sample_lengths[:c])
        range_t = np.sum(sample_lengths[:(c+1)])

        range_tumor_prop = tumor_prop[range_s:range_t, :]
        shifted_mu = np.exp(log_mu - logmu_shift[c,:])

        result = range_tumor_prop * shifted_mu[:, None, :]
        result /= (range_tumor_prop * shifted_mu[:, None, :] + 1. - range_tumor_prop)

        weighted_tumor_prop.append(result)
        
    return np.concatenate(weighted_tumor_prop, axis=1)
    
    
if __name__ == "__main__":
    n_state = 4
    n_spots = 10
    
    sample_lengths = np.array([14, 15, 16])
    n_obs = sample_lengths.sum()

    print(f"(n_state, n_obs, n_spots) = ({n_state}, {n_obs}, {n_spots})")
    
    tumor_prop = np.random.uniform(size=(n_obs, n_spots))
    log_mu = np.random.uniform(size=(n_state, n_spots))

    logmu_shift = np.random.uniform(size=(len(sample_lengths), n_spots))

    result = get_tumor_weight(sample_lengths, tumor_prop, log_mu, logmu_shift)

    assert result.shape == (n_state, n_obs, n_spots)
