import numpy as np


def get_tumor_weight(sample_lengths, tumor_prop, log_mu, logmu_shift):
    n_obs, n_spots = tumor_prop.shape
    n_state = log_mu.shape[0]

    weighted_tumor_prop = []

    for c in range(len(sample_lengths)):
        range_s = np.sum(sample_lengths[:c])
        range_t = np.sum(sample_lengths[: c + 1])

        range_tumor_prop = tumor_prop[range_s:range_t, :]
        shifted_mu = np.exp(log_mu - logmu_shift[c, :])[:, None, :]
        result = (
            range_tumor_prop
            * shifted_mu
            / (range_tumor_prop * shifted_mu + 1.0 - range_tumor_prop)
        )

        weighted_tumor_prop.append(result)

    return np.concatenate(weighted_tumor_prop, axis=1)
