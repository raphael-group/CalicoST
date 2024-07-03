import scipy
import numpy as np

from scipy.stats import nbinom, betabinom
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 1


def compute_nbinom_pmf_chunk(args):
    data_chunk, n, p = args
    return nbinom.logpmf(data_chunk, n, p)


def compute_betabinom_pmf_chunk(args):
    data_chunk, n, a, b = args
    return betabinom.logpmf(data_chunk, n, a, b)


def compute_logsumexp_chunk(data_chunk):
    return scipy.special.logsumexp(data_chunk)


def thread_nbinom(data, n, p, num_threads=NUM_THREADS):
    if num_threads == 1:
        return nbinom.logpmf(data, n, p)

    # NB defaults to 0th axis, see
    #    https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    data_chunks = np.array_split(data, num_threads)
    n_chunks = np.array_split(n, num_threads)
    p_chunks = np.array_split(p, num_threads)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        args = [xx for xx in zip(data_chunks, n_chunks, p_chunks)]
        results = executor.map(compute_nbinom_pmf_chunk, args)

        # NB defaults to 0th axis, see
        #    https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        pmf_values = np.concatenate(list(results))

    return pmf_values


def thread_betabinom(data, n, a, b, num_threads=NUM_THREADS):
    if num_threads == 1:
        return betabinom.logpmf(data, n, a, b)

    data_chunks = np.array_split(data, num_threads)
    n_chunks = np.array_split(n, num_threads)
    a_chunks = np.array_split(a, num_threads)
    b_chunks = np.array_split(b, num_threads)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        args = [xx for xx in zip(data_chunks, n_chunks, a_chunks, b_chunks)]
        results = executor.map(compute_betabinom_pmf_chunk, args)
        pmf_values = np.concatenate(list(results))

    return pmf_values


def thread_logsumexp(data, num_threads=NUM_THREADS):
    if num_threads == 1:
        return scipy.special.logsumexp(data)

    data_chunks = np.array_split(data, num_threads)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        args = list(data_chunks)

        results = executor.map(compute_logsumexp_chunk, args)
        results = np.array(list(results))

    return scipy.special.logsumexp(results)
