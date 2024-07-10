import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy

NUM_THREADS = 6

os.environ["RAYON_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)

def compute_nbinom_pmf_chunk(args):
    data_chunk, n, p = args
    return scipy.stats.nbinom.logpmf(data_chunk, n, p)


def compute_betabinom_pmf_chunk(args):
    data_chunk, n, a, b = args
    return scipy.stats.betabinom.logpmf(data_chunk, n, a, b)


def thread_nbinom(k, n, p, num_threads=NUM_THREADS, executor=None):
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=NUM_THREADS)

    k_chunks = np.array_split(k, num_threads)
    n_chunks = np.array_split(n, num_threads)
    p_chunks = np.array_split(p, num_threads)

    args = (xx for xx in zip(k_chunks, n_chunks, p_chunks))
    results = executor.map(compute_nbinom_pmf_chunk, args)

    return np.concatenate(list(results))

def thread_betabinom(k, n, a, b, num_threads=NUM_THREADS, executor=None):
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=NUM_THREADS)

    k_chunks = np.array_split(k, num_threads)
    n_chunks = np.array_split(n, num_threads)
    a_chunks = np.array_split(a, num_threads)
    b_chunks = np.array_split(b, num_threads)

    args = (xx for xx in zip(k_chunks, n_chunks, a_chunks, b_chunks))
    results = executor.map(compute_betabinom_pmf_chunk, args)

    return np.concatenate(list(results))
