import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from KDEpy import FFTKDE
from scipy.special import psi, polygamma


# copied from sctransformPy
def theta_ml(y,mu):
    n = y.size
    weights = np.ones(n)
    limit = 10
    _EPS = np.finfo(float).eps
    eps = (_EPS)**0.25
    # inner function
    def score(n,th,mu,y,w):
        return sum(w*(psi(th + y) - psi(th) + np.log(th) + 1 - np.log(th + mu) - (y + th)/(mu + th)))
    # inner function
    def info(n,th,mu,y,w):
        return sum(w*( - polygamma(1,th + y) + polygamma(1,th) - 1/th + 2/(mu + th) - (y + th)/(mu + th)**2))
    # initialize gradient descent
    t0 = n/sum(weights*(y/mu - 1)**2)
    it = 0
    de = 1
    # gradient descent
    while(it + 1 < limit and abs(de) > eps):
        it+=1
        t0 = abs(t0)
        i = info(n, t0, mu, y, weights)
        de = score(n, t0, mu, y, weights)/i
        t0 += de        
    t0 = max(t0,0)
    # note that t0 is the dispersion parameter: var = mu + mu^2 / t0
    return t0


def sample_gene_indices(log_geometric_mean, n_subsample, n_partitions=10):
    bounds = np.linspace(np.min(log_geometric_mean), np.max(log_geometric_mean), n_partitions+1)
    bounds[-1] += 1e-4
    idx_subsample = []
    for p in range(1, n_partitions):
        tmpidx = np.where(np.logical_and(log_geometric_mean >= bounds[p-1], log_geometric_mean < bounds[p]))[0]
        np.random.shuffle(tmpidx)
        idx_subsample.append(tmpidx[:int(n_subsample/n_partitions)])
    idx_subsample = np.sort(np.concatenate(idx_subsample))
    if len(idx_subsample) < n_subsample:
        mask = np.array([True] * len(log_geometric_mean))
        mask[idx_subsample] = False
        idx_rest = np.arange(len(log_geometric_mean))[mask]
        np.random.shuffle(idx_rest)
        n_rest = n_subsample - len(idx_subsample)
        idx_subsample = np.sort(np.concatenate([idx_subsample, idx_rest[:n_rest]]))
    return idx_subsample


def estimate_logmu_dispersion(counts, bw=None):
    '''
    counts of size number spots * number genes.
    '''
    N = counts.shape[0]
    G = counts.shape[1]
    eps = 1
    geometric_mean = np.exp(np.log(counts+eps).mean(axis=0).flatten()) - eps
    log_geometric_mean = np.log( geometric_mean )
    spot_umi = counts.sum(axis=1)
    # fitting logmu and theta (dispersion)
    logmu = np.zeros(G)
    theta = np.zeros(G)
    for i in range(G):
        y = counts[:,i]
        logmu[i] = np.log( np.sum(y) / np.sum(spot_umi) )
        mu = spot_umi * np.exp(logmu[i])
        theta[i] = theta_ml(y, mu)
    # ratio between geometric mean and dispersion parameter theta
    log_ratio = np.log(1 + geometric_mean / theta)
    # smoothing parameter for kernel ridge regression
    if bw is None:
        z = FFTKDE(kernel='gaussian', bw='ISJ').fit(log_geometric_mean)
        z.evaluate();
        bw_adjust = 3
        bw = z.bw*bw_adjust
    # kernel ridge regression for log_ratio (the log ratio between geometric mean expression and dispersion)
    kr = statsmodels.nonparametric.kernel_regression.KernelReg(log_ratio, log_geometric_mean[:,None], ['c'], reg_type='ll', bw=[bw])
    pred_log_ratio = kr.fit(data_predict = log_geometric_mean[:,None])[0]
    pred_theta = geometric_mean / (np.exp(pred_log_ratio) - 1)
    return logmu, pred_theta


def pearson_residual(counts, logmu, pred_theta):
    '''
    counts of size number spots * number genes.
    '''
    N = counts.shape[0]
    G = counts.shape[1]
    spot_umi = counts.sum(axis=1)
    # predicted mean and variance under NB model
    mud = np.exp(logmu.reshape(1,-1)) * spot_umi.reshape(-1,1)
    vard = mud + mud**2 / pred_theta.reshape(1,-1)
    X = (counts * 1.0 - mud) / vard**0.5
    # clipping
    clip = np.sqrt(counts.shape[0]/30)
    X[X > clip] = clip
    X[X < -clip] = -clip
    return X


def deviance_residual(counts, logmu, pred_theta):
    '''
    Equation is taken from Analytic Pearson Residual paper by Lause et al.
    counts of size number spots * number genes.
    '''
    N = counts.shape[0]
    G = counts.shape[1]
    spot_umi = counts.sum(axis=1)
    # predicted mean
    mud = np.exp(logmu.reshape(1,-1)) * spot_umi.reshape(-1,1)
    sign = (counts > mud)
    part1 = counts * np.log(counts / mud)
    part1[counts==0] = 0
    part2 = (counts + pred_theta) * np.log( (counts + pred_theta) / (mud + pred_theta) )
    X = sign * np.sqrt(2 * (part1 - part2))
    return X


def estimate_logmu_dispersion2(counts, n_subsample=None, bw=None):
    '''
    counts of size number spots * number genes.
    '''
    N = counts.shape[0]
    G = counts.shape[1]
    eps = 1
    geometric_mean = np.exp(np.log(counts+eps).mean(axis=0).flatten()) - eps
    log_geometric_mean = np.log( geometric_mean )
    spot_umi = counts.sum(axis=1)
    logmu = np.log( np.sum(counts, axis=0) / np.sum(spot_umi) )
    # fitting theta (dispersion)
    genes_subsample = np.array([i for i in range(G) if geometric_mean[i] > 0])
    if not (n_subsample is None):
        np.random.seed(0)
        genes_subsample = sample_gene_indices(log_geometric_mean, n_subsample)
    theta = np.zeros(len(genes_subsample))
    for idx,i in enumerate(genes_subsample):
        y = counts[:,i]
        mu = spot_umi * np.exp(logmu[i])
        theta[idx] = theta_ml(y, mu)
    # ratio between geometric mean and dispersion parameter theta
    log_ratio = np.log(1 + geometric_mean[genes_subsample] / theta)
    # smoothing parameter for kernel ridge regression
    if bw is None:
        z = FFTKDE(kernel='gaussian', bw='ISJ').fit(log_geometric_mean[genes_subsample])
        z.evaluate();
        bw_adjust = 3
        bw = z.bw*bw_adjust
    # kernel ridge regression for log_ratio (the log ratio between geometric mean expression and dispersion)
    kr = statsmodels.nonparametric.kernel_regression.KernelReg(log_ratio, log_geometric_mean[genes_subsample][:,None], ['c'], reg_type='ll', bw=[bw])
    pred_log_ratio = kr.fit(data_predict = log_geometric_mean[:,None])[0]
    pred_theta = geometric_mean / (np.exp(pred_log_ratio) - 1)
    return logmu, pred_theta


def pearson_residual2(counts, logmu, pred_theta):
    '''
    counts of size number spots * number genes.
    '''
    N = counts.shape[0]
    G = counts.shape[1]
    spot_umi = counts.sum(axis=1)
    # predicted mean and variance under NB model
    mud = np.exp(logmu.reshape(1,-1)) * spot_umi.reshape(-1,1)
    vard = mud + mud**2 / pred_theta.reshape(1,-1)
    X = (counts * 1.0 - mud) / vard**0.5
    # clipping
    clip = np.sqrt(counts.shape[0])
    X[X > clip] = clip
    X[X < -clip] = -clip
    return X
