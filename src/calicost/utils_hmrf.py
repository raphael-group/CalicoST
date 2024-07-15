import numpy as np
import pandas as pd
import scipy.special
import scipy.sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import copy
import anndata
import scanpy as sc
from statsmodels.tools.sm_exceptions import ValueWarning
from calicost.utils_distribution_fitting import *
from calicost.utils_profiling import profile

def compute_adjacency_mat(coords, unit_xsquared=9, unit_ysquared=3):
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    # adjacency
    A = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.int8 )
    for i in range(coords.shape[0]):
        indexes = np.where(pairwise_squared_dist[i,:] <= unit_xsquared + unit_ysquared)[0]
        indexes = np.array([j for j in indexes if j != i])
        if len(indexes) > 0:
            A[i, indexes] = 1
    A = scipy.sparse.csr_matrix(A)
    return A


def compute_adjacency_mat_v2(coords, unit_xsquared=9, unit_ysquared=3, ratio=1):
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    # adjacency
    A = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.int8 )
    for i in range(coords.shape[0]):
        indexes = np.where(pairwise_squared_dist[i,:] <= ratio * (unit_xsquared + unit_ysquared))[0]
        indexes = np.array([j for j in indexes if j != i])
        if len(indexes) > 0:
            A[i, indexes] = 1
    A = scipy.sparse.csr_matrix(A)
    return A


def compute_weighted_adjacency(coords, unit_xsquared=9, unit_ysquared=3, bandwidth=12, decay=5):
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    kern = np.exp(-(pairwise_squared_dist / bandwidth)**decay)
    # adjacency
    A = np.zeros( (coords.shape[0], coords.shape[0]) )
    for i in range(coords.shape[0]):
        indexes = np.where(kern[i,:] > 1e-4)[0]
        indexes = np.array([j for j in indexes if j != i])
        if len(indexes) > 0:
            A[i, indexes] = kern[i,indexes]
    A = scipy.sparse.csr_matrix(A)
    return A


def choose_adjacency_by_readcounts(coords, single_total_bb_RD, maxspots_pooling=7, unit_xsquared=9, unit_ysquared=3):
# def choose_adjacency_by_readcounts(coords, single_total_bb_RD, count_threshold=4000, unit_xsquared=9, unit_ysquared=3):
    # XXX: change from count_threshold 500 to 3000
    # pairwise distance
    x_dist = coords[:,0][None,:] - coords[:,0][:,None]
    y_dist = coords[:,1][None,:] - coords[:,1][:,None]
    tmp_pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared
    np.fill_diagonal(tmp_pairwise_squared_dist, np.max(tmp_pairwise_squared_dist))
    base_ratio = np.median(np.min(tmp_pairwise_squared_dist, axis=0)) / (unit_xsquared + unit_ysquared)
    s_ratio = 0
    for ratio in range(0, 10):
        smooth_mat = compute_adjacency_mat_v2(coords, unit_xsquared, unit_ysquared, ratio * base_ratio)
        smooth_mat.setdiag(1)
        if np.median(np.sum(smooth_mat > 0, axis=0).A.flatten()) > maxspots_pooling:
            s_ratio = ratio - 1
            break
        s_ratio = ratio
    smooth_mat = compute_adjacency_mat_v2(coords, unit_xsquared, unit_ysquared, s_ratio * base_ratio)
    smooth_mat.setdiag(1)
    for bandwidth in np.arange(unit_xsquared + unit_ysquared, 15*(unit_xsquared + unit_ysquared), unit_xsquared + unit_ysquared):
        adjacency_mat = compute_weighted_adjacency(coords, unit_xsquared, unit_ysquared, bandwidth=bandwidth)
        adjacency_mat.setdiag(1)
        adjacency_mat = adjacency_mat - smooth_mat
        adjacency_mat[adjacency_mat < 0] = 0
        if np.median(np.sum(adjacency_mat, axis=0).A.flatten()) >= 6:
            print(f"bandwidth: {bandwidth}")
            break
    return smooth_mat, adjacency_mat


def choose_adjacency_by_KNN(coords, exp_counts=None, w=1, maxspots_pooling=7):
    """
    Compute adjacency matrix for pooling and for HMRF by KNN of pairwise spatial distance + pairwise expression distance.
    
    Attributes
    ----------
    coords : array, shape (n_spots, 2)
        Spatial coordinates of spots.

    exp_counts : None or array, shape (n_spots, n_genes)
        Expression counts of spots.

    w : float
        Weight of spatial distance in computing adjacency matrix.

    maxspots_pooling : int
        Number of spots in the adjacency matrix for pooling.
    """
    n_spots = coords.shape[0]

    # pairwise expression distance if exp_counts is not None
    pair_exp_dist = scipy.sparse.csr_matrix( np.zeros((n_spots,n_spots)) )
    scaling_factor = 1
    if not exp_counts is None:
        adata = anndata.AnnData( pd.DataFrame(exp_counts) )
        sc.pp.normalize_total(adata, target_sum=np.median(np.sum(exp_counts.values,axis=1)) )
        sc.pp.log1p(adata)
        sc.tl.pca(adata)
        pair_exp_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(adata.obsm["X_pca"]))
        # compute the scaling factor to normalize coords such that it has the same sum of variance as PCA
        var_coord = np.sum(np.var(coords, axis=0))
        var_pca = np.sum(np.var(adata.obsm["X_pca"], axis=0))
        EPS = 1e-4
        scaling_factor = np.sqrt(var_coord / var_pca) if var_coord > EPS and var_pca > EPS else 1

    # pairwise spatial distance
    pair_spatial_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords / scaling_factor))

    # adjacency for pooling
    smooth_mat = NearestNeighbors(n_neighbors=maxspots_pooling, metric='precomputed').fit(w * pair_spatial_dist + (1-w) * pair_exp_dist).kneighbors_graph()
    smooth_mat.setdiag(1) # include self adjacency

    # adjacency for HMRF
    adjacency_mat = NearestNeighbors(n_neighbors=maxspots_pooling + 6, metric='precomputed').fit(w * pair_spatial_dist + (1-w) * pair_exp_dist).kneighbors_graph()
    adjacency_mat = adjacency_mat - smooth_mat
    adjacency_mat[adjacency_mat < 0] = 0
    adjacency_mat.setdiag(1) # include self adjacency
    return smooth_mat, adjacency_mat


def choose_adjacency_by_readcounts_slidedna(coords, maxspots_pooling=30):
    """
    Merge spots such that 95% quantile of read count per SNP per spot exceed count_threshold.
    """
    smooth_mat = kneighbors_graph(coords, n_neighbors=maxspots_pooling)
    adjacency_mat = kneighbors_graph(coords, n_neighbors=maxspots_pooling + 6)
    adjacency_mat = adjacency_mat - smooth_mat
    return smooth_mat, adjacency_mat


def multislice_adjacency(sample_ids, sample_list, coords, single_total_bb_RD, exp_counts, across_slice_adjacency_mat, construct_adjacency_method, maxspots_pooling, construct_adjacency_w):
    adjacency_mat = []
    smooth_mat = []
    for i,sname in enumerate(sample_list):
        index = np.where(sample_ids == i)[0]
        this_coords = np.array(coords[index,:])
        if construct_adjacency_method == "hexagon":
            tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts(this_coords, single_total_bb_RD[:,index], maxspots_pooling=maxspots_pooling)
        elif construct_adjacency_method == "KNN":
            tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_KNN(this_coords, exp_counts.iloc[index,:], w=construct_adjacency_w, maxspots_pooling=maxspots_pooling)
        else:
            raise("Unknown adjacency construction method")
        # tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts_slidedna(this_coords, maxspots_pooling=config["maxspots_pooling"])
        adjacency_mat.append( tmpadjacency_mat.toarray() )
        smooth_mat.append( tmpsmooth_mat.toarray() )
    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)
    adjacency_mat = scipy.sparse.csr_matrix( adjacency_mat )
    if not across_slice_adjacency_mat is None:
        adjacency_mat += across_slice_adjacency_mat
    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix( smooth_mat )
    return adjacency_mat, smooth_mat


def rectangle_initialize_initial_clone(coords, n_clones, random_state=0):
    """
    Initialize clone assignment by partition space into p * p blocks (s.t. p * p >= n_clones), and assign each block a clone id.
    
    Attributes
    ----------
    coords : array, shape (n_spots, 2)
        2D coordinates of spots.

    n_clones : int
        Number of clones in initialization.

    Returns
    ----------
    initial_clone_index : list
        A list of n_clones np arrays, each array is the index of spots that belong to one clone.
    """
    np.random.seed(random_state)
    p = int(np.ceil(np.sqrt(n_clones)))
    # partition the range of x and y axes
    px = np.random.dirichlet( np.ones(p) * 10 )
    px[-1] += 1e-4
    xrange = [np.percentile(coords[:,0], 5), np.percentile(coords[:,0], 95)]
    xboundary = xrange[0] + (xrange[1] - xrange[0]) * np.cumsum(px)
    xboundary[-1] = np.max(coords[:,0]) + 1
    xdigit = np.digitize(coords[:,0], xboundary, right=True)
    py = np.random.dirichlet( np.ones(p) * 10 )
    py[-1] += 1e-4
    yrange = [np.percentile(coords[:,1], 5), np.percentile(coords[:,1], 95)]
    yboundary = yrange[0] + (yrange[1] - yrange[0]) * np.cumsum(py)
    yboundary[-1] = np.max(coords[:,1]) + 1
    ydigit = np.digitize(coords[:,1], yboundary, right=True)
    block_id = xdigit * p + ydigit
    # assigning blocks to clone (note that if sqrt(n_clone) is not an integer, multiple blocks can be assigneed to one clone)
    # block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)
    # while len(np.unique(block_clone_map)) < n_clones:
    #     bc = np.bincount(block_clone_map, minlength=n_clones)
    #     assert np.any(bc==0)
    #     block_clone_map[np.where(block_clone_map==np.argmax(bc))[0][0]] = np.where(bc==0)[0][0]
    # block_clone_map = {i:block_clone_map[i] for i in range(len(block_clone_map))}
    # clone_id = np.array([block_clone_map[i] for i in block_id])
    # initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
    while True:
        block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)
        while len(np.unique(block_clone_map)) < n_clones:
            bc = np.bincount(block_clone_map, minlength=n_clones)
            assert np.any(bc==0)
            block_clone_map[np.where(block_clone_map==np.argmax(bc))[0][0]] = np.where(bc==0)[0][0]
        block_clone_map = {i:block_clone_map[i] for i in range(len(block_clone_map))}
        clone_id = np.array([block_clone_map[i] for i in block_id])
        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
        if np.min([len(x) for x in initial_clone_index]) > 0.2 * coords.shape[0] / n_clones:
            break
    return initial_clone_index


def fixed_rectangle_initialization(coords, x_part, y_part):
    #
    px = np.linspace(0, 1, x_part+1)
    px[-1] += 0.01
    px = px[1:]
    xrange = [np.min(coords[:,0]), np.max(coords[:,0])]
    xdigit = np.digitize(coords[:,0], xrange[0] + (xrange[1] - xrange[0]) * px, right=True)
    #
    py = np.linspace(0, 1, y_part+1)
    py[-1] += 0.01
    py = py[1:]
    yrange = [np.min(coords[:,1]), np.max(coords[:,1])]
    ydigit = np.digitize(coords[:,1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True)
    #
    initial_clone_index = []
    for xid in range(x_part):
        for yid in range(y_part):
            initial_clone_index.append( np.where((xdigit == xid) & (ydigit == yid))[0] )
    return initial_clone_index


def merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index):
    n_obs = single_X.shape[0]
    n_spots = len(clone_index)
    X = np.zeros((n_obs, 2, n_spots))
    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))

    for k,idx in enumerate(clone_index):
        if len(idx) == 0:
            continue
        X[:,:, k] = np.sum(single_X[:,:,idx], axis=2)
        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)

    return X, base_nb_mean, total_bb_RD


def rectangle_initialize_initial_clone_mix(coords, n_clones, single_tumor_prop, threshold=0.5, random_state=0, EPS=1e-8):
    np.random.seed(random_state)
    p = int(np.ceil(np.sqrt(n_clones)))
    # partition the range of x and y axes based on tumor spots coordinates
    idx_tumor = np.where(single_tumor_prop > threshold)[0]
    px = np.random.dirichlet( np.ones(p) * 10 )
    px[-1] -= EPS
    xboundary = np.percentile(coords[idx_tumor, 0], 100*np.cumsum(px))
    xboundary[-1] = np.max(coords[:,0]) + 1
    xdigit = np.digitize(coords[:,0], xboundary, right=True)
    ydigit = np.zeros(coords.shape[0], dtype=int)
    for x in range(p):
        idx_tumor = np.where((single_tumor_prop > threshold) & (xdigit==x))[0]
        idx_both = np.where(xdigit == x)[0]
        py = np.random.dirichlet( np.ones(p) * 10 )
        py[-1] -= EPS
        yboundary = np.percentile(coords[idx_tumor, 1], 100*np.cumsum(py))
        yboundary[-1] = np.max(coords[:,1]) + 1
        ydigit[idx_both] = np.digitize(coords[idx_both,1], yboundary, right=True)
    block_id = xdigit * p + ydigit
    # assigning blocks to clone (note that if sqrt(n_clone) is not an integer, multiple blocks can be assigneed to one clone)
    block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)
    while len(np.unique(block_clone_map)) < n_clones:
        bc = np.bincount(block_clone_map, minlength=n_clones)
        assert np.any(bc==0)
        block_clone_map[np.where(block_clone_map==np.argmax(bc))[0][0]] = np.where(bc==0)[0][0]
    block_clone_map = {i:block_clone_map[i] for i in range(len(block_clone_map))}
    clone_id = np.array([block_clone_map[i] for i in block_id])
    initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
    return initial_clone_index


def fixed_rectangle_initialization_mix(coords, x_part, y_part, single_tumor_prop, threshold=0.5):
    idx_tumor = np.where(single_tumor_prop > threshold)[0]
    #
    px = np.linspace(0, 1, x_part+1)
    px[-1] += 0.01
    px = px[1:]
    xrange = [np.min(coords[idx_tumor,0]), np.max(coords[idx_tumor,0])]
    xdigit = np.digitize(coords[:,0], xrange[0] + (xrange[1] - xrange[0]) * px, right=True)
    #
    py = np.linspace(0, 1, y_part+1)
    py[-1] += 0.01
    py = py[1:]
    yrange = [np.min(coords[idx_tumor,1]), np.max(coords[idx_tumor,1])]
    ydigit = np.digitize(coords[:,1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True)
    #
    initial_clone_index = []
    for xid in range(x_part):
        for yid in range(y_part):
            initial_clone_index.append( np.where((xdigit == xid) & (ydigit == yid))[0] )
    return initial_clone_index


def merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop, threshold=0.5):
    n_obs = single_X.shape[0]
    n_spots = len(clone_index)
    X = np.zeros((n_obs, 2, n_spots))
    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))
    tumor_prop = np.zeros(n_spots)

    for k,idx in enumerate(clone_index):
        if len(idx) == 0:
            continue
        idx = idx[np.where(single_tumor_prop[idx] > threshold)[0]]
        X[:,:, k] = np.sum(single_X[:,:,idx], axis=2)
        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)
        tumor_prop[k] = np.mean(single_tumor_prop[idx]) if len(idx) > 0 else 0

    return X, base_nb_mean, total_bb_RD, tumor_prop


def reorder_results(res_combine, posterior, single_tumor_prop):
    EPS_BAF = 0.05
    n_spots = posterior.shape[0]
    n_obs = res_combine["pred_cnv"].shape[0]
    n_states, n_clones = res_combine["new_p_binom"].shape
    new_res_combine = copy.copy(res_combine)
    new_posterior = copy.copy(posterior)
    if single_tumor_prop is None:
        # select near-normal clone and set to clone 0
        pred_cnv = res_combine["pred_cnv"]
        baf_profiles = np.array([ res_combine["new_p_binom"][pred_cnv[:,c], c] for c in range(n_clones) ])
        cid_normal = np.argmin(np.sum( np.maximum(np.abs(baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
        cid_rest = np.array([c for c in range(n_clones) if c != cid_normal]).astype(int)
        reidx = np.append(cid_normal, cid_rest)
        map_reidx = {cid:i for i,cid in enumerate(reidx)}
        # re-order entries in res_combine
        new_res_combine["new_assignment"] = np.array([ map_reidx[c] for c in res_combine["new_assignment"] ])
        new_res_combine["new_log_mu"] = res_combine["new_log_mu"][:, reidx]
        new_res_combine["new_alphas"] = res_combine["new_alphas"][:, reidx]
        new_res_combine["new_p_binom"] = res_combine["new_p_binom"][:, reidx]
        new_res_combine["new_taus"] = res_combine["new_taus"][:, reidx]
        new_res_combine["log_gamma"] = res_combine["log_gamma"][:, :, reidx]
        new_res_combine["pred_cnv"] = res_combine["pred_cnv"][:, reidx]
        new_posterior = new_posterior[:, reidx]
    else:
        # add normal clone as clone 0
        new_res_combine["new_assignment"] = new_res_combine["new_assignment"] + 1
        new_res_combine["new_log_mu"] = np.hstack([np.zeros((n_states,1)), res_combine["new_log_mu"]])
        new_res_combine["new_alphas"] = np.hstack([np.zeros((n_states,1)), res_combine["new_alphas"]])
        new_res_combine["new_p_binom"] = np.hstack([0.5 * np.ones((n_states,1)), res_combine["new_p_binom"]])
        new_res_combine["new_taus"] = np.hstack([np.zeros((n_states,1)), res_combine["new_taus"]])
        new_res_combine["log_gamma"] = np.dstack([np.zeros((n_states, n_obs, 1)), res_combine["log_gamma"]])
        new_res_combine["pred_cnv"] = np.hstack([np.zeros((n_obs,1), dtype=int), res_combine["pred_cnv"]])
        new_posterior = np.hstack([np.ones((n_spots,1)) * np.nan, posterior])
    return new_res_combine, new_posterior


def reorder_results_merged(res, n_obs):
    n_clones = int(len(res["pred_cnv"]) / n_obs)
    EPS_BAF = 0.05
    pred_cnv = np.array([ res["pred_cnv"][(c*n_obs):(c*n_obs + n_obs)] for c in range(n_clones) ]).T
    baf_profiles = np.array([ res["new_p_binom"][pred_cnv[:,c], 0] for c in range(n_clones) ])
    cid_normal = np.argmin(np.sum( np.maximum(np.abs(baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
    cid_rest = np.array([c for c in range(n_clones) if c != cid_normal])
    reidx = np.append(cid_normal, cid_rest)
    map_reidx = {cid:i for i,cid in enumerate(reidx)}
    # re-order entries in res
    new_res = copy.copy(res)
    new_res["new_assignment"] = np.array([ map_reidx[c] for c in res["new_assignment"] ])
    new_res["log_gamma"] = np.hstack([ res["log_gamma"][:, (c*n_obs):(c*n_obs + n_obs)] for c in reidx ])
    new_res["pred_cnv"] = np.concatenate([ res["pred_cnv"][(c*n_obs):(c*n_obs + n_obs)] for c in reidx ])
    return new_res
    

def load_hmrf_last_iteration(filename):
    allres = dict( np.load(filename, allow_pickle=True) )
    r = allres["num_iterations"] - 1
    res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
        "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
        "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
        "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
        "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
    if "barcodes" in allres.keys():
        res["barcodes"] = allres["barcodes"]
    return res


def load_hmrf_given_iteration(filename, r):
    allres = dict( np.load(filename, allow_pickle=True) )
    res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
        "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
        "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
        "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
        "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
    if "barcodes" in allres.keys():
        res["barcodes"] = allres["barcodes"]
    return res


def identify_normal_spots(single_X, single_total_bb_RD, new_assignment, pred_cnv, p_binom, min_count, EPS_BAF=0.05, COUNT_QUANTILE=0.05, MIN_TOTAL=10):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele count per bin per spot.

    new_assignment : array, shape (n_spots,)
        Clone assignment for each spot.

    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.
    """
    # aggregate counts for each state, and evaluate the betabinomial likelihood given 0.5
    # spots with the highest likelihood are identified as normal spots
    n_obs = single_X.shape[0]
    n_spots = single_X.shape[2]
    n_clones = int(len(pred_cnv) / n_obs)
    n_states = p_binom.shape[0]
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order='F')

    baf_profiles = p_binom[reshaped_pred_cnv, 0].T
    id_nearnormal_clone = np.argmin(np.sum( np.maximum(np.abs(baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
    umi_quantile = np.quantile(np.sum(single_X[:,0,:], axis=0), COUNT_QUANTILE)
    
    baf_deviations = np.ones(n_spots)
    for i in range(n_spots):
        if new_assignment[i] == id_nearnormal_clone and np.sum(single_X[:,0,i]) >= umi_quantile:
            # enumerate the partition of all clones to aggregate counts, and list the BAF of each partition
            this_bafs = []
            for c in range(n_clones):
                agg_b_count = np.array([ np.sum(single_X[reshaped_pred_cnv[:,c]==s, 1, i]) for s in range(n_states) ])
                agg_t_count = np.array([ np.sum(single_total_bb_RD[reshaped_pred_cnv[:,c]==s, i]) for s in range(n_states) ])
                this_bafs.append( agg_b_count[agg_t_count>=MIN_TOTAL] / agg_t_count[agg_t_count>=MIN_TOTAL] )
            this_bafs = np.concatenate(this_bafs)
            baf_deviations[i] = np.max(np.abs(this_bafs - 0.5))

    sorted_idx = np.argsort(baf_deviations)
    summed_counts = np.cumsum( np.sum(single_X[:,0,sorted_idx], axis=0) )
    n_normal = np.where(summed_counts >= min_count)[0][0]

    return (baf_deviations <= baf_deviations[sorted_idx[n_normal]])


# def identify_loh_per_clone(single_X, new_assignment, pred_cnv, p_binom, normal_candidate, MIN_BAF_DEVIATION_RANGE=[0.25, 0.12], MIN_BINS_PER_STATE=10, MIN_BINS_ALL=50):
#     """
#     Attributes
#     ----------
#     single_X : array, shape (n_obs, 2, n_spots)
#         Observed transcript counts and B allele count per bin per spot.

#     new_assignment : array, shape (n_spots,)
#         Clone assignment for each spot.
    
#     pred_cnv : array, shape (n_obs * n_clones)
#         Copy number states across bins for each clone.

#     p_binom : array, shape (n_states, 1)
#         Estimated BAF per copy number state (shared across clones).

#     Returns
#     ----------
#     loh_states : array
#         An array of copy number states that are identified as LOH.

#     is_B_loss : array
#         A boolean array indicating whether B allele is lost (alternative A allele is lost).

#     rdr_values : array
#         An array of RDR values corresponding to LOH states.
#     """
#     n_obs = single_X.shape[0]
#     n_clones = int(len(pred_cnv) / n_obs)
#     n_states = p_binom.shape[0]
#     reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order='F')
#     # clones that have a decent tumor proportion
#     # for each clone, if the clones_hightumor-th BAF deviation is large enough
#     k_baf_deviation = np.sort( np.abs(p_binom[reshaped_pred_cnv, 0]-0.5), axis=0)[-MIN_BINS_ALL,:]
#     clones_hightumor = np.where(k_baf_deviation >= MIN_BAF_DEVIATION_RANGE[1])[0]
#     if len(clones_hightumor) == 0:
#         clones_hightumor = np.argsort(k_baf_deviation)[-1:]
#     if len(clones_hightumor) == n_clones:
#         clones_hightumor = np.argsort(k_baf_deviation)[1:]
#     print(f"clones with high tumor proportion: {clones_hightumor}")
#     # LOH states
#     for threshold in np.arange(MIN_BAF_DEVIATION_RANGE[0], MIN_BAF_DEVIATION_RANGE[1]-0.01, -0.01):
#         loh_states = np.where( (np.abs(p_binom[:,0] - 0.5) > threshold) & (np.bincount(pred_cnv, minlength=n_states) >= MIN_BINS_PER_STATE) )[0]
#         is_B_lost = (p_binom[loh_states,0] < 0.5)
#         if np.all([ np.sum(pd.Series(reshaped_pred_cnv[:,c]).isin(loh_states)) >= MIN_BINS_ALL for c in clones_hightumor ]):
#             print(f"BAF deviation threshold = {threshold}, LOH states: {loh_states}")
#             break
#     # RDR values
#     # first get the normal baseline expression per spot per bin
#     simple_rdr_normal = np.sum(single_X[:, 0, (normal_candidate==True)], axis=1)
#     simple_rdr_normal = simple_rdr_normal / np.sum(simple_rdr_normal)
#     simple_single_base_nb_mean = simple_rdr_normal.reshape(-1,1) @ np.sum(single_X[:,0,:], axis=0).reshape(1,-1)
#     # then aggregate to clones
#     clone_index = [np.where(new_assignment == c)[0] for c in range(n_clones)]
#     X, base_nb_mean, _ = merge_pseudobulk_by_index(single_X, simple_single_base_nb_mean, np.zeros(simple_single_base_nb_mean.shape), clone_index)
#     rdr_values = []
#     for s in loh_states:
#         rdr_values.append( np.sum(X[:,0,:][reshaped_pred_cnv==s]) / np.sum(base_nb_mean[reshaped_pred_cnv==s]) )
#     rdr_values = np.array(rdr_values)

#     """
#     Update ideas: why not finding high purity clone and loh states together by varying BAF deviation threshold?
#     Current we first identify high purity clone using BAF deviation threshold = 0.15, then identify loh states.
#     But we can vary BAF deviation threshold from the large to small, identify high purity clones and loh states based on the same threshold.
#     At very large threshold value, there will be no high purity clone, which is unreasonable. 
#     While lowering the threshold, purity clone(s) will appear, and we terminate once we are able to find one high purity clone.

#     Another update idea: identification of loh states is unaware of RDR. 
#     We can first find low-copy-number loh states first by thresholding RDR. If we can't find any, increase RDR threshold.
#     """

#     return loh_states, is_B_lost, rdr_values, clones_hightumor


def identify_loh_per_clone(single_X, new_assignment, pred_cnv, p_binom, normal_candidate, single_total_bb_RD, MIN_SNPUMI=10, MAX_RDR=1, MIN_BAF_DEVIATION_RANGE=[0.25, 0.12], MIN_BINS_PER_STATE=10, MIN_BINS_ALL=25):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    new_assignment : array, shape (n_spots,)
        Clone assignment for each spot.
    
    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.

    p_binom : array, shape (n_states, 1)
        Estimated BAF per copy number state (shared across clones).

    Returns
    ----------
    loh_states : array
        An array of copy number states that are identified as LOH.

    is_B_loss : array
        A boolean array indicating whether B allele is lost (alternative A allele is lost).

    rdr_values : array
        An array of RDR values corresponding to LOH states.
    """
    n_obs = single_X.shape[0]
    n_clones = int(len(pred_cnv) / n_obs)
    n_states = p_binom.shape[0]
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order='F')
    
    # per-state RDR values
    # first get the normal baseline expression per spot per bin
    simple_rdr_normal = np.sum(single_X[:, 0, (normal_candidate==True)], axis=1)
    simple_rdr_normal = simple_rdr_normal / np.sum(simple_rdr_normal)
    simple_single_base_nb_mean = simple_rdr_normal.reshape(-1,1) @ np.sum(single_X[:,0,:], axis=0).reshape(1,-1)
    # then aggregate to clones
    clone_index = [np.where(new_assignment == c)[0] for c in range(n_clones)]
    X, base_nb_mean, _ = merge_pseudobulk_by_index(single_X, simple_single_base_nb_mean, np.zeros(simple_single_base_nb_mean.shape), clone_index)
    rdr_values = []
    for s in np.arange(n_states):
        rdr_values.append( np.sum(X[:,0,:][reshaped_pred_cnv==s]) / np.sum(base_nb_mean[reshaped_pred_cnv==s]) )
    rdr_values = np.array(rdr_values)

    # SNP-covering UMI per clone
    clone_snpumi = np.array([np.sum(single_total_bb_RD[:,new_assignment==c]) for c in range(n_clones)])

    # clones that have a decent tumor proportion
    # for each clone, if the clones_hightumor-th BAF deviation is large enough
    k_baf_deviation = np.sort( np.abs(p_binom[reshaped_pred_cnv, 0]-0.5), axis=0)[-MIN_BINS_ALL,:]
    # LOH states
    for threshold in np.arange(MIN_BAF_DEVIATION_RANGE[0], MIN_BAF_DEVIATION_RANGE[1]-0.01, -0.02):
        clones_hightumor = np.where( (k_baf_deviation >= threshold) & (clone_snpumi >= MIN_SNPUMI*n_obs) )[0]
        if len(clones_hightumor) == 0:
            continue
        if len(clones_hightumor) == n_clones:
            clones_hightumor = np.argsort(k_baf_deviation)[1:]
        # LOH states
        loh_states = np.where( (np.abs(p_binom[:,0] - 0.5) > threshold) & (np.bincount(pred_cnv, minlength=n_states) >= MIN_BINS_PER_STATE) & (rdr_values <= MAX_RDR) )[0]
        is_B_lost = (p_binom[loh_states,0] < 0.5)
        if np.all([ np.sum(pd.Series(reshaped_pred_cnv[:,c]).isin(loh_states)) >= MIN_BINS_ALL for c in clones_hightumor ]):
            print(f"threshold = {threshold}")
            print(f"clones with high tumor proportion: {clones_hightumor}")
            print(f"BAF deviation threshold = {threshold}, LOH states: {loh_states}")
            break

    """
    Update ideas: why not finding high purity clone and loh states together by varying BAF deviation threshold?
    Current we first identify high purity clone using BAF deviation threshold = 0.15, then identify loh states.
    But we can vary BAF deviation threshold from the large to small, identify high purity clones and loh states based on the same threshold.
    At very large threshold value, there will be no high purity clone, which is unreasonable. 
    While lowering the threshold, purity clone(s) will appear, and we terminate once we are able to find one high purity clone.

    Another update idea: identification of loh states is unaware of RDR. 
    We can first find low-copy-number loh states first by thresholding RDR. If we can't find any, increase RDR threshold.
    """

    return loh_states, is_B_lost, rdr_values[loh_states], clones_hightumor


def estimator_tumor_proportion(single_X, single_total_bb_RD, assignments, pred_cnv, loh_states, is_B_lost, rdr_values, clone_to_consider, smooth_mat=None, MIN_TOTAL=10):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele count per bin per spot.

    assignments : pd.DataFrame of size n_spots with columns "coarse", "combined" 
        Clone assignment for each spot.

    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.
    
    loh_states, is_B_lost, rdr_values: array
        Copy number states and RDR values corresponding to LOH.

    Formula
    ----------
    0.5 ( 1-theta ) / (theta * RDR + 1 - theta) = B_count / Total_count for each LOH state.
    """
    # def estimate_purity(T_loh, B_loh, rdr_values):
    #     features =(T_loh / 2.0 + rdr_values * B_loh - B_loh)[T_loh>0].reshape(-1,1)
    #     y = (T_loh / 2.0 - B_loh)[T_loh>0]
    #     return np.linalg.lstsq(features, y, rcond=None)[0]
    def estimate_purity(T_loh, B_loh, rdr_values):
        idx = np.where(T_loh > 0)[0]
        model = BAF_Binom(endog=B_loh[idx], exog=np.ones((len(idx),1)), weights=np.ones(len(idx)), exposure=T_loh[idx], offset=np.log(rdr_values[idx]), scaling=0.5)
        res = model.fit(disp=False)
        return 1.0 / (1.0 + np.exp(res.params))
    #
    n_obs = single_X.shape[0]
    n_spots = single_X.shape[2]
    n_clones = int(len(pred_cnv) / n_obs)
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order='F')

    clone_mapping = assignments.groupby(['coarse', 'combined']).agg('first').reset_index()

    tumor_proportion = np.zeros(n_spots)
    full_tumor_proportion = np.zeros((n_spots, n_clones))
    for i in range(n_spots):
        # get adjacent spots for smoothing
        if smooth_mat is not None:
            idx_adj = smooth_mat[i,:].nonzero()[1]
        else:
            idx_adj = np.array([i])
        estimation_based_on_clones_single = np.ones(n_clones) * np.nan
        estimation_based_on_clones_smoothed = np.ones(n_clones) * np.nan
        summed_T_single = np.ones(n_clones)
        summed_T_smoothed = np.ones(n_clones)
        for c in clone_to_consider:
            # single
            B_loh = np.array([ np.sum(single_X[:,1,i][reshaped_pred_cnv[:,c]==s]) if is_B_lost[j] else np.sum(single_total_bb_RD[:,i][reshaped_pred_cnv[:,c]==s]) - np.sum(single_X[:,1,i][reshaped_pred_cnv[:,c]==s]) for j,s in enumerate(loh_states)])
            T_loh = np.array([ np.sum(single_total_bb_RD[:,i][reshaped_pred_cnv[:,c]==s]) for s in loh_states])
            if np.all(T_loh == 0):
                continue
            estimation_based_on_clones_single[c] = estimate_purity(T_loh, B_loh, rdr_values)
            summed_T_single[c] = np.sum(T_loh)
            # smoothed
            B_loh = np.array([ np.sum(single_X[:,1,idx_adj][reshaped_pred_cnv[:,c]==s]) if is_B_lost[j] else np.sum(single_total_bb_RD[:,idx_adj][reshaped_pred_cnv[:,c]==s]) - np.sum(single_X[:,1,idx_adj][reshaped_pred_cnv[:,c]==s]) for j,s in enumerate(loh_states)])
            T_loh = np.array([ np.sum(single_total_bb_RD[:,idx_adj][reshaped_pred_cnv[:,c]==s]) for s in loh_states])
            if np.all(T_loh == 0):
                continue
            estimation_based_on_clones_smoothed[c] = estimate_purity(T_loh, B_loh, rdr_values)
            summed_T_smoothed[c] = np.sum(T_loh)
        full_tumor_proportion[i,:] = estimation_based_on_clones_single
        if (assignments.combined.values[i] in clone_to_consider) and summed_T_single[assignments.combined.values[i]] >= MIN_TOTAL:
            tumor_proportion[i] = estimation_based_on_clones_single[ assignments.combined.values[i] ]
        elif (assignments.combined.values[i] in clone_to_consider) and summed_T_smoothed[assignments.combined.values[i]] >= MIN_TOTAL:
            tumor_proportion[i] = estimation_based_on_clones_smoothed[ assignments.combined.values[i] ]
        elif not assignments.combined.values[i] in clone_to_consider:
            tumor_proportion[i] = estimation_based_on_clones_single[np.argmax(summed_T_single)]
        else:
            tumor_proportion[i] = np.nan

    tumor_proportion = np.where(tumor_proportion < 0, 0, tumor_proportion)
    return tumor_proportion, full_tumor_proportion
