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
        scaling_factor = np.sqrt(var_coord / var_pca)

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


def sample_initialize_initial_clone(adata, sample_list, n_clones, random_state=0):
    np.random.seed(random_state)
    occurences = 1 + np.random.multinomial(len(sample_list) - n_clones, pvals=np.ones(n_clones) / n_clones)
    sample_clone_id = sum([[i] * occurences[i] for i in range(len(occurences))], [])
    sample_clone_id = np.array(sample_clone_id)
    np.random.shuffle(sample_clone_id)
    print(sample_clone_id)
    clone_id = np.zeros(adata.shape[0], dtype=int)
    for i, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        clone_id[index] = sample_clone_id[i]
    print(np.bincount(clone_id))
    initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
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


def rectangle_initialize_initial_clone_mix(coords, n_clones, single_tumor_prop, threshold=0.5, random_state=0):
    np.random.seed(random_state)
    p = int(np.ceil(np.sqrt(n_clones)))
    # partition the range of x and y axes based on tumor spots coordinates
    idx_tumor = np.where(single_tumor_prop > threshold)[0]
    px = np.random.dirichlet( np.ones(p) * 10 )
    xboundary = np.percentile(coords[idx_tumor, 0], 100*np.cumsum(px))
    xboundary[-1] = np.max(coords[:,0]) + 1
    xdigit = np.digitize(coords[:,0], xboundary, right=True)
    ydigit = np.zeros(coords.shape[0], dtype=int)
    for x in range(p):
        idx_tumor = np.where((single_tumor_prop > threshold) & (xdigit==x))[0]
        idx_both = np.where(xdigit == x)[0]
        py = np.random.dirichlet( np.ones(p) * 10 )
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


def get_LoH_for_phylogeny(df_seglevel_cnv, min_num_bins=3):
    """
    Treating LoH as irreversible point mutations, output a clone-by-mutation matrix for phylogeny reconstruction.
    Mutation states: 0 for no LoH, 1 for lossing A allele, 2 for lossing B allele.

    Attributes
    ----------
    df_seglevel_cnv : pd.DataFrame, (n_obs, 3+2*n_clones)
        Dataframe from cnv_*seglevel.tsv output.

    Returns
    ----------
    df_loh : pd.DataFrame, (n_clones, n_segments)
    """
    def get_shared_intervals(acn_profile):
        '''
        Takes in allele-specific copy numbers, output a segmentation of genome such that all clones are in the same CN state within each segment.

        anc_profile : array, (n_obs, 2*n_clones)
            Allele-specific integer copy numbers for each genomic bin (obs) across all clones.
        '''
        intervals = []
        seg_acn = []
        s = 0
        while s < acn_profile.shape[0]:
            t = np.where( ~np.all(acn_profile[s:,] == acn_profile[s,:], axis=1) )[0]
            if len(t) == 0:
                intervals.append( (s, acn_profile.shape[0])  )
                seg_acn.append( acn_profile[s,:] )
                s = acn_profile.shape[0]
            else:
                t = t[0]
                intervals.append( (s,s+t) )
                seg_acn.append( acn_profile[s,:] )
                s = s+t
        return intervals, seg_acn
    
    clone_ids = [x.split(" ")[0] for x in df_seglevel_cnv.columns[ np.arange(3, df_seglevel_cnv.shape[1], 2) ] ]
    
    acn_profile = df_seglevel_cnv.iloc[:,3:].values
    intervals, seg_acn = get_shared_intervals(acn_profile)
    df_loh = []
    for i, acn in enumerate(seg_acn):
        if np.all(acn != 0):
            continue
        if intervals[i][1] - intervals[i][0] < min_num_bins:
            continue
        idx_zero = np.where(acn == 0)[0]
        idx_clones = (idx_zero / 2).astype(int)
        is_A = (idx_zero % 2 == 0)
        # vector of mutation states
        mut = np.zeros( int(len(acn) / 2), dtype=int )
        mut[idx_clones] = np.where(is_A, 1, 2)
        df_loh.append( pd.DataFrame(mut.reshape(1, -1), index=[f"bin_{intervals[i][0]}_{intervals[i][1]}"], columns=clone_ids) )

    df_loh = pd.concat(df_loh).T
    return df_loh