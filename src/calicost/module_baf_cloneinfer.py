import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import scanpy as sc
import anndata
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
import copy
from pathlib import Path
import functools
import subprocess
from calicost.arg_parse import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.utils_distribution_fitting import *
from calicost.utils_hmrf import *
from calicost.hmrf import *
from calicost.utils_IO import *
from calicost.module_parse_input import *


def run_inferbaf_clone(config, foldername, single_X, single_total_bb_RD, lengths, log_sitewise_transmat, coords, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat):
    # setting RD counts by zero for not considering RDR
    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    single_X[:,0,:] = 0
    single_base_nb_mean = np.zeros(single_total_bb_RD.shape)

    # random seed
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"
    if config["tumorprop_file"] is None:
        initial_clone_index = rectangle_initialize_initial_clone(coords, config["n_clones"], random_state=r_hmrf_initialization)
    else:
        initial_clone_index = rectangle_initialize_initial_clone_mix(coords, config["n_clones"], single_tumor_prop, threshold=config["tumorprop_threshold"], random_state=r_hmrf_initialization)

    # create directory
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()
    # save clone initialization into npz file
    prefix = "allspots"
    if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz").exists():
        initial_assignment = np.zeros(single_X.shape[2], dtype=int)
        for c,idx in enumerate(initial_clone_index):
            initial_assignment[idx] = c
        allres = {"num_iterations":0, "round-1_assignment":initial_assignment}
        np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz", **allres)

    # run HMRF + HMM
    if config["tumorprop_file"] is None:
        hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_states"], \
            log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
            hmmclass=hmm_nophasing_v2, params="sp", t=config["t"], random_state=config["gmm_random_state"], \
            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
            is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
    else:
        hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states=config["n_states"], \
            log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
            hmmclass=hmm_nophasing_v2, params="sp", t=config["t"], random_state=config["gmm_random_state"], \
            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
            is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"], tumorprop_threshold=config["tumorprop_threshold"])
    
    # merge by thresholding BAF profile similarity
    res = load_hmrf_last_iteration(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz")
    n_obs = single_X.shape[0]
    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))])
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))], single_tumor_prop, threshold=config["tumorprop_threshold"])
        tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1,1)
    merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, threshold=config["np_threshold"], minlength=config["np_eventminlen"], params="sp", tumor_prop=tumor_prop, hmmclass=hmm_nophasing_v2)
    print(f"BAF clone merging after comparing similarity: {merging_groups}")
    #
    if config["tumorprop_file"] is None:
        merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD, min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=config["min_avgumi_per_clone"]*n_obs)
    else:
        merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD, min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=config["min_avgumi_per_clone"]*n_obs, single_tumor_prop=single_tumor_prop, threshold=config["tumorprop_threshold"])
    print(f"BAF clone merging after requiring minimum # spots: {merging_groups}")
    n_baf_clones = len(merging_groups)
    np.savez(f"{outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", **merged_res)

    # reset single_X
    single_X[:,0,:] = copy_single_X_rdr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    parser.add_argument("--foldername", help="folder name to save inferred BAF clone results", required=True, type=str)
    args = parser.parse_args()

    try:
        config = read_configuration_file(args.configfile)
    except:
        config = read_joint_configuration_file(args.configfile)

    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")
    
    # load initial parsed input
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)

    run_inferbaf_clone(config, args.foldername, single_X, single_total_bb_RD, lengths, log_sitewise_transmat, coords, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat)
