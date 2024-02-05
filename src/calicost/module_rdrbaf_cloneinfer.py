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


def run_infer_rdrbafclone(config, foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, coords, barcodes, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat, merged_baf_assignment):
    # outdir
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"
    
    # create directory
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()
    
    # BAF clone info
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # For each BAF clone, further refine it using RDR+BAF signals
    for bafc in range(n_baf_clones):
        prefix = f"clone{bafc}"
        idx_spots = np.where(merged_baf_assignment == bafc)[0]
        if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 20: # put a minimum B allele read count on pseudobulk to split clones
            continue
        # initialize clone
        if config["tumorprop_file"] is None:
            initial_clone_index = rectangle_initialize_initial_clone(coords[idx_spots], config['n_clones_rdr'], random_state=r_hmrf_initialization)
        else:
            initial_clone_index = rectangle_initialize_initial_clone_mix(coords[idx_spots], config['n_clones_rdr'], single_tumor_prop[idx_spots], threshold=config["tumorprop_threshold"], random_state=r_hmrf_initialization)
        if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz").exists():
            initial_assignment = np.zeros(len(idx_spots), dtype=int)
            for c,idx in enumerate(initial_clone_index):
                initial_assignment[idx] = c
            allres = {"barcodes":barcodes[idx_spots], "num_iterations":0, "round-1_assignment":initial_assignment}
            np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", **allres)

        # HMRF + HMM using RDR information
        copy_slice_sample_ids = copy.copy(sample_ids[idx_spots])
        if config["tumorprop_file"] is None:
            hmrf_concatenate_pipeline(outdir, prefix, single_X[:,:,idx_spots], lengths, single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat[np.ix_(idx_spots,idx_spots)], adjacency_mat=adjacency_mat[np.ix_(idx_spots,idx_spots)], sample_ids=copy_slice_sample_ids, max_iter_outer=10, nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
        else:
            hmrfmix_concatenate_pipeline(outdir, prefix, single_X[:,:,idx_spots], lengths, single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], single_tumor_prop[idx_spots], initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat[np.ix_(idx_spots,idx_spots)], adjacency_mat=adjacency_mat[np.ix_(idx_spots,idx_spots)], sample_ids=copy_slice_sample_ids, max_iter_outer=10, nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"], tumorprop_threshold=config["tumorprop_threshold"])


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

    # load BAF clone results
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    bafclone_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_BAFCLONES}"
    merged_res = dict(np.load(f"{bafclone_outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", allow_pickle=True))
    merged_baf_assignment = merged_res["new_assignment"]

    # load ASE-filtered data matrix
    filter_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_FILTERASE}"
    df_bininfo = pd.read_csv(f"{filter_outdir}/table_bininfo.csv.gz", header=0, index_col=None, sep="\t")
    table_rdrbaf = pd.read_csv(f"{filter_outdir}/table_rdrbaf.csv.gz", header=0, index_col=None, sep="\t")
    # reconstruct single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, and lengths
    n_bins = df_bininfo.shape[0]
    n_spots = len(table_rdrbaf.BARCODES.unique())
    single_X = np.zeros((n_bins, 2, n_spots))
    single_X[:, 0, :] = table_rdrbaf["EXP"].values.reshape((n_bins, n_spots), order="F")
    single_X[:, 1, :] = table_rdrbaf["B"].values.reshape((n_bins, n_spots), order="F")
    single_base_nb_mean = df_bininfo["NORMAL_COUNT"].values.reshape(-1,1) / np.sum(df_bininfo["NORMAL_COUNT"].values) @ np.sum(single_X[:,0,:], axis=0).reshape(1,-1)
    single_total_bb_RD = table_rdrbaf["TOT"].values.reshape((n_bins, n_spots), order="F")
    log_sitewise_transmat = df_bininfo["LOG_PHASE_TRANSITION"].values
    lengths = np.array([ np.sum(df_bininfo.CHR == c) for c in df_bininfo.CHR.unique() ])

    # refine BAF clones using RDR+BAF signals
    run_infer_rdrbafclone(config, args.foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, coords, barcodes, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat, merged_baf_assignment)
