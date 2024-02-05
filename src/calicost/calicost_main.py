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
from calicost.phasing import *
from calicost.utils_IO import *
from calicost.find_integer_copynumber import *
from calicost.module_parse_input import *
from calicost.module_baf_cloneinfer import *
from calicost.module_filterase import *
from calicost.module_rdrbaf_cloneinfer import *
from calicost.module_combine_rdrbafclones import *
from calicost.module_integer_acn import *
from calicost.module_plot import *


def main(configuration_file):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    ##### Module 1: parse spaceranger and SNP-parsing output #####
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)
    
    ##### Module 2: infer clones based on BAF signals #####
    run_inferbaf_clone(config, FOLDER_BAFCLONES, single_X, single_total_bb_RD, lengths, log_sitewise_transmat, coords, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat)

    ##### Module 3: filter bins with potential allele-specific expression and save results #####
    # load BAF clone results
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    bafclone_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_BAFCLONES}"
    merged_res = dict(np.load(f"{bafclone_outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", allow_pickle=True))
    n_baf_clones = len(np.unique(merged_res["new_assignment"]))
    pred = np.argmax(merged_res["log_gamma"], axis=0)
    n_obs = single_X.shape[0]
    pred = np.array([ pred[(c*n_obs):(c*n_obs+n_obs)] for c in range(n_baf_clones) ])
    merged_baf_profiles = np.array([ np.where(pred[c,:] < config["n_states"], merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0], 1-merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0]) \
                                    for c in range(n_baf_clones) ])

    # filter allele-specific expression
    run_filter_ase(config, FOLDER_FILTERASE, single_X, single_base_nb_mean, single_total_bb_RD, lengths, df_gene_snp, barcodes, single_tumor_prop, sample_list, sample_ids, smooth_mat, exp_counts, merged_baf_profiles)
    
    ##### Module 4: infer clones based on RDR + BAF signals #####
    # adding RDR information
    if not config["bafonly"]:
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
        run_infer_rdrbafclone(config, FOLDER_RDRBAFCLONES, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, coords, barcodes, single_tumor_prop, sample_ids, adjacency_mat, smooth_mat, merged_res["new_assignment"])

    ##### Module 5: combine the refined clones #####
    if not config["bafonly"]:
        run_combine_rdrbafclones(config, FOLDER_RDRBAFCLONES, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, barcodes, single_tumor_prop, sample_ids, sample_list, adjacency_mat, smooth_mat, merged_res["new_assignment"])

    ##### Module 6: infer integer copy number #####
    if not config["bafonly"]:
        res_combine = dict(np.load(f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_RDRBAFCLONES}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))
        run_infer_integer_acn(config, FOLDER_INTEGER_CN, res_combine, single_X, single_base_nb_mean, single_total_bb_RD, barcodes, single_tumor_prop, df_gene_snp, df_bininfo)

    ##### Module 7: make plots #####
    if not config["bafonly"]:
        run_makeplots(config, FOLDER_PLOTS, configuration_file, res_combine, coords, single_tumor_prop, sample_ids, sample_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    main(args.configfile)