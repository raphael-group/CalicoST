import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
import copy
import functools
import subprocess
from calicost.arg_parse import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.parse_input import *
from calicost.utils_hmrf import *
from calicost.hmrf import *


def main(configuration_file):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)

    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)
    
    single_base_nb_mean[:,:] = 0

    n_states_for_tumorprop = 5
    n_clones_for_tumorprop = 3
    max_outer_iter_for_tumorprop = 10
    max_iter_for_tumorprop = 20
    MIN_PROP_UNCERTAINTY = 0.05
    initial_clone_index = rectangle_initialize_initial_clone(coords, n_clones_for_tumorprop, random_state=0)
    # save clone initialization into npz file
    prefix = "initialhmm"
    if not Path(f"{config['output_dir']}/{prefix}_nstates{n_states_for_tumorprop}_sp.npz").exists():
        initial_assignment = np.zeros(single_X.shape[2], dtype=int)
        for c,idx in enumerate(initial_clone_index):
            initial_assignment[idx] = c
        allres = {"num_iterations":0, "round-1_assignment":initial_assignment}
        np.savez(f"{config['output_dir']}/{prefix}_nstates{n_states_for_tumorprop}_sp.npz", **allres)

    hmrf_concatenate_pipeline(config['output_dir'], prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=n_states_for_tumorprop, \
            log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=max_outer_iter_for_tumorprop, nodepotential=config["nodepotential"], \
            hmmclass=hmm_nophasing_v2, params="sp", t=config["t"], random_state=config["gmm_random_state"], \
            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
            is_diag=True, max_iter=max_iter_for_tumorprop, tol=config["tol"], spatial_weight=config["spatial_weight"])
    
    res = load_hmrf_last_iteration(f"{config['output_dir']}/{prefix}_nstates{n_states_for_tumorprop}_sp.npz")
    merging_groups, merged_res = merge_by_minspots(res["new_assignment"], res, single_total_bb_RD, min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=config["min_avgumi_per_clone"]*single_X.shape[0])

    # loh_states, is_B_lost = identify_loh_per_clone(single_X, res['new_assignment'], res['pred_cnv'], res['new_p_binom'], normal_candidate=None)
    # normal_candidate, rdr_values = get_rdr_for_loh_states(single_X, single_total_bb_RD, res['new_assignment'], res['pred_cnv'], loh_states, is_B_lost, min_count=single_X.shape[0] * 200, MIN_TOTAL=10)
    # single_tumor_prop, _ = estimator_tumor_proportion(single_X, single_total_bb_RD, res['new_assignment'], res['pred_cnv'], loh_states, is_B_lost, rdr_values)

    normal_candidate = identify_normal_spots(single_X, single_total_bb_RD, merged_res['new_assignment'], merged_res['pred_cnv'], merged_res['new_p_binom'], min_count=single_X.shape[0] * 200)
    loh_states, is_B_lost, rdr_values = identify_loh_per_clone(single_X, merged_res['new_assignment'], merged_res['pred_cnv'], merged_res['new_p_binom'], normal_candidate)
    single_tumor_prop, _ = estimator_tumor_proportion(single_X, single_total_bb_RD, merged_res['new_assignment'], merged_res['pred_cnv'], loh_states, is_B_lost, rdr_values)
    single_tumor_prop = np.where(single_tumor_prop < MIN_PROP_UNCERTAINTY, MIN_PROP_UNCERTAINTY, single_tumor_prop)
    single_tumor_prop[normal_candidate] = 0
    # save single_tumor_prop to file
    pd.DataFrame({"Tumor":single_tumor_prop}, index=barcodes).to_csv(f"{config['output_dir']}/loh_estimator_tumor_prop.tsv", header=True, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    main(args.configfile)