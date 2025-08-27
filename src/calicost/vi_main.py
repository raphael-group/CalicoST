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
from calicost.parse_input import *
from calicost.utils_plotting import *
from calicost.sample_posterior_clone import *
from calicost.hmm_NB_BB_nophasing_float import *


def main(configuration_file):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)
    
    single_tumor_prop[np.isnan(single_tumor_prop)] = 0.05

    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)
    single_X[:,0,:] = 0
    single_base_nb_mean[:,:] = 0

    # # remove spots with tumor purity less than config['tumorprop_threshold']
    # single_X = single_X[:,:,single_tumor_prop > config['tumorprop_threshold']]
    # single_base_nb_mean = single_base_nb_mean[:,single_tumor_prop > config['tumorprop_threshold']]
    # single_total_bb_RD = single_total_bb_RD[:,single_tumor_prop > config['tumorprop_threshold']]
    # barcodes = barcodes[single_tumor_prop > config['tumorprop_threshold']]
    # coords = coords[single_tumor_prop > config['tumorprop_threshold'],:]
    # sample_ids = sample_ids[single_tumor_prop > config['tumorprop_threshold']]
    # adjacency_mat = adjacency_mat[single_tumor_prop > config['tumorprop_threshold'],:][:,single_tumor_prop > config['tumorprop_threshold']]
    # smooth_mat = smooth_mat[single_tumor_prop > config['tumorprop_threshold'],:][:,single_tumor_prop > config['tumorprop_threshold']]
    # single_tumor_prop = single_tumor_prop[single_tumor_prop > config['tumorprop_threshold']]

    # Run HMRF
    for r_hmrf_initialization in range(config["num_hmrf_initialization_start"], config["num_hmrf_initialization_end"]):
        outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
        if config["tumorprop_file"] is None:
            initial_clone_index = rectangle_initialize_initial_clone(coords, config["n_clones"], random_state=r_hmrf_initialization)
        else:
            initial_clone_index = rectangle_initialize_initial_clone_mix(coords, config["n_clones"], single_tumor_prop, threshold=config["tumorprop_threshold"], random_state=r_hmrf_initialization)
        
        # create directory
        p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = p.communicate()
        # save clone initialization into npz file
        prefix = "vi_gibbs_hightp_allspots"
        initial_assignment = np.zeros(single_X.shape[2], dtype=int)
        for c,idx in enumerate(initial_clone_index):
            initial_assignment[idx] = c

        posterior_clones = np.zeros((single_X.shape[2], config['n_clones']))
        posterior_clones[ (np.arange(single_X.shape[2]), initial_assignment) ] = 1

        # run HMRF + HMM
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop, threshold=config["tumorprop_threshold"])
        init_log_mu, init_p_binom = initialization_by_gmm(config['n_states'], np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params='sp', random_state=config['gmm_random_state'], in_log_space=False, only_minor=False)
        
        list_posterior_clones, list_cna_states, list_emission_llf, list_log_mu,list_alphas, list_p_binom, list_taus, list_log_startprob, list_elbo, list_h, list_labels = infer_all_v2(
                            single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, posterior_clones, config['n_states'],
                            coords, adjacency_mat, config['tumorprop_threshold'], config['spatial_weight'], 'visium', max_iter_outer=20, num_chains=20, burnin=50, sampling_tol=1e-10, temperature=2.0,
                            hmmclass=hmm_nophasing_float, hmm_params='sp', hmm_t=config['t'], hmm_random_state=config['gmm_random_state'], hmm_max_iter=config['max_iter'], hmm_tol=config['tol'], hmm_num_draws=200, fun=block_gibbs_sampling_labels, 
                            smooth_mat=smooth_mat, init_p_binom=init_p_binom)
        
        # save results
        np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz", 
                 list_posterior_clones=list_posterior_clones, 
                 list_cna_states=list_cna_states, 
                 list_emission_llf=list_emission_llf,
                 list_log_mu=list_log_mu, 
                 list_p_binom=list_p_binom, 
                 list_elbo=list_elbo, 
                 list_h=list_h, 
                 list_labels=list_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    main(args.configfile)