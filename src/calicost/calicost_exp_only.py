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


def main(configuration_file):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    ########## Proprocess input data ##########
    if "input_filelist" in config:
        adata, across_slice_adjacency_mat = load_joint_data_exp_only(config["input_filelist"], config["alignment_files"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"], config['min_snpumi_perspot'], config['min_percent_expressed_spots'])
        sample_list = [adata.obs["sample"][0]]
        for i in range(1, adata.shape[0]):
            if adata.obs["sample"][i] != sample_list[-1]:
                sample_list.append( adata.obs["sample"][i] )
        # convert sample name to index
        sample_ids = np.zeros(adata.shape[0], dtype=int)
        for s,sname in enumerate(sample_list):
            index = np.where(adata.obs["sample"] == sname)[0]
            sample_ids[index] = s
    else:
        adata = load_data_exp_only(config["spaceranger_dir"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"], config['min_snpumi_perspot'], config['min_percent_expressed_spots'])
        adata.obs["sample"] = "unique_sample"
        sample_list = [adata.obs["sample"][0]]
        sample_ids = np.zeros(adata.shape[0], dtype=int)
        across_slice_adjacency_mat = None

    coords = adata.obsm["X_pos"]

    if not config["tumorprop_file"] is None:
        df_tumorprop = pd.read_csv(config["tumorprop_file"], sep="\t", header=0, index_col=0)
        df_tumorprop = df_tumorprop[["Tumor"]]
        df_tumorprop.columns = ["tumor_proportion"]
        adata.obs = adata.obs.join(df_tumorprop)
        single_tumor_prop = adata.obs["tumor_proportion"]
    else:
        single_tumor_prop = None

    # binning
    df_gene_snp = create_genetable_exp_only(config['hgtable_file'], adata)
    df_gene_snp = create_bin_ranges_exp_only(df_gene_snp, adata, min_genes=3, secondary_min_umi=config['secondary_min_umi'])
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat = summarize_counts_for_blocks(df_gene_snp, \
            adata, np.empty(0), np.empty(0), np.empty(0), nu=config['nu'], logphase_shift=config['logphase_shift'], geneticmap_file=config['geneticmap_file'])
    
    df_bininfo = genesnp_to_bininfo(df_gene_snp)

    adjacency_mat, smooth_mat = multislice_adjacency(sample_ids, sample_list, coords, single_total_bb_RD, pd.DataFrame(adata.layers['count'], index=adata.obs.index, columns=adata.var.index), 
                                                     across_slice_adjacency_mat, construct_adjacency_method=config['construct_adjacency_method'], 
                                                     maxspots_pooling=config['maxspots_pooling'], construct_adjacency_w=config['construct_adjacency_w'])

    # compute baseline transcript counts (single_base_nb_mean)
    if config["normalidx_file"] is None:
        logging.error("Please provide normalidx_file to compute baseline transcript counts!")
        sys.exit(1)
    
    normal_candidate = (adata.obs["tumor_annotation"].values == "normal")

    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    # filter out high-UMI DE genes, which may bias RDR estimates
    copy_single_X_rdr, _ = filter_de_genes_tri(pd.DataFrame(adata.layers['count'], index=adata.obs.index, columns=adata.var.index), 
                                               df_bininfo, normal_candidate, sample_list=sample_list, sample_ids=sample_ids)
    MIN_NORMAL_COUNT_PERBIN = 20
    bidx_inconfident = np.where( np.sum(copy_single_X_rdr[:, (normal_candidate==True)], axis=1) < MIN_NORMAL_COUNT_PERBIN )[0]
    rdr_normal = np.sum(copy_single_X_rdr[:, (normal_candidate==True)], axis=1)
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)
    copy_single_X_rdr[bidx_inconfident, :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
    copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
        
    # adding back RDR signal
    single_X[:,0,:] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean
    n_obs = single_X.shape[0]

    ########## Run HMRF-HMM ##########
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
        prefix = "allspots"
        if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_sm.npz").exists():
            initial_assignment = np.zeros(single_X.shape[2], dtype=int)
            for c,idx in enumerate(initial_clone_index):
                initial_assignment[idx] = c
            allres = {"num_iterations":0, "round-1_assignment":initial_assignment}
            np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_sm.npz", **allres)

        # run HMRF + HMM
        if config["tumorprop_file"] is None:
            hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="sm", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
        else:
            hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="sm", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"], tumorprop_threshold=config["tumorprop_threshold"])

        # load HMRF results and output a table adjusted log rdr per bin per clone
        res = load_hmrf_last_iteration(f"{outdir}/{prefix}_nstates{config['n_states']}_sm.npz")
        n_obs = single_X.shape[0]
        # aggregate counts into clones
        unique_assignment = np.sort(np.unique(res['new_assignment']))
        clone_index = [np.where(res['new_assignment'] == c)[0] for c in unique_assignment]
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        # adjust the scaling factor for each clone
        df_adj_log_rdr = df_bininfo[['CHR', 'START', 'END']].copy()
        for s in range(X.shape[2]):
            lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
            this_pred_cnv = res['pred_cnv'][(s*n_obs):(s*n_obs+n_obs)]
            adjusted_log_mu = np.log( np.exp(res["new_log_mu"][:,0]) / np.sum(np.exp(res["new_log_mu"][this_pred_cnv,0]) * lambd) )
            df_adj_log_rdr[f'clone_{s}_logrdr'] = adjusted_log_mu[this_pred_cnv]
        df_adj_log_rdr.to_csv(f"{outdir}/adjusted_logrdr.txt", sep="\t", index=False)

        # merge by thresholding BAF profile similarity
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
        np.savez(f"{outdir}/mergedallspots_nstates{config['n_states']}_sm.npz", **merged_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    main(args.configfile)