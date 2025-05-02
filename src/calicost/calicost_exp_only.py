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
from calicost.utils_hmrf import *
from calicost.utils_hmm import *
# from calicost.hmrf_exp_only import *
from calicost.hmrf import *
from calicost.utils_IO import *
from calicost.parse_input import *
from calicost.utils_plotting import *


def convert_calicost_3_logrdr_to_states(logRDR, max_logRDR=1.9, min_logRDR=-1.9):
    """
    Convert logRDR to copy number states. Median of logRDR is treated as normal. Above/below are treated as amplification/deletion.

    Attributes
    ----------
    logRDR : np.array, shape (n_bins)
        The log RDR (account for scaling factor) for the clone.
    
    Returns
    -------
    coarse_states : np.array, shape (n_bins,)
        The copy number states for each bin.
    """
    unique_logRDR = np.unique(logRDR)
    sorted_logRDR = np.sort(logRDR)
    # log RDR of neu state is the median of all logRDR
    neu_state_logrdr = sorted_logRDR[len(sorted_logRDR) // 2]
    # # log RDR of amp state is the highest log RDR above neu_state_logrdr (if not exceeding max_logRDR)
    # amp_state_logrdr = sorted_logRDR[(sorted_logRDR > neu_state_logrdr) & (sorted_logRDR < max_logRDR)][-1] if len(sorted_logRDR[(sorted_logRDR > neu_state_logrdr) & (sorted_logRDR < max_logRDR)]) > 0 else None
    # log RDR of del state is the lowest log RDR below neu_state_logrdr (if not exceeding min_logRDR)
    del_state_logrdr = sorted_logRDR[(sorted_logRDR < neu_state_logrdr) & (sorted_logRDR > min_logRDR)][0] if len(sorted_logRDR[(sorted_logRDR < neu_state_logrdr) & (sorted_logRDR > min_logRDR)]) > 0 else None

    EPS = 1e-4

    coarse_states = np.array(["neutral"] * logRDR.shape[0])
    # if not amp_state_logrdr is None:
    #     coarse_states[logRDR > amp_state_logrdr - EPS] = "amp"
    coarse_states[logRDR > neu_state_logrdr] = 'amp'
    if not del_state_logrdr is None:
        coarse_states[logRDR < del_state_logrdr + EPS] = "del"
    coarse_states[coarse_states == "neutral"] = "neu"
    return coarse_states


def get_shared_intervals(cn_profile):
    '''
    Takes in copy numbers, output a segmentation of genome such that all clones are in the same CN state within each segment.

    anc_profile : array, (n_obs, 2*n_clones)
        Copy numbers for each genomic bin (obs) across all clones.
    '''
    intervals = []
    seg_acn = []
    s = 0
    while s < cn_profile.shape[0]:
        t = np.where( ~np.all(cn_profile[s:,] == cn_profile[s,:], axis=1) )[0]
        if len(t) == 0:
            intervals.append( (s, cn_profile.shape[0])  )
            seg_acn.append( cn_profile[s,:] )
            s = cn_profile.shape[0]
        else:
            t = t[0]
            intervals.append( (s,s+t) )
            seg_acn.append( cn_profile[s,:] )
            s = s+t
    return intervals, seg_acn


def collapse_cnv_profiles(df_adj_log_rdr):
    """
    Combine adjacent bins with identical copy number states into a single row in df_adj_log_rdr dataframe.

    Attributes
    ----------
    df_adj_log_rdr : pd.DataFrame
        The dataframe contains CHR, START, END, and copy number states for each clone (columns). Assume the fourth and later columns are copy number states.
    
    Returns
    -------
    collapse_df_adj_log_rdr : pd.DataFrame
        The dataframe contains CHR, START, END, and copy number states for each clone (columns). Assume the fourth and later columns are copy number states.
    """
    col_to_combine = np.append(0, np.arange(3, df_adj_log_rdr.shape[1])) # CHR and copy number state columns. Adding CHR column to make sure not to combine bins from different chromosomes
    intervals, seg_acn = get_shared_intervals(df_adj_log_rdr.iloc[:,col_to_combine].values)

    collapse_df_adj_log_rdr = []
    for i, p in enumerate(intervals):
        s = p[0]
        t = p[1]
        this_df = df_adj_log_rdr.iloc[s:(s+1), :].copy()
        this_df['END'] = df_adj_log_rdr.END.values[t-1]
        collapse_df_adj_log_rdr.append(this_df)

    collapse_df_adj_log_rdr = pd.concat(collapse_df_adj_log_rdr, ignore_index=True)
    return collapse_df_adj_log_rdr


def main(config):
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    ########## Proprocess input data ##########
    if "input_filelist" in config:
        adata, across_slice_adjacency_mat = load_joint_data_exp_only(config["input_filelist"], config["alignment_files"], None, None, config["normalidx_file"], config['min_umi_perspot'], config['min_percent_expressed_spots'])
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
        adata = load_data_exp_only(config["spaceranger_dir"], None, None, config["normalidx_file"], config['min_umi_perspot'], config['min_percent_expressed_spots'])
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
            adata, np.empty(0), np.empty(0), np.empty(0), nu=1.0, logphase_shift=-2, geneticmap_file=None)
    
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
        Path(f'{outdir}').mkdir(parents=True, exist_ok=True)
        Path(f'{outdir}/plots').mkdir(parents=True, exist_ok=True)
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
            # pseudobulk
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index)
            # initialize HMM parameters by percentiles of linspace(30, 70, n_states)
            init_log_mu = initialization_log_mu_percentile(config["n_states"], X[:,0,:].flatten("F").reshape(-1,1), base_nb_mean.flatten("F").reshape(-1,1), random_state=0)
            init_p_binom = np.linspace(0.01, 0.99, config["n_states"]).reshape(-1,1)

            hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="sm", t=config["t"], random_state=0, \
                fix_NB_dispersion=False, shared_NB_dispersion=True, \
                fix_BB_dispersion=False, shared_BB_dispersion=True, \
                init_log_mu=init_log_mu, init_p_binom=init_p_binom, \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

        else:
            # pseudobulk
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, initial_clone_index, single_tumor_prop, threshold=config["tumorprop_threshold"])
            # initialize HMM parameters by percentiles of linspace(30, 70, n_states)
            init_log_mu = initialization_log_mu_percentile(config["n_states"], X[:,0,:].flatten("F").reshape(-1,1), base_nb_mean.flatten("F").reshape(-1,1), random_state=0)
            init_p_binom = np.linspace(0.01, 0.99, config["n_states"]).reshape(-1,1)

            hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing_v2, params="sm", t=config["t"], random_state=0, \
                fix_NB_dispersion=False, shared_NB_dispersion=True, \
                fix_BB_dispersion=False, shared_BB_dispersion=True, \
                init_log_mu=init_log_mu, init_p_binom=None, \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"], tumorprop_threshold=config["tumorprop_threshold"])

        # load HMRF results and output a table adjusted log rdr per bin per clone
        res = load_hmrf_last_iteration(f"{outdir}/{prefix}_nstates{config['n_states']}_sm.npz")
        n_obs = single_X.shape[0]
        res = reorder_results_exponly(res, n_obs)
        
        # aggregate counts into clones
        unique_assignment = np.sort(np.unique(res['new_assignment']))
        clone_index = [np.where(res['new_assignment'] == c)[0] for c in unique_assignment]
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))])
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))], single_tumor_prop, threshold=config["tumorprop_threshold"])

        # adjust the scaling factor for each clone
        df_adj_log_rdr = df_bininfo[['CHR', 'START', 'END']].copy()
        for s in range(X.shape[2]):
            lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
            this_pred_cnv = res['pred_cnv'][(s*n_obs):(s*n_obs+n_obs)]
            adjusted_log_mu = np.log( np.exp(res["new_log_mu"][:,0]) / np.sum(np.exp(res["new_log_mu"][this_pred_cnv,0]) * lambd) )
            df_adj_log_rdr[f'clone{s} logrdr'] = adjusted_log_mu[this_pred_cnv]
        df_adj_log_rdr.to_csv(f"{outdir}/adjusted_logrdr_beforemerging.tsv", sep="\t", index=False)

        # convert to copy number states (neu, amp, del)
        for s in range(X.shape[2]):
            df_adj_log_rdr[f'clone{s} cnv'] = convert_calicost_3_logrdr_to_states(df_adj_log_rdr[f'clone{s} logrdr'].values)
        df_adj_log_rdr = df_adj_log_rdr[ ['CHR', 'START', 'END']  + [f'clone{s} cnv' for s in range(X.shape[2])] ]
        # plot total copy number along genome
        fig, ax = plt.subplots(1, 1, figsize=(15, 1+0.5*(X.shape[2])))
        plot_total_cn(df_adj_log_rdr.rename(columns={f'clone{s} cnv':f'clone {s}' for s in range(X.shape[2])}), ax, palette_mode=3)
        fig.savefig(f"{outdir}/plots/total_cn_beforemerging.pdf", dpi=300, bbox_inches='tight', transparent=True)

        # save copy number states to file
        df_adj_log_rdr = collapse_cnv_profiles(df_adj_log_rdr)
        df_adj_log_rdr.to_csv(f"{outdir}/cnv_beforemerging.tsv", sep="\t", index=False)

        # output clone labels before merging as a tsv file
        pd.DataFrame({'clone_label':res['new_assignment']}, index=adata.obs.index).to_csv(f"{outdir}/clone_labels_beforemerging.tsv", sep="\t", index=True, header=True)

        # plot clones in space and cnv states along the genome
        assignment = pd.Series([f"clone {x}" for x in res['new_assignment'] ])
        fig = plot_individual_spots_in_space(coords, assignment, single_tumor_prop=None, sample_list=sample_list, sample_ids=sample_ids, base_height=3, palette='Set2')
        fig.savefig(f"{outdir}/plots/clone_spatial_beforemerging.pdf", dpi=300, bbox_inches='tight', transparent=True)

        # plot RDR along the genome for each clone
        df_rdr = df_bininfo[['CHR', 'START', 'END']].copy()
        for s in range(X.shape[2]):
            this_pred_cnv = res['pred_cnv'][(s*n_obs):(s*n_obs+n_obs)]
            df_rdr[f'clone{s} RD'] = X[:,0,s]/base_nb_mean[:,s]
            df_rdr[f'clone{s} state'] = this_pred_cnv
        fig, axes = plot_rdr_exponly_from_df(df_rdr, res['new_log_mu'], rdr_ylim=4, remove_xticks=True, palette='tab10')
        fig.savefig(f"{outdir}/plots/rdr_beforemerging.pdf", dpi=300, bbox_inches='tight', transparent=True)

        ########### Post-process clone merging ##########
        # merge by thresholding BAF profile similarity
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))])
            tumor_prop = None
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))], single_tumor_prop, threshold=config["tumorprop_threshold"])
            tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1,1)

        merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, threshold=config['clone_similarity_threshold'], minlength=10, params="sm", tumor_prop=None, hmmclass=hmm_nophasing_v2)
        print(f"BAF clone merging after comparing similarity: {merging_groups}")
        #
        if config["tumorprop_file"] is None:
            merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD, min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=0)
        else:
            merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD, min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=0, single_tumor_prop=single_tumor_prop, threshold=config["tumorprop_threshold"])
        print(f"BAF clone merging after requiring minimum # spots: {merging_groups}")

        # save final HMM results
        final_res = reorder_results_exponly(merged_res, n_obs)
        np.savez(f"{outdir}/mergedallspots_nstates{config['n_states']}_sm.npz", **final_res)

        # clones in a merging group have different HMM states, which HMM states to use for the merged clone?
        # dirctly infer the HMM state again using forward-backward algorithm on the merged clone
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(merged_res["new_assignment"]==c)[0] for c in np.sort(np.unique(merged_res["new_assignment"]))])
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(merged_res["new_assignment"]==c)[0] for c in np.sort(np.unique(merged_res["new_assignment"]))], single_tumor_prop, threshold=config["tumorprop_threshold"])
        # # HMM
        # final_res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config['n_states'], \
        #                     base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), \
        #                     hmmclass=hmm_nophasing_v2, params='sm', t=config['t'], random_state=0, \
        #                     fix_NB_dispersion=False, shared_NB_dispersion=False, fix_BB_dispersion=True, shared_BB_dispersion=True, \
        #                     is_diag=True, init_log_mu=merged_res['new_log_mu'], init_p_binom=merged_res['new_p_binom'], init_alphas=None, init_taus=None, max_iter=1, tol=config['tol'])
        # final_res['new_log_mu'] = res['new_log_mu']
        # final_res['new_alphas'] = res['new_alphas']
        # final_res['new_assignment'] = merged_res['new_assignment']
        # final_res = reorder_results_exponly(final_res, n_obs)
        # np.savez(f"{outdir}/mergedallspots_nstates{config['n_states']}_sm.npz", **final_res)

        # adjust the scaling factor for each clone
        df_adj_log_rdr = df_bininfo[['CHR', 'START', 'END']].copy()
        for s in range(X.shape[2]):
            lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
            this_pred_cnv = final_res['pred_cnv'][(s*n_obs):(s*n_obs+n_obs)]
            adjusted_log_mu = np.log( np.exp(final_res["new_log_mu"][:,0]) / np.sum(np.exp(final_res["new_log_mu"][this_pred_cnv,0]) * lambd) )
            df_adj_log_rdr[f'clone{s} logrdr'] = adjusted_log_mu[this_pred_cnv]
        df_adj_log_rdr.to_csv(f"{outdir}/adjusted_logrdr.tsv", sep="\t", index=False)

        # convert to copy number states (neu, amp, del)
        for s in range(X.shape[2]):
            df_adj_log_rdr[f'clone{s} cnv'] = convert_calicost_3_logrdr_to_states(df_adj_log_rdr[f'clone{s} logrdr'].values)
        df_adj_log_rdr = df_adj_log_rdr[ ['CHR', 'START', 'END']  + [f'clone{s} cnv' for s in range(X.shape[2])] ]
        # plot total copy number along genome
        fig, ax = plt.subplots(1, 1, figsize=(15, 1+0.5*(X.shape[2])))
        plot_total_cn(df_adj_log_rdr.rename(columns={f'clone{s} cnv':f'clone {s}' for s in range(X.shape[2])}), ax, palette_mode=3)
        fig.savefig(f"{outdir}/plots/total_cn.pdf", dpi=300, bbox_inches='tight', transparent=True)

        # save copy number states to file
        df_adj_log_rdr = collapse_cnv_profiles(df_adj_log_rdr)
        df_adj_log_rdr.to_csv(f"{outdir}/cnv.tsv", sep="\t", index=False)

        # output clone labels as a tsv file
        pd.DataFrame({'clone_label':final_res['new_assignment']}, index=adata.obs.index).to_csv(f"{outdir}/clone_labels.tsv", sep="\t", index=True, header=True)

        # plot clones in space and cnv states along the genome
        assignment = pd.Series([f"clone {x}" for x in final_res['new_assignment'] ])
        fig = plot_individual_spots_in_space(coords, assignment, single_tumor_prop=None, sample_list=sample_list, sample_ids=sample_ids, base_height=3, palette='Set2')
        fig.savefig(f"{outdir}/plots/clone_spatial.pdf", dpi=300, bbox_inches='tight', transparent=True)

        # plot RDR along the genome for each clone
        df_rdr = df_bininfo[['CHR', 'START', 'END']].copy()
        for s in range(X.shape[2]):
            this_pred_cnv = final_res['pred_cnv'][(s*n_obs):(s*n_obs+n_obs)]
            df_rdr[f'clone{s} RD'] = X[:,0,s]/base_nb_mean[:,s]
            df_rdr[f'clone{s} state'] = this_pred_cnv
        fig, axes = plot_rdr_exponly_from_df(df_rdr, final_res['new_log_mu'], rdr_ylim=4, remove_xticks=True, palette='tab10')
        fig.savefig(f"{outdir}/plots/rdr.pdf", dpi=300, bbox_inches='tight', transparent=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters to run expression-only CalicoST', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    required_group = parser.add_argument_group("Required arguments")
    # input_filelist and spaceranger_dir are mutually exclusive, add them as mutually exclusive group
    group = required_group.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_filelist', type=str, help='Input file list')
    group.add_argument('--spaceranger_dir', type=str, help='spaceranger output directory')
    required_group.add_argument('--normalidx_file', type=str, required=True, help='Normal index file')
    required_group.add_argument('--hgtable_file', type=str, required=True, help='Human genome table file')
    required_group.add_argument('--output_dir', type=str, required=True, help='Output directory')

    # optional arguments
    # Clone inference
    clone_infer_group = parser.add_argument_group("Options for clone inference (by HMRF)")
    clone_infer_group.add_argument('--n_clones', type=int, default=8, help='Number of clones')
    clone_infer_group.add_argument('--random_state', type=int, default=0, help='Random state for initialization')
    clone_infer_group.add_argument('--spatial_weight', type=float, default=1.0, help='Weight of the spatial coherence in HMRF')
    clone_infer_group.add_argument('--max_iter_outer', type=int, default=20, help='Maximum number of clone inference (HMRF) iterations')
    clone_infer_group.add_argument('--nodepotential', type=str, default="weighted_sum", choices=["max", "weighted_sum"], help='Node potential in HMRF, "max" evaluates the probability at each node/spot using the most probable HMM path (viterbi decoding), "weighted_sum" evaluates the probability using the probability of multiple paths (forward-backward)')

    # CNA inference
    cna_infer_group = parser.add_argument_group("Options for CNA inference (by HMM)")
    cna_infer_group.add_argument('--HMM_n_states', type=int, default=3, help='Number of states')
    cna_infer_group.add_argument('--HMM_max_iter', type=int, default=30, help='Maximum number of HMM iterations')
    cna_infer_group.add_argument('--HMM_self_transition_prob', type=float, default=1-1e-4, help='Self transition probability for HMM')
    cna_infer_group.add_argument('--HMM_tol', type=float, default=0.0001, help='Tolerance for HMM convergence (Baum-Welch)')

    # preprocessing, spot/gene filtering
    preprocessing_group = parser.add_argument_group("Options for preprocessing, spot/gene filtering")
    preprocessing_group.add_argument('--min_umi_perspot', type=int, default=50, help='Minimum umi per spot')
    preprocessing_group.add_argument('--min_percent_expressed_spots', type=float, default=0.005, help='Minimum percent expressed spots of each gene to be included')
    preprocessing_group.add_argument('--min_umi_binning', type=int, default=600, help='Minimum umi per bin for binning along the genome')
    preprocessing_group.add_argument('--maxspots_pooling', type=int, default=7, help='Maximum number of adjacent spots to pool together')
    preprocessing_group.add_argument('--construct_adjacency_method', type=str, choices=['hexagon', 'KNN'], default="hexagon", help='Method to construct spatial adjacency matrix')
    preprocessing_group.add_argument('--construct_adjacency_spatial_weight', type=float, default=1.0, help='Weight of spatial adjacency (vs expression similarity) in the entire adjacency matrix')

    # additional information (3D alignment, tumor purity)
    additional_info_group = parser.add_argument_group("Options for additional information (3D alignment, tumor purity)")
    additional_info_group.add_argument('--alignment_files', type=str, default=None, help='Alignment files if there are multiple slices. The alignment files should be separated by "," without any spaces.')
    additional_info_group.add_argument('--tumorprop_file', type=str, default=None, help='Tumor proportion file')
    additional_info_group.add_argument('--tumorprop_threshold', type=float, default=0.5, help='Tumor proportion threshold for merging clones')

    # post-processing (clone merging based on similarity and clone size)
    post_process_group = parser.add_argument_group("Options for post-processing (clone merging based on similarity and clone size)")
    post_process_group.add_argument('--clone_similarity_threshold', type=float, default=0.3, help='Threshold to merrge based on their CNA similarity')
    post_process_group.add_argument('--min_spots_perclone', type=int, default=20, help='Minimum number of spots per clone')

    args = parser.parse_args()

    config = {
        "input_filelist": args.input_filelist,
        "spaceranger_dir": args.spaceranger_dir,
        "normalidx_file": args.normalidx_file,
        "hgtable_file": args.hgtable_file,
        "output_dir": args.output_dir,

        # Clone inference
        "n_clones": args.n_clones,
        "num_hmrf_initialization_start": args.random_state,
        "num_hmrf_initialization_end": args.random_state + 1,
        "spatial_weight": args.spatial_weight,
        "max_iter_outer": args.max_iter_outer,
        "nodepotential": args.nodepotential,

        # CNA inference
        "n_states": args.HMM_n_states,
        "max_iter": args.HMM_max_iter,
        "t": args.HMM_self_transition_prob,
        "tol": args.HMM_tol,

        # preprocessing, spot/gene filtering
        "min_umi_perspot": args.min_umi_perspot,
        "min_percent_expressed_spots": args.min_percent_expressed_spots,
        "secondary_min_umi": args.min_umi_binning,
        "maxspots_pooling": args.maxspots_pooling,
        "construct_adjacency_method": args.construct_adjacency_method,
        "construct_adjacency_w": args.construct_adjacency_spatial_weight,

        # additional information (3D alignment, tumor purity)
        "alignment_files": [] if args.alignment_files is None else args.alignment_files.split(','),
        "tumorprop_file": args.tumorprop_file,
        "tumorprop_threshold": args.tumorprop_threshold,

        # post-processing (clone merging based on similarity and clone size)
        "clone_similarity_threshold": args.clone_similarity_threshold,
        "min_spots_per_clone": args.min_spots_perclone,
    }

    # remove either args.input_filelist or args.spaceranger_dir from config dictionary, depending on which is None
    if args.input_filelist is None:
        del config["input_filelist"]
    if args.spaceranger_dir is None:
        del config["spaceranger_dir"]

    main(config)
