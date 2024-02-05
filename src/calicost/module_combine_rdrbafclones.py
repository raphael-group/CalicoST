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


def run_combine_rdrbafclones(config, foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, barcodes, single_tumor_prop, sample_ids, sample_list, adjacency_mat, smooth_mat, merged_baf_assignment):
    # foldername is the same as FOLDER_RDRBAFCLONES
    # outdir
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"

    # BAF clone info
    n_baf_clones = len(np.unique(merged_baf_assignment))

    n_obs = single_X.shape[0]

    res_combine = {"prev_assignment":np.zeros(single_X.shape[2], dtype=int)}
    offset_clone = 0
    for bafc in range(n_baf_clones):
        prefix = f"clone{bafc}"
        idx_spots = np.where(merged_baf_assignment == bafc)[0]
        if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 20: # put a minimum B allele read count on pseudobulk to split clones
            continue
        # load HMRF-HMM results of this BAF clone
        allres = dict( np.load(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", allow_pickle=True) )
        r = allres["num_iterations"] - 1
        res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
            "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
            "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
            "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
            "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
        idx_spots = np.where(barcodes.isin( allres["barcodes"] ))[0]
        if len(np.unique(res["new_assignment"])) == 1:
            n_merged_clones = 1
            c = res["new_assignment"][0]
            merged_res = copy.copy(res)
            merged_res["new_assignment"] = np.zeros(len(idx_spots), dtype=int)
            try:
                log_gamma = res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)].reshape((2*config["n_states"], n_obs, 1))
            except:
                log_gamma = res["log_gamma"][:, (c*n_obs):(c*n_obs+n_obs)].reshape((config["n_states"], n_obs, 1))
            pred_cnv = res["pred_cnv"][ (c*n_obs):(c*n_obs+n_obs) ].reshape((-1,1))
        else:
            if config["tumorprop_file"] is None:
                X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"])) ])
                tumor_prop = None
            else:
                X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"])) ], single_tumor_prop[idx_spots], threshold=config["tumorprop_threshold"])
                tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1,1)
            merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, threshold=config["np_threshold"], minlength=config["np_eventminlen"], params="smp", tumor_prop=tumor_prop, hmmclass=hmm_nophasing_v2)
            print(f"part {bafc} merging_groups: {merging_groups}")
            #
            if config["tumorprop_file"] is None:
                merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD[:,idx_spots], min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=config["min_avgumi_per_clone"]*n_obs)
            else:
                merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, single_total_bb_RD[:,idx_spots], min_spots_thresholds=config["min_spots_per_clone"], min_umicount_thresholds=config["min_avgumi_per_clone"]*n_obs, single_tumor_prop=single_tumor_prop[idx_spots], threshold=config["tumorprop_threshold"])
            print(f"part {bafc} merging after requiring minimum # spots: {merging_groups}")
            # compute posterior using the newly merged pseudobulk
            n_merged_clones = len(merging_groups)
            tmp = copy.copy(merged_res["new_assignment"])
            if config["tumorprop_file"] is None:
                X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)])
                tumor_prop = None
            else:
                X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)], single_tumor_prop[idx_spots], threshold=config["tumorprop_threshold"])
            #
            merged_res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config["n_states"], \
                    base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), np.repeat(tumor_prop, X.shape[0]).reshape(-1,1) if not tumor_prop is None else None, \
                    hmmclass=hmm_nophasing_v2, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                    fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                    is_diag=True, init_log_mu=res["new_log_mu"], init_p_binom=res["new_p_binom"], init_alphas=res["new_alphas"], init_taus=res["new_taus"], max_iter=config["max_iter"], tol=config["tol"], lambd=np.sum(base_nb_mean,axis=1)/np.sum(base_nb_mean), sample_length=np.ones(X.shape[2],dtype=int)*X.shape[0])
            merged_res["new_assignment"] = copy.copy(tmp)
            merged_res = combine_similar_states_across_clones(X, base_nb_mean, total_bb_RD, merged_res, params="smp", tumor_prop=np.repeat(tumor_prop, X.shape[0]).reshape(-1,1) if not tumor_prop is None else None, hmmclass=hmm_nophasing_v2, merge_threshold=0.1)
            log_gamma = np.stack([ merged_res["log_gamma"][:,(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ], axis=-1)
            pred_cnv = np.vstack([ merged_res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ]).T
        #
        # add to res_combine
        if len(res_combine) == 1:
            res_combine.update({"new_log_mu":np.hstack([ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([ merged_res["new_alphas"] ] * n_merged_clones), \
                "new_p_binom":np.hstack([ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([ merged_res["new_taus"] ] * n_merged_clones), \
                "log_gamma":log_gamma, "pred_cnv":pred_cnv})
        else:
            res_combine.update({"new_log_mu":np.hstack([res_combine["new_log_mu"]] + [ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([res_combine["new_alphas"]] + [ merged_res["new_alphas"] ] * n_merged_clones), \
                "new_p_binom":np.hstack([res_combine["new_p_binom"]] + [ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([res_combine["new_taus"]] + [ merged_res["new_taus"] ] * n_merged_clones), \
                "log_gamma":np.dstack([res_combine["log_gamma"], log_gamma ]), "pred_cnv":np.hstack([res_combine["pred_cnv"], pred_cnv])})
        res_combine["prev_assignment"][idx_spots] = merged_res["new_assignment"] + offset_clone
        offset_clone += n_merged_clones
    # temp: make dispersions the same across all clones
    res_combine["new_alphas"][:,:] = np.max(res_combine["new_alphas"])
    res_combine["new_taus"][:,:] = np.min(res_combine["new_taus"])
    # end temp
    n_final_clones = len(np.unique(res_combine["prev_assignment"]))

    # final re-assignment across all clones using estimated RDR + BAF after merging similar ones
    # HMRF prior based on proportions of each clone (within each sample)
    log_persample_weights = np.zeros((n_final_clones, len(sample_list)))
    for sidx in range(len(sample_list)):
        index = np.where(sample_ids == sidx)[0]
        this_persample_weight = np.bincount(res_combine["prev_assignment"][index], minlength=n_final_clones) / len(index)
        log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
        log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])
    if config["tumorprop_file"] is None:
        if config["nodepotential"] == "max":
            pred = np.vstack([ np.argmax(res_combine["log_gamma"][:,:,c], axis=0) for c in range(res_combine["log_gamma"].shape[2]) ]).T
            new_assignment, single_llf, total_llf, posterior = aggr_hmrf_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, res_combine, pred, \
                smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing_v2, return_posterior=True)
        elif config["nodepotential"] == "weighted_sum":
            new_assignment, single_llf, total_llf, posterior = hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res_combine, \
                smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing_v2, return_posterior=True)
    else:
        if config["nodepotential"] == "max":
            pred = np.vstack([ np.argmax(res_combine["log_gamma"][:,:,c], axis=0) for c in range(res_combine["log_gamma"].shape[2]) ]).T
            new_assignment, single_llf, total_llf, posterior = aggr_hmrfmix_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res_combine, pred, \
                smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing_v2, return_posterior=True)
        elif config["nodepotential"] == "weighted_sum":
            new_assignment, single_llf, total_llf, posterior = hmrfmix_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res_combine, \
                smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing_v2, return_posterior=True)
    res_combine["total_llf"] = total_llf
    res_combine["new_assignment"] = new_assignment
    # re-order clones such that normal clones are always clone 0
    res_combine, posterior = reorder_results(res_combine, posterior, single_tumor_prop)
    # save results
    np.savez(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", **res_combine)
    np.save(f"{outdir}/posterior_clone_probability.npy", posterior)


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
    run_combine_rdrbafclones(config, args.foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, coords, barcodes, single_tumor_prop, sample_ids, sample_list, adjacency_mat, smooth_mat, merged_baf_assignment)


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

    # combine RDR-BAF clones
    run_combine_rdrbafclones(config, args.foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, log_sitewise_transmat, barcodes, single_tumor_prop, sample_ids, sample_list, adjacency_mat, smooth_mat, merged_baf_assignment)