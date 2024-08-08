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

    # Assuming the B counts are calculated by the cellsnp-lite and Eagle pipeline
    # If assuming each spot contains a mixture of normal/tumor cells, the tumor proportion should be provided in the config file.
    # load data
    ## If the data is loaded for the first time: infer phasing using phase-switch HMM (hmm_NB_BB_phaseswitch.py and phasing.py) -> output initial_phase.npz, matrices in parsed_inputs folder
    ## If the data is already loaded: load the matrices from parsed_inputs folder
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)
    
    """
    Initial clustering spots using only BAF values.
    """
    # setting transcript count to 0, and baseline so that emission probability calculation will ignore them.
    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)
    single_X[:,0,:] = 0
    single_base_nb_mean[:,:] = 0
    
    # run HMRF
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
        if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz").exists():
            initial_assignment = np.zeros(single_X.shape[2], dtype=int)
            for c,idx in enumerate(initial_clone_index):
                initial_assignment[idx] = c
            allres = {"num_iterations":0, "round-1_assignment":initial_assignment}
            np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz", **allres)

        # run HMRF + HMM
        # store the results of each iteration of HMRF in a npz file outdir/prefix_nstates{config['n_states']}_sp.npz
        # if a specific iteration is computed, hmrf will directly load the results from the file
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
        # merge "similar" clones from the initial number of clones.
        # "similar" defined by Neyman Pearson statistics/ Likelihood ratios P(clone A counts | BAF parameters for clone A) / P(clone A counts | BAF parameters for clone B)
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

        # load merged results
        n_obs = single_X.shape[0]
        merged_res = dict(np.load(f"{outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", allow_pickle=True))
        merged_baf_assignment = copy.copy(merged_res["new_assignment"])
        n_baf_clones = len(np.unique(merged_baf_assignment))
        pred = np.argmax(merged_res["log_gamma"], axis=0)
        pred = np.array([ pred[(c*n_obs):(c*n_obs+n_obs)] for c in range(n_baf_clones) ])
        merged_baf_profiles = np.array([ np.where(pred[c,:] < config["n_states"], merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0], 1-merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0]) \
                                        for c in range(n_baf_clones) ])
        
        """
        Refined clustering using BAF and RDR values.
        """
        # adding RDR information
        if not config["bafonly"]:
            # Only used when assuming each spot is pure normal or tumor and if we don't know which spots are normal spots.
            # select normal spots
            if (config["normalidx_file"] is None) and (config["tumorprop_file"] is None):
                EPS_BAF = 0.05
                PERCENT_NORMAL = 40
                vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
                id_nearnormal_clone = np.argmin(np.sum( np.maximum(np.abs(merged_baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
                while True:
                    stdthreshold = np.percentile(vec_stds[merged_res["new_assignment"] == id_nearnormal_clone], PERCENT_NORMAL)
                    normal_candidate = (vec_stds < stdthreshold) & (merged_res["new_assignment"] == id_nearnormal_clone)
                    if np.sum(copy_single_X_rdr[:, (normal_candidate==True)]) > single_X.shape[0] * 200 or PERCENT_NORMAL == 100:
                        break
                    PERCENT_NORMAL += 10
                pd.Series(barcodes[normal_candidate==True].index).to_csv(f"{outdir}/normal_candidate_barcodes.txt", header=False, index=False)

            elif (not config["normalidx_file"] is None):
                # single_base_nb_mean has already been added in loading data step.
                if not config["tumorprop_file"] is None:
                    logger.warning(f"Mixed sources of information for normal spots! Using {config['normalidx_file']}")
            
            # If tumor purity is provided, we can use it to select normal spots.
            else:
                for prop_threshold in np.arange(0.05, 0.6, 0.05):
                    normal_candidate = (single_tumor_prop < prop_threshold)
                    if np.sum(copy_single_X_rdr[:, (normal_candidate==True)]) > single_X.shape[0] * 200:
                        break
            # To avoid allele-specific expression that are not relevant to CNA, filter bins where normal pseudobulk has large |BAF - 0.5|
            index_normal = np.where(normal_candidate)[0]
            lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_gene_snp = bin_selection_basedon_normal(df_gene_snp, \
                single_X, single_base_nb_mean, single_total_bb_RD, config['nu'], config['logphase_shift'], index_normal, config['geneticmap_file'])
            assert df_bininfo.shape[0] == copy_single_X_rdr.shape[0]
            df_bininfo = genesnp_to_bininfo(df_gene_snp)
            copy_single_X_rdr = copy.copy(single_X[:,0,:])

            # If a gene has way higher expression than adjacent genes, its transcript count will dominate RDR values
            # To avoid the domination, filter out high-UMI DE genes, which may bias RDR estimates
            # Assume the remaining genes will still carry the CNA info.
            copy_single_X_rdr, _ = filter_de_genes_tri(exp_counts, df_bininfo, normal_candidate, sample_list=sample_list, sample_ids=sample_ids)
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

            # save binned data
            np.savez(f"{outdir}/binned_data.npz", lengths=lengths, single_X=single_X, single_base_nb_mean=single_base_nb_mean, single_total_bb_RD=single_total_bb_RD, log_sitewise_transmat=log_sitewise_transmat, single_tumor_prop=(None if config["tumorprop_file"] is None else single_tumor_prop))

            # run HMRF on each clone individually to further split BAF clone by RDR+BAF signal
            for bafc in range(n_baf_clones):
                prefix = f"clone{bafc}"
                idx_spots = np.where(merged_baf_assignment == bafc)[0]
                if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 20: # put a minimum B allele read count on pseudobulk to split clones
                    continue
                # initialize clone
                # write the initialization in a npz file outdir/prefix_nstates{config['n_states']}_smp.npz
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

            ##### combine results across clones #####
            res_combine = {"prev_assignment":np.zeros(single_X.shape[2], dtype=int)}
            offset_clone = 0
            for bafc in range(n_baf_clones):
                prefix = f"clone{bafc}"
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
            # per-sample weights across clones
            log_persample_weights = np.zeros((n_final_clones, len(sample_list)))
            for sidx in range(len(sample_list)):
                index = np.where(sample_ids == sidx)[0]
                this_persample_weight = np.bincount(res_combine["prev_assignment"][index], minlength=n_final_clones) / len(index)
                log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
                log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])
            # final re-assignment across all clones using estimated RDR + BAF
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
            
            ##### infer integer copy #####
            res_combine = dict(np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))
            final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
            nonempty_clone_ids = copy.copy(final_clone_ids)
            # add clone 0 as normal clone if it doesn't appear in final_clone_ids
            if not (0 in final_clone_ids):
                final_clone_ids = np.append(0, final_clone_ids)
            # chr position
            medfix = ["", "_diploid", "_triploid", "_tetraploid"]
            for o,max_medploidy in enumerate([None, 2, 3, 4]):
                # A/B copy number per bin
                allele_specific_copy = []
                # A/B copy number per state
                state_cnv = []

                df_genelevel_cnv = None
                if config["tumorprop_file"] is None:
                    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids])
                else:
                    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids], single_tumor_prop, threshold=config["tumorprop_threshold"])

                for s, cid in enumerate(final_clone_ids):
                    if np.sum(base_nb_mean[:,s]) == 0:
                        continue
                    # adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1
                    lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
                    this_pred_cnv = res_combine["pred_cnv"][:,s]
                    adjusted_log_mu = np.log( np.exp(res_combine["new_log_mu"][:,s]) / np.sum(np.exp(res_combine["new_log_mu"][this_pred_cnv,s]) * lambd) )
                    if not max_medploidy is None:
                        best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, max_medploidy=max_medploidy)
                    else:
                        try:
                            best_integer_copies, _ = hill_climbing_integer_copynumber_fixdiploid(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, nonbalance_bafdist=config["nonbalance_bafdist"], nondiploid_rdrdist=config["nondiploid_rdrdist"])
                        except:
                            try:
                                best_integer_copies, _ = hill_climbing_integer_copynumber_fixdiploid(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, nonbalance_bafdist=config["nonbalance_bafdist"], nondiploid_rdrdist=config["nondiploid_rdrdist"], min_prop_threshold=0.02)
                            except:
                                finding_distate_failed = True
                                continue

                    print(f"max med ploidy = {max_medploidy}, clone {s}, integer copy inference loss = {_}")
                    #
                    allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 0].reshape(1,-1), index=[f"clone{cid} A"], columns=np.arange(n_obs) ) )
                    allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 1].reshape(1,-1), index=[f"clone{cid} B"], columns=np.arange(n_obs) ) )
                    #
                    state_cnv.append( pd.DataFrame( res_combine["new_log_mu"][:,s].reshape(-1,1), columns=[f"clone{cid} logmu"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( res_combine["new_p_binom"][:,s].reshape(-1,1), columns=[f"clone{cid} p"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( best_integer_copies[:,0].reshape(-1,1), columns=[f"clone{cid} A"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( best_integer_copies[:,1].reshape(-1,1), columns=[f"clone{cid} B"], index=np.arange(config['n_states']) ) )
                    #
                    # tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[res_combine["pred_cnv"][:,s], 0], best_integer_copies[res_combine["pred_cnv"][:,s], 1], x_gene_list)
                    # tmpdf.columns = [f"clone{s} A", f"clone{s} B"]
                    bin_Acopy_mappers = {i:x for i,x in enumerate(best_integer_copies[res_combine["pred_cnv"][:,s], 0])}
                    bin_Bcopy_mappers = {i:x for i,x in enumerate(best_integer_copies[res_combine["pred_cnv"][:,s], 1])}
                    tmpdf = pd.DataFrame({"gene":df_gene_snp[df_gene_snp.is_interval].gene, f"clone{s} A":df_gene_snp[df_gene_snp.is_interval]['bin_id'].map(bin_Acopy_mappers), \
                        f"clone{s} B":df_gene_snp[df_gene_snp.is_interval]['bin_id'].map(bin_Bcopy_mappers)}).set_index('gene')
                    if df_genelevel_cnv is None:
                        df_genelevel_cnv = copy.copy( tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int) )
                    else:
                        df_genelevel_cnv = df_genelevel_cnv.join( tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int) )
                if len(state_cnv) == 0:
                    continue
                # output gene-level copy number
                df_genelevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_genelevel.tsv", header=True, index=True, sep="\t")
                # output segment-level copy number
                allele_specific_copy = pd.concat(allele_specific_copy)
                df_seglevel_cnv = pd.DataFrame({"CHR":df_bininfo.CHR.values, "START":df_bininfo.START.values, "END":df_bininfo.END.values })
                df_seglevel_cnv = df_seglevel_cnv.join( allele_specific_copy.T )
                df_seglevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_seglevel.tsv", header=True, index=False, sep="\t")
                # output per-state copy number
                state_cnv = functools.reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True, how='inner'), state_cnv)
                state_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_perstate.tsv", header=True, index=False, sep="\t")
                # #
                # # posterior using integer-copy numbers
                # log_persample_weights = np.zeros((len(nonempty_clone_ids), len(sample_list)))
                # for sidx in range(len(sample_list)):
                #     index = np.where(sample_ids == sidx)[0]
                #     this_persample_weight = np.array([ np.sum(res_combine["new_assignment"][index] == cid) for cid in nonempty_clone_ids]) / len(index)
                #     log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
                #     log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])
                # pred = np.vstack([ np.argmax(res_combine["log_gamma"][:,:,cid], axis=0) for cid in nonempty_clone_ids ]).T
                # df_posterior = clonelabel_posterior_withinteger(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, state_cnv, res_combine, pred, \
                #     smooth_mat, adjacency_mat, res_combine["new_assignment"], sample_ids, base_nb_mean, log_persample_weights, config["spatial_weight"], hmmclass=hmm_nophasing_v2)
                # df_posterior.to_pickle(f"{outdir}/posterior{medfix[o]}.pkl")
            
            ##### output clone label #####
            df_clone_label = pd.DataFrame({"clone_label":res_combine["new_assignment"]}, index=barcodes)
            if not config["tumorprop_file"] is None:
                df_clone_label["tumor_proportion"] = single_tumor_prop
            df_clone_label.to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")

            ##### plotting #####
            # make a directory for plots
            p = subprocess.Popen(f"mkdir -p {outdir}/plots", shell=True)
            out, err = p.communicate()

            # plot RDR and BAF
            cn_file = f"{outdir}/cnv_diploid_seglevel.tsv"
            fig = plot_rdr_baf(configuration_file, r_hmrf_initialization, cn_file, clone_ids=None, remove_xticks=True, rdr_ylim=5, chrtext_shift=-0.3, base_height=3.2, pointsize=30, palette="tab10")
            fig.savefig(f"{outdir}/plots/rdr_baf_defaultcolor.pdf", transparent=True, bbox_inches="tight")
            # plot allele-specific copy number
            for o,max_medploidy in enumerate([None, 2, 3, 4]):
                cn_file = f"{outdir}/cnv{medfix[o]}_seglevel.tsv"
                if not Path(cn_file).exists():
                    continue
                df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
                df_cnv = expand_df_cnv(df_cnv)
                df_cnv = df_cnv[~df_cnv.iloc[:,-1].isnull()]
                fig, axes = plt.subplots(1, 1, figsize=(15, 0.9*len(final_clone_ids) + 0.6), dpi=200, facecolor="white")
                axes = plot_acn_from_df_anotherscheme(df_cnv, axes, chrbar_pos='top', chrbar_thickness=0.3, add_legend=False, remove_xticks=True)
                fig.tight_layout()
                fig.savefig(f"{outdir}/plots/acn_genome{medfix[o]}.pdf", transparent=True, bbox_inches="tight")
                # additionally plot the allele-specific copy number per region
                if not config["supervision_clone_file"] is None:
                    fig, axes = plt.subplots(1, 1, figsize=(15, 0.6*len(unique_clone_ids) + 0.4), dpi=200, facecolor="white")
                    merged_df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
                    df_cnv = merged_df_cnv[["CHR", "START", "END"]]
                    df_cnv = df_cnv.join( pd.DataFrame({f"clone{x} A":merged_df_cnv[f"clone{res_combine['new_assignment'][i]} A"] for i,x in enumerate(unique_clone_ids)}) )
                    df_cnv = df_cnv.join( pd.DataFrame({f"clone{x} B":merged_df_cnv[f"clone{res_combine['new_assignment'][i]} B"] for i,x in enumerate(unique_clone_ids)}) )
                    df_cnv = expand_df_cnv(df_cnv)
                    clone_ids = np.concatenate([ unique_clone_ids[res_combine["new_assignment"]==c].astype(str) for c in final_clone_ids ])
                    axes = plot_acn_from_df(df_cnv, axes, clone_ids=clone_ids, clone_names=[f"region {x}" for x in clone_ids], add_chrbar=True, add_arrow=False, chrbar_thickness=0.4/(0.6*len(unique_clone_ids) + 0.4), add_legend=True, remove_xticks=True)
                    fig.tight_layout()
                    fig.savefig(f"{outdir}/plots/acn_genome{medfix[o]}_per_region.pdf", transparent=True, bbox_inches="tight")
            # plot clones in space
            if not config["supervision_clone_file"] is None:
                before_assignments = pd.Series([None] * before_coords.shape[0])
                for i,c in enumerate(unique_clone_ids):
                    before_assignments.iloc[before_df_clones.clone_id.isin([c])] = f"clone {res_combine['new_assignment'][i]}"
                fig = plot_clones_in_space(before_coords, before_assignments, sample_list, before_sample_ids, palette="Set2", labels=unique_clone_ids, label_coords=coords, label_sample_ids=sample_ids)
                fig.savefig(f"{outdir}/plots/clone_spatial.pdf", transparent=True, bbox_inches="tight")
            else:
                assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
                fig = plot_individual_spots_in_space(coords, assignment, single_tumor_prop, sample_list=sample_list, sample_ids=sample_ids)
                fig.savefig(f"{outdir}/plots/clone_spatial.pdf", transparent=True, bbox_inches="tight")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    main(args.configfile)