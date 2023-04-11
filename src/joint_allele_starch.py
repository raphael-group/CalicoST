import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import anndata
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
import copy
from pathlib import Path
import functools
import subprocess
from hmm_NB_BB_phaseswitch import *
from utils_distribution_fitting import *
from utils_hmrf import *
from hmrf import *
from phasing import *
from utils_IO import *
from find_integer_copynumber import *

import mkl
mkl.set_num_threads(1)


def read_joint_configuration_file(filename):
    ##### [Default settings] #####
    config = {
        "input_filelist" : None,
        "snp_dir" : None,
        "output_dir" : None,
        # supporting files and preprocessing arguments
        "hgtable_file" : None,
        "normalidx_file" : None,
        "tumorprop_file" : None,
        "alignment_files" : [],
        "filtergenelist_file" : None,
        "filterregion_file" : None,
        "binsize" : 1,
        "rdrbinsize" : 1,
        # "secondbinning_min_umi" : 500,
        "max_nbins" : 1200,
        "avg_umi_perbinspot" : 1.5,
        "bafonly" : True,
        # phase switch probability
        "nu" : 1,
        "logphase_shift" : 1,
        "npart_phasing" : 2,
        # HMRF configurations
        "n_clones" : None,
        "min_spots_per_clone" : 100,
        "maxspots_pooling" : 7,
        "tumorprop_threshold" : 0.5, 
        "max_iter_outer" : 20,
        "nodepotential" : "max", # max or weighted_sum
        "initialization_method" : "rectangle", # rectangle or datadrive
        "num_hmrf_initialization_start" : 0, 
        "num_hmrf_initialization_end" : 10,
        # HMM configurations
        "n_states" : None,
        "params" : None,
        "t" : None,
        "fix_NB_dispersion" : False,
        "shared_NB_dispersion" : True,
        "fix_BB_dispersion" : False,
        "shared_BB_dispersion" : True,
        "max_iter" : 30,
        "tol" : 1e-3,
        "spatial_weight" : 2.0,
        "gmm_random_state" : 0,
        "np_threshold" : 2.0,
        "np_eventminlen" : 10,
    }

    argument_type = {
        "input_filelist" : "str",
        "snp_dir" : "str",
        "output_dir" : "str",
        # supporting files and preprocessing arguments
        "hgtable_file" : "str",
        "normalidx_file" : "str",
        "tumorprop_file" : "str",
        "alignment_files" : "list_str",
        "filtergenelist_file" : "str",
        "filterregion_file" : "str",
        "binsize" : "int",
        "rdrbinsize" : "int",
        # "secondbinning_min_umi" : "int",
        "max_nbins" : "int",
        "avg_umi_perbinspot" : "float",
        "bafonly" : "bool",
        # phase switch probability
        "nu" : "float",
        "logphase_shift" : "float",
        "npart_phasing" : "int",
        # HMRF configurations
        "n_clones" : "int",
        "min_spots_per_clone" : "int",
        "maxspots_pooling" : "int",
        "tumorprop_threshold" : "float", 
        "max_iter_outer" : "int",
        "nodepotential" : "str",
        "initialization_method" : "str",
        "num_hmrf_initialization_start" : "int", 
        "num_hmrf_initialization_end" : "int",
        # HMM configurations
        "n_states" : "int",
        "params" : "str",
        "t" : "eval",
        "fix_NB_dispersion" : "bool",
        "shared_NB_dispersion" : "bool",
        "fix_BB_dispersion" : "bool",
        "shared_BB_dispersion" : "bool",
        "max_iter" : "int",
        "tol" : "float",
        "spatial_weight" : "float",
        "gmm_random_state" : "int",
        "np_threshold" : "float",
        "np_eventminlen" : "int"
    }

    ##### [ read configuration file to update settings ] #####
    with open(filename, 'r') as fp:
        for line in fp:
            if line.strip() == "" or line[0] == "#":
                continue
            strs = [x.strip() for x in line.strip().split(":") if x != ""]
            assert strs[0] in config.keys(), f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
            if len(strs) == 1:
                config[strs[0]] = []
            elif strs[1].upper() == "NONE":
                config[strs[0]] = None
            elif argument_type[strs[0]] == "str":
                config[strs[0]] = strs[1]
            elif argument_type[strs[0]] == "int":
                config[strs[0]] = int(strs[1])
            elif argument_type[strs[0]] == "float":
                config[strs[0]] = float(strs[1])
            elif argument_type[strs[0]] == "eval":
                config[strs[0]] = eval(strs[1])
            elif argument_type[strs[0]] == "bool":
                config[strs[0]] = (strs[1].upper() == "TRUE")
            elif argument_type[strs[0]] == "list_str":
                config[strs[0]] = strs[1].split(" ")
    # assertions
    assert not config["input_filelist"] is None, "No input file list!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    return config


def main(configuration_file):
    config = read_joint_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, across_slice_adjacency_mat = load_joint_data(config["input_filelist"], config["snp_dir"], config["alignment_files"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"])
    coords = adata.obsm["X_pos"]
    sample_list = [adata.obs["sample"][0]]
    for i in range(1, adata.shape[0]):
        if adata.obs["sample"][i] != sample_list[-1]:
            sample_list.append( adata.obs["sample"][i] )
    # convert sample name to index
    sample_ids = np.zeros(adata.shape[0], dtype=int)
    for s,sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        sample_ids[index] = s

    if not config["tumorprop_file"] is None:
        df_tumorprop = pd.read_csv(config["tumorprop_file"], sep="\t", header=0, index_col=0)
        df_tumorprop = df_tumorprop[["Tumor"]]
        df_tumorprop.columns = ["tumor_proportion"]
        adata.obs = adata.obs.join(df_tumorprop)
        single_tumor_prop = adata.obs["tumor_proportion"]
    else:
        single_tumor_prop = None
    
    # read original data
    # lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list = convert_to_hmm_input_v2(adata, \
    #     cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, config["hgtable_file"], config["nu"], config["logphase_shift"])
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list = convert_to_hmm_input_new(adata, \
        cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, config["hgtable_file"], config["nu"], config["logphase_shift"])
    # infer an initial phase using pseudobulk
    if not Path(f"{config['output_dir']}/initial_phase.npz").exists():
        # phase_indicator, refined_lengths = infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, 5, log_sitewise_transmat, "sp", \
        #     1-1e-6, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], config["max_iter"], 1e-3)
        initial_clone_for_phasing = perform_partition(coords, sample_ids, x_part=config["npart_phasing"], y_part=config["npart_phasing"], single_tumor_prop=single_tumor_prop, threshold=config["tumorprop_threshold"])
        phase_indicator, refined_lengths = initial_phase_given_partition(single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_for_phasing, 5, log_sitewise_transmat, \
            "sp", 1-1e-4, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], 30, 1e-3)
        np.savez(f"{config['output_dir']}/initial_phase.npz", phase_indicator=phase_indicator, refined_lengths=refined_lengths)
    else:
        tmp = dict(np.load(f"{config['output_dir']}/initial_phase.npz"))
        phase_indicator, refined_lengths = tmp["phase_indicator"], tmp["refined_lengths"]
    # binning with inferred phase
    # lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list = perform_binning(lengths, single_X, \
    #     single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, x_gene_list, phase_indicator, refined_lengths, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"])
    expected_nbins = min(config["max_nbins"], np.median(np.sum(single_total_bb_RD, axis=0)) * config["maxspots_pooling"] / config["avg_umi_perbinspot"] )
    expected_nbins = int(expected_nbins)
    secondary_min_umi = choose_umithreshold_given_nbins(single_total_bb_RD, refined_lengths, expected_nbins)
    print(f"Secondary_min_umi = {secondary_min_umi}")
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list = perform_binning_new(lengths, single_X, \
        single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, phase_indicator, refined_lengths, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"], secondary_min_umi=secondary_min_umi)

    if not config["tumorprop_file"] is None:
        index_normal = np.where(single_tumor_prop < 0.01)[0]
        lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, _ = bin_selection_basedon_normal(single_X, \
                single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, config["nu"], config["logphase_shift"], index_normal)

    unique_chrs = np.arange(1, 23)
    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)
    single_X[:,0,:] = 0
    single_base_nb_mean[:,:] = 0

    # smooth and adjacency matrix for each sample
    adjacency_mat = []
    smooth_mat = []
    for sname in sample_list:
        index = np.where(adata.obs["sample"] == sname)[0]
        this_coords = np.array(coords[index,:])
        tmpsmooth_mat, tmpadjacency_mat, sw_adjustment = choose_adjacency_by_readcounts(this_coords, single_total_bb_RD[:,index], maxspots_pooling=config["maxspots_pooling"])
        adjacency_mat.append( tmpadjacency_mat.A )
        smooth_mat.append( tmpsmooth_mat.A )
    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)
    adjacency_mat = scipy.sparse.csr_matrix( adjacency_mat )
    if not across_slice_adjacency_mat is None:
        adjacency_mat += across_slice_adjacency_mat
    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix( smooth_mat )
    n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())
    print(f"Set up number of spots to pool in HMRF: {n_pooled}")

    # save binned data
    np.savez(f"{config['output_dir']}/binned_data_withoutrdr.npz", lengths=lengths, single_X=single_X, single_base_nb_mean=single_base_nb_mean, single_total_bb_RD=single_total_bb_RD, single_tumor_prop=(None if config["tumorprop_file"] is None else single_tumor_prop))

    # run BAF-only HMRF
    for r_hmrf_initialization in range(config["num_hmrf_initialization_start"], config["num_hmrf_initialization_end"]):
        if config["initialization_method"] == "rectangle":
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            if config["tumorprop_file"] is None:
                initial_clone_index = rectangle_initialize_initial_clone(coords, config["n_clones"], random_state=r_hmrf_initialization)
            else:
                initial_clone_index = rectangle_initialize_initial_clone_mix(coords, config["n_clones"], single_tumor_prop, threshold=config["tumorprop_threshold"], random_state=r_hmrf_initialization)
        elif config["initialization_method"] == "datadrive":
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_datadriven{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            initial_clone_index = data_driven_initialize_initial_clone(single_X, single_total_bb_RD, phase_indicator, config["n_clones"], config["n_clones"], sorted_chr_pos, coords, r_hmrf_initialization)
        elif config["initialization_method"] == "sample":
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_sample{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            initial_clone_index = sample_initialize_initial_clone(adata, sample_list, config["n_clones"], random_state=r_hmrf_initialization)

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
                hmmclass=hmm_nophasing, params="sp", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
        else:
            hmrfmix_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_index, n_states=config["n_states"], \
                log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, sample_ids=sample_ids, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
                hmmclass=hmm_nophasing, params="sp", t=config["t"], random_state=config["gmm_random_state"], \
                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

        # merge by thresholding BAF profile similarity
        res = load_hmrf_last_iteration(f"{outdir}/{prefix}_nstates{config['n_states']}_sp.npz")
        n_obs = single_X.shape[0]
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))])
            tumor_prop = None
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res["new_assignment"]==c)[0] for c in np.sort(np.unique(res["new_assignment"]))], single_tumor_prop)
            tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1,1)
        merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, threshold=config["np_threshold"], minlength=config["np_eventminlen"], params="sp", tumor_prop=tumor_prop, hmmclass=hmm_nophasing)
        print(f"BAF clone merging after comparing similarity: {merging_groups}")
        #
        if config["tumorprop_file"] is None:
            merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, min_spots_thresholds=config["min_spots_per_clone"])
        else:
            merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], merged_res, min_spots_thresholds=config["min_spots_per_clone"], single_tumor_prop=single_tumor_prop)
        print(f"BAF clone merging after requiring minimum # spots: {merging_groups}")
        n_baf_clones = len(merging_groups)
        merged_baf_assignment = copy.copy(merged_res["new_assignment"])
        np.savez(f"{outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", **merged_res)

        # # adjust phasing
        n_obs = single_X.shape[0]
        merged_res = dict(np.load(f"{outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", allow_pickle=True))
        merged_baf_assignment = copy.copy(merged_res["new_assignment"])
        n_baf_clones = len(np.unique(merged_baf_assignment))
        pred = np.argmax(merged_res["log_gamma"], axis=0)
        pred = np.array([ pred[(c*n_obs):(c*n_obs+n_obs)] for c in range(n_baf_clones) ])
        merged_baf_profiles = np.array([ np.where(pred[c,:] < config["n_states"], merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0], 1-merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0]) \
                                        for c in range(n_baf_clones) ])

        # adding RDR information
        if not config["bafonly"]:
            n_clones_rdr = 2
            # select normal spots
            if (config["normalidx_file"] is None) and (config["tumorprop_file"] is None):
                EPS_BAF = 0.05
                PERCENT_NORMAL = 40
                vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
                id_nearnormal_clone = np.argmin(np.sum( np.maximum(np.abs(merged_baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
                while True:
                    stdthreshold = np.percentile(vec_stds[merged_res["new_assignment"] == id_nearnormal_clone], PERCENT_NORMAL)
                    adata.obs["normal_candidate"] = (vec_stds < stdthreshold) & (merged_res["new_assignment"] == id_nearnormal_clone)
                    if np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)]) > single_X.shape[0] * 200 or PERCENT_NORMAL == 100:
                        break
                    PERCENT_NORMAL += 10
                copy_single_X_rdr, _ = filter_de_genes(adata, x_gene_list, sample_list) # TO BE CHECKED REGARDING SAMPLE BATCHES
                MIN_NORMAL_COUNT_PERBIN = 20
                bidx_inconfident = np.where( np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)], axis=1) < MIN_NORMAL_COUNT_PERBIN )[0]
                rdr_normal = np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)], axis=1)
                rdr_normal[bidx_inconfident] = 0
                rdr_normal = rdr_normal / np.sum(rdr_normal)
                copy_single_X_rdr[bidx_inconfident, :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
                copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
                pd.Series(adata.obs[adata.obs["normal_candidate"]==True].index).to_csv(f"{outdir}/normal_candidate_barcodes.txt", header=False, index=False)
                #
                index_normal = np.where(adata.obs["normal_candidate"])[0]
                lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, index_remaining = bin_selection_basedon_normal(single_X, \
                        single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, config["nu"], config["logphase_shift"], index_normal)
                copy_single_X_rdr = copy_single_X_rdr[index_remaining, :]
                copy_single_base_nb_mean = copy_single_base_nb_mean[index_remaining, :]
                
            elif (not config["normalidx_file"] is None):
                # single_base_nb_mean has already been added in loading data step.
                if not config["tumorprop_file"] is None:
                    logger.warning(f"Mixed sources of information for normal spots! Using {config['normalidx_file']}")
            else:
                for prop_threshold in np.arange(0.05, 0.3, 0.05):
                    adata.obs["normal_candidate"] = (adata.obs["tumor_proportion"] < prop_threshold)
                    if np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)]) > single_X.shape[0] * 200:
                        break
                copy_single_X_rdr,_ = filter_de_genes(adata, x_gene_list, sample_list)
                MIN_NORMAL_COUNT_PERBIN = 20
                bidx_inconfident = np.where( np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)], axis=1) < MIN_NORMAL_COUNT_PERBIN )[0]
                rdr_normal = np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)], axis=1)
                rdr_normal[bidx_inconfident] = 0
                rdr_normal = rdr_normal / np.sum(rdr_normal)
                copy_single_X_rdr[bidx_inconfident, :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
                copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
                
            # adding back RDR signal
            single_X[:,0,:] = copy_single_X_rdr
            single_base_nb_mean = copy_single_base_nb_mean
            n_obs = single_X.shape[0]

            # save binned data
            np.savez(f"{outdir}/binned_data.npz", lengths=lengths, single_X=single_X, single_base_nb_mean=single_base_nb_mean, single_total_bb_RD=single_total_bb_RD, single_tumor_prop=(None if config["tumorprop_file"] is None else single_tumor_prop))

            # run HMRF on each clone individually to further split BAF clone by RDR+BAF signal
            for bafc in range(n_baf_clones):
                prefix = f"clone{bafc}"
                idx_spots = np.where(merged_baf_assignment == bafc)[0]
                if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 20: # put a minimum B allele read count on pseudobulk to split clones
                    continue
                # initialize clone
                if config["tumorprop_file"] is None:
                    initial_clone_index = rectangle_initialize_initial_clone(coords[idx_spots], n_clones_rdr, random_state=r_hmrf_initialization)
                else:
                    initial_clone_index = rectangle_initialize_initial_clone_mix(coords[idx_spots], n_clones_rdr, single_tumor_prop[idx_spots], threshold=config["tumorprop_threshold"], random_state=r_hmrf_initialization)
                if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz").exists():
                    initial_assignment = np.zeros(len(idx_spots), dtype=int)
                    for c,idx in enumerate(initial_clone_index):
                        initial_assignment[idx] = c
                    allres = {"barcodes":adata.obs.index[idx_spots], "num_iterations":0, "round-1_assignment":initial_assignment}
                    np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", **allres)
                #
                # HMRF + HMM using RDR information
                copy_slice_sample_ids = copy.copy(sample_ids[idx_spots])
                if config["tumorprop_file"] is None:
                    hmrf_concatenate_pipeline(outdir, prefix, single_X[:,:,idx_spots], lengths, single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], initial_clone_index, n_states=config["n_states"], \
                        log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat[np.ix_(idx_spots,idx_spots)], adjacency_mat=adjacency_mat[np.ix_(idx_spots,idx_spots)], sample_ids=copy_slice_sample_ids, max_iter_outer=10, nodepotential=config["nodepotential"], \
                        hmmclass=hmm_nophasing, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                        fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                        fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                        is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
                else:
                    hmrfmix_concatenate_pipeline(outdir, prefix, single_X[:,:,idx_spots], lengths, single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], single_tumor_prop[idx_spots], initial_clone_index, n_states=config["n_states"], \
                        log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat[np.ix_(idx_spots,idx_spots)], adjacency_mat=adjacency_mat[np.ix_(idx_spots,idx_spots)], sample_ids=copy_slice_sample_ids, max_iter_outer=10, nodepotential=config["nodepotential"], \
                        hmmclass=hmm_nophasing, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                        fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                        fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                        is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

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
                idx_spots = np.where(adata.obs.index.isin( allres["barcodes"] ))[0]
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
                        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in range(n_clones_rdr)])
                        tumor_prop = None
                    else:
                        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(res["new_assignment"]==c)[0] for c in range(n_clones_rdr)], single_tumor_prop[idx_spots])
                        tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1,1)
                    merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(X, base_nb_mean, total_bb_RD, res, threshold=config["np_threshold"], minlength=config["np_eventminlen"], params="smp", tumor_prop=tumor_prop, hmmclass=hmm_nophasing)
                    print(f"part {bafc} merging_groups: {merging_groups}")
                    #
                    if config["tumorprop_file"] is None:
                        merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], res, min_spots_thresholds=config["min_spots_per_clone"])
                    else:
                        merging_groups, merged_res = merge_by_minspots(merged_res["new_assignment"], res, min_spots_thresholds=config["min_spots_per_clone"], single_tumor_prop=single_tumor_prop[idx_spots])
                    print(f"part {bafc} merging after requiring minimum # spots: {merging_groups}")
                    # compute posterior using the newly merged pseudobulk
                    n_merged_clones = len(merging_groups)
                    tmp = copy.copy(merged_res["new_assignment"])
                    if config["tumorprop_file"] is None:
                        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)])
                        tumor_prop = None
                    else:
                        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)], single_tumor_prop[idx_spots])
                    #
                    merged_res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config["n_states"], \
                            # base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), tumor_prop, \
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), np.repeat(tumor_prop, X.shape[0]).reshape(-1,1) if not tumor_prop is None else None, \
                            hmmclass=hmm_nophasing, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                            is_diag=True, init_log_mu=res["new_log_mu"], init_p_binom=res["new_p_binom"], init_alphas=res["new_alphas"], init_taus=res["new_taus"], max_iter=config["max_iter"], tol=config["tol"])
                    merged_res["new_assignment"] = copy.copy(tmp)
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
            # res_combine["new_alphas"][:,:] = np.max(res_combine["new_alphas"])
            # res_combine["new_taus"][:,:] = np.min(res_combine["new_taus"])
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
                        smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing, return_posterior=True)
                elif config["nodepotential"] == "weighted_sum":
                    new_assignment, single_llf, total_llf, posterior = hmrf_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, res_combine, \
                        smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing, return_posterior=True)
            else:
                if config["nodepotential"] == "max":
                    pred = np.vstack([ np.argmax(res_combine["log_gamma"][:,:,c], axis=0) for c in range(res_combine["log_gamma"].shape[2]) ]).T
                    new_assignment, single_llf, total_llf, posterior = aggr_hmrfmix_reassignment(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res_combine, pred, \
                        smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing, return_posterior=True)
                elif config["nodepotential"] == "weighted_sum":
                    new_assignment, single_llf, total_llf, posterior = hmrfmix_reassignment_posterior(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, res_combine, \
                        smooth_mat, adjacency_mat, res_combine["prev_assignment"], copy.copy(sample_ids), log_persample_weights, spatial_weight=config["spatial_weight"], hmmclass=hmm_nophasing, return_posterior=True)
            res_combine["total_llf"] = total_llf
            res_combine["new_assignment"] = new_assignment
            # res_combine = dict(np.load(f"{outdir}/original_rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))
            # posterior = np.load(f"{outdir}/original_posterior_clone_probability.npy")
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
            medfix = ["", "_diploid", "_triploid", "_tetraploid"]
            for o,max_medploidy in enumerate([None, 2, 3, 4]):
                # A/B copy number per bin
                allele_specific_copy = []
                # A/B copy number per state
                state_cnv = []
                #
                df_genelevel_cnv = None
                if config["tumorprop_file"] is None:
                    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids])
                else:
                    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids], single_tumor_prop)
                #
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
                        best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv)
                    print(f"max med ploidy = {max_medploidy}, clone {s}, integer copy inference loss = {_}")
                    
                    allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 0].reshape(1,-1), index=[f"clone{cid} A"], columns=np.arange(n_obs) ) )
                    allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 1].reshape(1,-1), index=[f"clone{cid} B"], columns=np.arange(n_obs) ) )
                    #
                    state_cnv.append( pd.DataFrame( res_combine["new_log_mu"][:,s].reshape(-1,1), columns=[f"clone{cid} logmu"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( res_combine["new_p_binom"][:,s].reshape(-1,1), columns=[f"clone{cid} p"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( best_integer_copies[:,0].reshape(-1,1), columns=[f"clone{cid} A"], index=np.arange(config['n_states']) ) )
                    state_cnv.append( pd.DataFrame( best_integer_copies[:,1].reshape(-1,1), columns=[f"clone{cid} B"], index=np.arange(config['n_states']) ) )
                    #
                    tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[res_combine["pred_cnv"][:,s], 0], best_integer_copies[res_combine["pred_cnv"][:,s], 1], x_gene_list)
                    tmpdf.columns = [f"clone{s} A", f"clone{s} B"]
                    if df_genelevel_cnv is None:
                        df_genelevel_cnv = copy.copy(tmpdf)
                    else:
                        df_genelevel_cnv = df_genelevel_cnv.join(tmpdf)
                # output gene-level copy number
                df_genelevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_genelevel.tsv", header=True, index=True, sep="\t")
                # output segment-level copy number
                allele_specific_copy = pd.concat(allele_specific_copy)
                df_seglevel_cnv = pd.DataFrame({"CHR":[x[0] for x in sorted_chr_pos], "START":[x[1] for x in sorted_chr_pos], \
                    "END":[ (sorted_chr_pos[i+1][1] if i+1 < len(sorted_chr_pos) and x[0]==sorted_chr_pos[i+1][0] else -1) for i,x in enumerate(sorted_chr_pos)] })
                df_seglevel_cnv = df_seglevel_cnv.join( allele_specific_copy.T )
                df_seglevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_seglevel.tsv", header=True, index=False, sep="\t")
                # output per-state copy number
                state_cnv = functools.reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True, how='inner'), state_cnv)
                state_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_perstate.tsv", header=True, index=False, sep="\t")
                #
                # posterior using integer-copy numbers
                log_persample_weights = np.zeros((len(nonempty_clone_ids), len(sample_list)))
                for sidx in range(len(sample_list)):
                    index = np.where(sample_ids == sidx)[0]
                    this_persample_weight = np.array([ np.sum(res_combine["new_assignment"][index] == cid) for cid in nonempty_clone_ids]) / len(index)
                    log_persample_weights[:, sidx] = np.where(this_persample_weight > 0, np.log(this_persample_weight), -50)
                    log_persample_weights[:, sidx] = log_persample_weights[:, sidx] - scipy.special.logsumexp(log_persample_weights[:, sidx])
                pred = np.vstack([ np.argmax(res_combine["log_gamma"][:,:,cid], axis=0) for cid in nonempty_clone_ids ]).T
                df_posterior = clonelabel_posterior_withinteger(single_X, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, state_cnv, res_combine, pred, \
                    smooth_mat, adjacency_mat, res_combine["new_assignment"], sample_ids, base_nb_mean, log_persample_weights, config["spatial_weight"], hmmclass=hmm_nophasing)
                df_posterior.to_pickle(f"{outdir}/posterior{medfix[o]}.pkl")

            ##### output clone label #####
            adata.obs["clone_label"] = res_combine["new_assignment"]
            if config["tumorprop_file"] is None:
                adata.obs[["clone_label"]].to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")
            else:
                adata.obs[["tumor_proportion", "clone_label"]].to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])