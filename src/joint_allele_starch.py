import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import anndata
import logging
import copy
from pathlib import Path
import subprocess
from hmm_NB_BB_phaseswitch import *
from composite_hmm_NB_BB_phaseswitch import *
from utils_distribution_fitting import *
from hmrf import *
from utils_IO import *

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
        "filtergenelist_file" : None,
        "binsize" : 1,
        "rdrbinsize" : 1,
        "bafonly" : True,
        "logfcthreshold" : 2.5,
        # phase switch probability
        "nu" : 1,
        "logphase_shift" : 1,
        # HMRF configurations
        "n_clones" : None,
        "max_iter_outer" : 20,
        "nodepotential" : "max", # max or weighted_sum
        "initialization_method" : "rectangle", # rectangle or datadrive
        "num_hmrf_initialization_start" : 0, 
        "num_hmrf_initialization_end" : 10,
        # HMM configurations
        "n_states" : None,
        "n_baf_states" : None,
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
        "relative_rdr_weight" : 1.0
    }

    argument_type = {
        "input_filelist" : "str",
        "snp_dir" : "str",
        "output_dir" : "str",
        # supporting files and preprocessing arguments
        "hgtable_file" : "str",
        "normalidx_file" : "str",
        "filtergenelist_file" : "str",
        "binsize" : "int",
        "rdrbinsize" : "int",
        "bafonly" : "bool",
        "logfcthreshold" : "float",
        # phase switch probability
        "nu" : "float",
        "logphase_shift" : "float",
        # HMRF configurations
        "n_clones" : "int",
        "max_iter_outer" : "int",
        "nodepotential" : "str",
        "initialization_method" : "str",
        "num_hmrf_initialization_start" : "int", 
        "num_hmrf_initialization_end" : "int",
        # HMM configurations
        "n_states" : "int",
        "n_baf_states" : "int",
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
        "relative_rdr_weight" : "float"
    }

    ##### [ read configuration file to update settings ] #####
    with open(filename, 'r') as fp:
        for line in fp:
            if line.strip() == "" or line[0] == "#":
                continue
            strs = [x.replace(" ", "") for x in line.strip().split(":") if x != ""]
            assert strs[0] in config.keys(), f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
            if strs[1].upper() == "NONE":
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
    # assertions
    assert not config["input_filelist"] is None, "No input file list!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    if config["n_baf_states"] is None:
        config["n_baf_states"] = config["n_states"]

    return config


def main(configuration_file):
    config = read_joint_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids = load_joint_data(config["input_filelist"], config["snp_dir"], config["filtergenelist_file"], config["normalidx_file"], config["logfcthreshold"])
    # read original data
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list = convert_to_hmm_input_v2(adata, \
        cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, config["hgtable_file"], config["nu"], config["logphase_shift"])
    # infer an initial phase using pseudobulk
    if not Path(f"{config['output_dir']}/initial_phase.npz").exists():
        # phase_prob = infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, 5, log_sitewise_transmat, "sp", \
        #     1-1e-6, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], config["max_iter"], 1e-3)
        # np.save(f"{config['output_dir']}/initial_phase_prob.npy", phase_prob)
        phase_indicator, refined_lengths = infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, 5, log_sitewise_transmat, "sp", \
            1-1e-6, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], config["max_iter"], 1e-3)
        np.savez(f"{config['output_dir']}/initial_phase.npz", phase_indicator=phase_indicator, refined_lengths=refined_lengths)
    else:
        tmp = dict(np.load(f"{config['output_dir']}/initial_phase.npz"))
        phase_indicator, refined_lengths = tmp["phase_indicator"], tmp["refined_lengths"]
    # binning with inferred phase
    # lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list = perform_binning(lengths, single_X, \
    #     single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, x_gene_list, phase_prob, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"])
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list = perform_binning(lengths, single_X, \
        single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, x_gene_list, phase_indicator, refined_lengths, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"])


    unique_chrs = np.arange(1, 23)
    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)
    single_X[:,0,:] = 0
    single_base_nb_mean[:,:] = 0
    
    # construct adjacency matrix
    adjacency_mat = []
    smooth_mat = []
    sample_list = [adata.obs["sample"][0]]
    for i in range(1, adata.shape[0]):
        if adata.obs["sample"][i] != sample_list[-1]:
            sample_list.append( adata.obs["sample"][i] )
    for sname in sample_list:
        index = np.where(adata.obs["sample"] == sname)[0]
        this_coords = np.array(adata.obsm["X_pos"][index,:])
        # adjacency_mat.append( compute_adjacency_mat(this_coords).A )
        tmpsmooth_mat, tmpadjacency_mat, sw_adjustment = choose_adjacency_by_readcounts(this_coords, single_total_bb_RD[:,index])
        adjacency_mat.append( tmpadjacency_mat.A )
        smooth_mat.append( tmpsmooth_mat.A )
    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)
    adjacency_mat = scipy.sparse.csr_matrix( adjacency_mat )
    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix( smooth_mat )
    coords = adata.obsm["X_pos"]
    config["spatial_weight"] *= sw_adjustment
    n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())
    print(f"Set up number of spots to pool in HMRF: {n_pooled}")

    # run BAF-only HMRF
    for r_hmrf_initialization in range(config["num_hmrf_initialization_start"], config["num_hmrf_initialization_end"]):
        if config["initialization_method"] == "rectangle":
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            initial_clone_index = rectangle_initialize_initial_clone(coords, config["n_clones"], random_state=r_hmrf_initialization)
        elif config["initialization_method"] == "datadrive":
            if not Path(f"{config['output_dir']}/initial_phase_prob.npy").exists():
                phase_prob = infer_initial_phase(single_X, lengths, single_base_nb_mean, single_total_bb_RD, 5, log_sitewise_transmat, config["params"], \
                    1-1e-6, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], config["max_iter"], 1e-3)
                np.save(f"{config['output_dir']}/initial_phase_prob.npy", phase_prob)
            else:
                phase_prob = np.load(f"{config['output_dir']}/initial_phase_prob.npy")
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_datadriven{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            initial_clone_index = data_driven_initialize_initial_clone(single_X, single_total_bb_RD, phase_prob, config["n_clones"], config["n_clones"], sorted_chr_pos, coords, r_hmrf_initialization)
        elif config["initialization_method"] == "sample":
            outdir = f"{config['output_dir']}/clone{config['n_clones']}_sample{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
            initial_clone_index = sample_initialize_initial_clone(adata, sample_list, config["n_clones"], random_state=r_hmrf_initialization)

        # create directory
        p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = p.communicate()
        # save clone initialization into npz file
        prefix = "allspots"
        if not Path(f"{outdir}/{prefix}_nstates{config['n_baf_states']}_sp.npz").exists():
            initial_assignment = np.zeros(single_X.shape[2], dtype=int)
            for c,idx in enumerate(initial_clone_index):
                initial_assignment[idx] = c
            allres = {"num_iterations":0, "round-1_assignment":initial_assignment}
            # np.savez(f"{outdir}/{prefix}_nstates{config['n_baf_states']}_sp.npz", **allres)
            np.savez(f"{outdir}/{prefix}_nstates{config['n_baf_states']}_sp.npz", **allres)
        
        # run HMRF + HMM
        # hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_baf_states"], \
        #     log_sitewise_transmat=log_sitewise_transmat, adjacency_mat=adjacency_mat, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
        #     params=config["params"], t=config["t"], random_state=config["gmm_random_state"], \
        #     fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
        #     fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
        #     relative_rdr_weight=config["relative_rdr_weight", is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])
        hmrf_concatenate_pipeline(outdir, prefix, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_baf_states"], \
            log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat, adjacency_mat=adjacency_mat, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
            params="sp", t=config["t"], random_state=config["gmm_random_state"], \
            # params=config["params"], t=config["t"], random_state=config["gmm_random_state"], \
            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
            relative_rdr_weight=config["relative_rdr_weight"], is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

        # merge by thresholding BAF profile similarity
        allres = np.load(f"{outdir}/{prefix}_nstates{config['n_baf_states']}_sp.npz")
        allres = dict(allres)
        r = allres["num_iterations"] - 1
        res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
            "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
            "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
            "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
            "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
        n_obs = single_X.shape[0]
        baf_profiles = np.array([ res["new_p_binom"][res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)], 0] for c in range(config["n_clones"]) ])
        merging_groups, merged_res = similarity_components_baf(baf_profiles, res)
        n_baf_clones = len(merging_groups)
        merged_baf_assignment = copy.copy(merged_res["new_assignment"])
        np.savez(f"{outdir}/mergedallspots_nstates{config['n_baf_states']}_sp.npz", **merged_res)
        
        # adding RDR information
        if not config["bafonly"]:
            n_clones_rdr = 2
            # select normal spots
            if config["normalidx_file"] is None:
                EPS_BAF = 0.05
                PERCENT_NORMAL = 40
                vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
                id_nearnormal_clone = np.argmin(np.sum( np.maximum(np.abs(baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
                while True:
                    stdthreshold = np.percentile(vec_stds[res["new_assignment"] == id_nearnormal_clone], PERCENT_NORMAL)
                    adata.obs["normal_candidate"] = (vec_stds < stdthreshold) & (res["new_assignment"] == id_nearnormal_clone)
                    if np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)]) > single_X.shape[0] * 200 or PERCENT_NORMAL == 100:
                        break
                    PERCENT_NORMAL += 10
                rdr_normal = np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)], axis=1) / np.sum(copy_single_X_rdr[:, (adata.obs["normal_candidate"]==True)])
                copy_single_X_rdr[(rdr_normal==0), :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
                copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
                pd.Series(adata.obs[adata.obs["normal_candidate"]==True].index).to_csv(f"{outdir}/normal_candidate_barcodes.txt", header=False, index=False)
            
            # adding back RDR signal
            single_X[:,0,:] = copy_single_X_rdr
            single_base_nb_mean = copy_single_base_nb_mean

            # run HMRF on each clone individually to further split BAF clone by RDR+BAF signal
            for bafc in range(n_baf_clones):
                prefix = f"clone{bafc}"
                idx_spots = np.where(merged_res["new_assignment"] == bafc)[0]
                if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 20: # put a minimum B allele read count on pseudobulk to split clones
                    continue
                # initialize clone
                initial_clone_index = rectangle_initialize_initial_clone(coords[idx_spots], n_clones_rdr, random_state=r_hmrf_initialization)
                if not Path(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz").exists():
                    initial_assignment = np.zeros(len(idx_spots), dtype=int)
                    for c,idx in enumerate(initial_clone_index):
                        initial_assignment[idx] = c
                    allres = {"barcodes":adata.obs.index[idx_spots], "num_iterations":0, "round-1_assignment":initial_assignment}
                    np.savez(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", **allres)

                hmrf_concatenate_pipeline(outdir, prefix, single_X[:,:,idx_spots], lengths, single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], initial_clone_index, n_states=config["n_states"], \
                    log_sitewise_transmat=log_sitewise_transmat, smooth_mat=smooth_mat[np.ix_(idx_spots,idx_spots)], adjacency_mat=adjacency_mat[np.ix_(idx_spots,idx_spots)], max_iter_outer=10, nodepotential=config["nodepotential"], \
                    params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                    fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                    fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                    relative_rdr_weight=config["relative_rdr_weight"], is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

            # combine results across clones
            res_combine = {"new_assignment":np.zeros(single_X.shape[2], dtype=int)}
            offset_clone = 0
            for bafc in range(n_baf_clones):
                prefix = f"clone{bafc}"
                idx_spots = np.where(merged_baf_assignment == bafc)[0]
                allres = dict( np.load(f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz", allow_pickle=True) )
                r = allres["num_iterations"] - 1
                res = {"new_log_mu":allres[f"round{r}_new_log_mu"], "new_alphas":allres[f"round{r}_new_alphas"], \
                    "new_p_binom":allres[f"round{r}_new_p_binom"], "new_taus":allres[f"round{r}_new_taus"], \
                    "new_log_startprob":allres[f"round{r}_new_log_startprob"], "new_log_transmat":allres[f"round{r}_new_log_transmat"], "log_gamma":allres[f"round{r}_log_gamma"], \
                    "pred_cnv":allres[f"round{r}_pred_cnv"], "llf":allres[f"round{r}_llf"], "total_llf":allres[f"round{r}_total_llf"], \
                    "prev_assignment":allres[f"round{r-1}_assignment"], "new_assignment":allres[f"round{r}_assignment"]}
                n_obs = single_X.shape[0]
                baf_profiles = np.array([ res["new_p_binom"][res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)], 0] for c in range(n_clones_rdr) ])
                rdr_profiles = np.array([ res["new_log_mu"][res["pred_cnv"][(c*n_obs):(c*n_obs+n_obs)], 0] for c in range(n_clones_rdr) ])
                merging_groups, merged_res = similarity_components_rdrbaf(baf_profiles, rdr_profiles, res)
                # compute posterior using the newly merged pseudobulk
                n_merged_clones = len(merging_groups)
                X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X[:,:,idx_spots], single_base_nb_mean[:,idx_spots], single_total_bb_RD[:,idx_spots], [np.where(merged_res["new_assignment"]==c)[0] for c in range(n_merged_clones)])
                log_emission = compute_emission_probability_nb_betabinom_v3(np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
                    base_nb_mean.flatten("F").reshape(-1,1), merged_res["new_log_mu"], merged_res["new_alphas"], total_bb_RD.flatten("F").reshape(-1,1), merged_res["new_p_binom"], merged_res["new_taus"])
                log_alpha = forward_lattice_sitewise(np.tile(lengths, X.shape[2]), merged_res["new_log_transmat"], merged_res["new_log_startprob"], log_emission, np.tile(log_sitewise_transmat, X.shape[2]),)
                log_beta = backward_lattice_sitewise(np.tile(lengths, X.shape[2]), merged_res["new_log_transmat"], merged_res["new_log_startprob"], log_emission, np.tile(log_sitewise_transmat, X.shape[2]),)
                log_gamma = compute_posterior_obs(log_alpha, log_beta)
                pred_cnv = np.argmax(log_gamma, axis=0) % config["n_states"]
                log_gamma = np.stack([ log_gamma[:,(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ], axis=-1)
                pred_cnv = np.vstack([ pred_cnv[(c*n_obs):(c*n_obs+n_obs)] for c in range(n_merged_clones) ]).T
                # add to res_combine
                if len(res_combine) == 1:
                    res_combine.update({"new_log_mu":np.hstack([ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([ merged_res["new_alphas"] ] * n_merged_clones), \
                        "new_p_binom":np.hstack([ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([ merged_res["new_taus"] ] * n_merged_clones), \
                        "log_gamma":log_gamma, "pred_cnv":pred_cnv})
                else:
                    res_combine.update({"new_log_mu":np.hstack([res_combine["new_log_mu"]] + [ merged_res["new_log_mu"] ] * n_merged_clones), "new_alphas":np.hstack([res_combine["new_alphas"]] + [ merged_res["new_alphas"] ] * n_merged_clones), \
                        "new_p_binom":np.hstack([res_combine["new_p_binom"]] + [ merged_res["new_p_binom"] ] * n_merged_clones), "new_taus":np.hstack([res_combine["new_taus"]] + [ merged_res["new_taus"] ] * n_merged_clones), \
                        "log_gamma":np.dstack([res_combine["log_gamma"], log_gamma ]), "pred_cnv":np.hstack([res_combine["pred_cnv"], pred_cnv])})
                res_combine["new_assignment"][idx_spots] = merged_res["new_assignment"] + offset_clone
                offset_clone += n_merged_clones
            # compute HMRF log likelihood
            total_llf = hmrf_log_likelihood(config["nodepotential"], single_X, single_base_nb_mean, single_total_bb_RD, res_combine, np.argmax(res_combine["log_gamma"],axis=0), smooth_mat, adjacency_mat, res_combine["new_assignment"], config["spatial_weight"])
            res_combine["total_llf"] = total_llf
            # save results
            np.savez(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", **res_combine)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])