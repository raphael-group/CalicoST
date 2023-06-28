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
import functools
import subprocess
from hmm_NB_BB_phaseswitch import *
from utils_distribution_fitting import *
from utils_hmrf import *
from hmrf import *
from phasing import *
from utils_IO import *
from find_integer_copynumber import *
from parse_input import *

import mkl
mkl.set_num_threads(1)


def read_configuration_file(filename):
    ##### [Default settings] #####
    config = {
        "spaceranger_dir" : None,
        "snp_dir" : None,
        "output_dir" : None,
        # supporting files and preprocessing arguments
        "hgtable_file" : None,
        "normalidx_file" : None,
        "tumorprop_file" : None,
        "supervision_clone_file" : None,
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
        "n_clones_rdr" : 2,
        "min_spots_per_clone" : 100,
        "maxspots_pooling" : 7,
        "tumorprop_threshold" : 0.5, 
        "max_iter_outer" : 20,
        "nodepotential" : "max", # max or weighted_sum
        "initialization_method" : "rectangle", # rectangle or datadrive
        "num_hmrf_initialization_start" : 0, 
        "num_hmrf_initialization_end" : 10,
        "spatial_weight" : 2.0,
        "construct_adjacency_method" : "hexagon",
        "construct_adjacency_w" : 1.0,
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
        "gmm_random_state" : 0,
        "np_threshold" : 2.0,
        "np_eventminlen" : 10
    }

    argument_type = {
        "spaceranger_dir" : "str",
        "snp_dir" : "str",
        "output_dir" : "str",
        # supporting files and preprocessing arguments
        "hgtable_file" : "str",
        "normalidx_file" : "str",
        "tumorprop_file" : "str",
        "supervision_clone_file" : "str",
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
        "n_clones_rdr" : "int",
        "min_spots_per_clone" : "int",
        "maxspots_pooling" : "int",
        "tumorprop_threshold" : "float", 
        "max_iter_outer" : "int",
        "nodepotential" : "str",
        "initialization_method" : "str",
        "num_hmrf_initialization_start" : "int", 
        "num_hmrf_initialization_end" : "int",
        "spatial_weight" : "float",
        "construct_adjacency_method" : "str",
        "construct_adjacency_w" : "float",
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
        "gmm_random_state" : "int",
        "np_threshold" : "float",
        "np_eventminlen" : "int"
    }

    ##### [ read configuration file to update settings ] #####
    with open(filename, 'r') as fp:
        for line in fp:
            if line.strip() == "" or line[0] == "#":
                continue
            # strs = [x.replace(" ", "") for x in line.strip().split(":") if x != ""]
            strs = [x.strip() for x in line.strip().split(":") if x != ""]
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
            elif argument_type[strs[0]] == "list_str":
                config[strs[0]] = strs[1].split(" ")
    # assertions
    assert not config["spaceranger_dir"] is None, "No spaceranger directory!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    return config


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
        "supervision_clone_file" : None,
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
        "n_clones_rdr" : 2,
        "min_spots_per_clone" : 100,
        "maxspots_pooling" : 7,
        "tumorprop_threshold" : 0.5, 
        "max_iter_outer" : 20,
        "nodepotential" : "max", # max or weighted_sum
        "initialization_method" : "rectangle", # rectangle or datadrive
        "num_hmrf_initialization_start" : 0, 
        "num_hmrf_initialization_end" : 10,
        "spatial_weight" : 2.0,
        "construct_adjacency_method" : "hexagon",
        "construct_adjacency_w" : 1.0,
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
        "gmm_random_state" : 0,
        "np_threshold" : 2.0,
        "np_eventminlen" : 10
    }

    argument_type = {
        "input_filelist" : "str",
        "snp_dir" : "str",
        "output_dir" : "str",
        # supporting files and preprocessing arguments
        "hgtable_file" : "str",
        "normalidx_file" : "str",
        "tumorprop_file" : "str",
        "supervision_clone_file" : "str",
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
        "n_clones_rdr" : "int",
        "min_spots_per_clone" : "int",
        "maxspots_pooling" : "int",
        "tumorprop_threshold" : "float", 
        "max_iter_outer" : "int",
        "nodepotential" : "str",
        "initialization_method" : "str",
        "num_hmrf_initialization_start" : "int", 
        "num_hmrf_initialization_end" : "int",
        "spatial_weight" : "float",
        "construct_adjacency_method" : "str",
        "construct_adjacency_w" : "float",
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
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
        
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, x_gene_list, \
            barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)

    outdir = f"{config['output_dir']}/gmmrandomstate_{config['gmm_random_state']}"
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()

    # process RDR (single_base_nb_mean)
    EXPECTED_NORMAL_PROP = 0.05
    q = np.sort(single_tumor_prop)[ int(EXPECTED_NORMAL_PROP * len(barcodes)) ]
    normal_candidate = ( single_tumor_prop < q )

    copy_single_X_rdr,_ = filter_de_genes(exp_counts, x_gene_list, normal_candidate)
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

    # save binned data
    if not Path(f"{config['output_dir']}/binned_data.npz").exists():
        np.savez(f"{config['output_dir']}/binned_data.npz", lengths=lengths, single_X=single_X, single_base_nb_mean=single_base_nb_mean, single_total_bb_RD=single_total_bb_RD, log_sitewise_transmat=log_sitewise_transmat, single_tumor_prop=(None if config["tumorprop_file"] is None else single_tumor_prop))

    # supervision clones
    tmp_df_clones = pd.read_csv(config["supervision_clone_file"], header=0, index_col=0, sep="\t")
    df_clones = pd.DataFrame({"barcodes":barcodes.values}, index=barcodes.values).join(tmp_df_clones)
    df_clones.columns = ["barcodes", "clone_id"]

    unique_clone_ids = np.unique( df_clones["clone_id"][~df_clones["clone_id"].isnull()].values )
    clone_index = [np.where(df_clones["clone_id"] == c)[0] for c in unique_clone_ids]
    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop, threshold=config["tumorprop_threshold"])

    ##### HMM inference for abstract copy numbers #####
    # check whether the HMM parameters are already outputted, and only run Baum-Welch algorithm to infer if not.
    if not Path(f"{outdir}/hmm_nstates{config['n_states']}_smp.npz").exists():
        # logger print the progress to stdout
        logger.info("Running Baum-Welch algorithm for HMM inference...")

        # initialize HMM parameters by GMM
        init_log_mu, init_p_binom = initialization_by_gmm(config["n_states"], np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), params="smp", random_state=config["gmm_random_state"], in_log_space=False, only_minor=False)
        # initialization parameters for HMM
        last_log_mu = init_log_mu
        last_p_binom = init_p_binom
        last_alphas = None
        last_taus = None

        if config["tumorprop_file"] is None:
            res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config["n_states"], \
                                base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), \
                                hmmclass=hmm_nophasing_v2, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                                fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                                fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                                is_diag=True, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=config["max_iter"], tol=config["tol"])
        else:
            # baseline proportion of UMI counts
            lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
            sample_length = np.ones(X.shape[2],dtype=int) * X.shape[0]
            remain_kwargs = {"sample_length":sample_length, "lambd":lambd}
            res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), config["n_states"], \
                            base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), \
                            tumor_prop=np.repeat(tumor_prop, X.shape[0]).reshape(-1,1), hmmclass=hmm_nophasing_v2, params="smp", t=config["t"], random_state=config["gmm_random_state"], \
                            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
                            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
                            is_diag=True, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=config["max_iter"], tol=config["tol"], **remain_kwargs)

        np.savez(f"{outdir}/hmm_nstates{config['n_states']}_smp.npz", **res)
    else:
        # logger print the progress to stdout
        logger.info("Find HMM output file, skip Baum-Welch algorithm...")

    ##### infer integer copy #####
    # load HMM parameters
    res = dict(np.load(f"{outdir}/hmm_nstates{config['n_states']}_smp.npz", allow_pickle=True))
    n_obs = X.shape[0]
    # assuming genome is diploid, triploid, and tetraploid, convert HMM parameters to integer copy numbers
    ploidy_choices = [None, 2, 3, 4]
    outprefix = ["", "_diploid", "_triploid", "_tetraploid"]
    for o,pld in enumerate(ploidy_choices):
        # A/B copy number per bin
        allele_specific_copy = []
        # A/B copy number per state
        state_cnv = []
        # A/B copy number per gene
        df_genelevel_cnv = []

        # a test version on inferring integer copy numbers jointly across all clones
        lambd = np.sum(base_nb_mean,axis=1) / np.sum(base_nb_mean)
        pred_cnv = np.vstack([ res["pred_cnv"][(s*n_obs):(s*n_obs+n_obs)] for s in range(X.shape[2])]).T # size of (n_bins, n_clones)
        adjusted_log_mu = np.vstack([ np.log(np.exp(res["new_log_mu"][:,0]) / np.sum(np.exp(res["new_log_mu"][pred_cnv[:,s],0]) * lambd)) for s in range(X.shape[2]) ]).T # size of (n_states, n_clones)
        if not pld is None:
            best_integer_copies, _ = hill_climbing_integer_copynumber_joint(adjusted_log_mu, base_nb_mean, res["new_p_binom"][:,0], pred_cnv, max_medploidy=pld)
        else:
            best_integer_copies, _ = hill_climbing_integer_copynumber_joint(adjusted_log_mu, base_nb_mean, res["new_p_binom"][:,0], pred_cnv)
        # save inferred integer copy numbers to above defined variables
        # A/B copy number per bin
        for s, cid in enumerate(unique_clone_ids):
            allele_specific_copy.append( pd.DataFrame( best_integer_copies[pred_cnv[:,s], 0].reshape(1,-1), index=[f"clone{cid} A"], columns=np.arange(n_obs) ) )
            allele_specific_copy.append( pd.DataFrame( best_integer_copies[pred_cnv[:,s], 1].reshape(1,-1), index=[f"clone{cid} B"], columns=np.arange(n_obs) ) )
        # A/B copy number per state
        state_cnv.append( pd.DataFrame( res["new_log_mu"][:,0].reshape(-1,1), columns=[f"logmu"], index=np.arange(config['n_states']) ) )
        state_cnv.append( pd.DataFrame( res["new_p_binom"][:,0].reshape(-1,1), columns=[f"p"], index=np.arange(config['n_states']) ) )
        state_cnv.append( pd.DataFrame( best_integer_copies[:,0].reshape(-1,1), columns=[f"A"], index=np.arange(config['n_states']) ) )
        state_cnv.append( pd.DataFrame( best_integer_copies[:,1].reshape(-1,1), columns=[f"B"], index=np.arange(config['n_states']) ) )
        # A/B copy number per gene
        for s, cid in enumerate(unique_clone_ids):
            tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[pred_cnv[:,s], 0], best_integer_copies[pred_cnv[:,s], 1], x_gene_list)
            tmpdf.columns = [f"clone{cid} A", f"clone{cid} B"]
            df_genelevel_cnv.append(tmpdf)
            
        # for s, cid in enumerate(unique_clone_ids):
        #     if np.sum(base_nb_mean[:,s]) == 0:
        #         continue
        #     # adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1
        #     lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
        #     this_pred_cnv = res["pred_cnv"][(s*n_obs):(s*n_obs+n_obs)]
        #     adjusted_log_mu = np.log( np.exp(res["new_log_mu"][:,0]) / np.sum(np.exp(res["new_log_mu"][this_pred_cnv,0]) * lambd) )
        #     # infer A/B integer copy numbers by hill_climbing_integer_copynumber_oneclone function
        #     if not pld is None:
        #         best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res["new_p_binom"][:,0], this_pred_cnv, max_medploidy=pld)
        #     else:
        #         best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res["new_p_binom"][:,0], this_pred_cnv)
        #     print(f"max med ploidy = {pld}, clone {s}, integer copy inference loss = {_}")
        #     # save inferred integer copy numbers to above defined variables
        #     # A/B copy number per bin
        #     allele_specific_copy.append( pd.DataFrame( best_integer_copies[this_pred_cnv, 0].reshape(1,-1), index=[f"clone{cid} A"], columns=np.arange(n_obs) ) )
        #     allele_specific_copy.append( pd.DataFrame( best_integer_copies[this_pred_cnv, 1].reshape(1,-1), index=[f"clone{cid} B"], columns=np.arange(n_obs) ) )
        #     # A/B copy number per state
        #     state_cnv.append( pd.DataFrame( res["new_log_mu"][:,0].reshape(-1,1), columns=[f"clone{cid} logmu"], index=np.arange(config['n_states']) ) )
        #     state_cnv.append( pd.DataFrame( res["new_p_binom"][:,0].reshape(-1,1), columns=[f"clone{cid} p"], index=np.arange(config['n_states']) ) )
        #     state_cnv.append( pd.DataFrame( best_integer_copies[:,0].reshape(-1,1), columns=[f"clone{cid} A"], index=np.arange(config['n_states']) ) )
        #     state_cnv.append( pd.DataFrame( best_integer_copies[:,1].reshape(-1,1), columns=[f"clone{cid} B"], index=np.arange(config['n_states']) ) )
        #     # A/B copy number per gene
        #     tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[this_pred_cnv, 0], best_integer_copies[this_pred_cnv, 1], x_gene_list)
        #     tmpdf.columns = [f"clone{cid} A", f"clone{cid} B"]
        #     df_genelevel_cnv.append(tmpdf)
        
        # reorganize to make a dataframe and save
        # A/B copy number per bin
        allele_specific_copy = pd.concat(allele_specific_copy)
        df_bininfo[["CHR", "START", "END"]].join( allele_specific_copy.T ).to_csv(f"{outdir}/cnv{outprefix[o]}_seglevel.tsv", header=True, index=False, sep="\t")
        # A/B copy number per state
        state_cnv = functools.reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True, how='inner'), state_cnv)
        state_cnv.to_csv(f"{outdir}/cnv{outprefix[o]}_perstate.tsv", header=True, index=False, sep="\t")
        # A/B copy number per gene
        df_genelevel_cnv = functools.reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True, how='inner'), df_genelevel_cnv)
        df_genelevel_cnv.to_csv(f"{outdir}/cnv{outprefix[o]}_genelevel.tsv", header=True, index=False, sep="\t")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])