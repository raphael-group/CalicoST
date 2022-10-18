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


def read_configuration_file(filename):
    ##### [Default settings] #####
    config = {
        "spaceranger_dir" : None,
        "snp_dir" : None,
        "output_dir" : None,
        # supporting files and preprocessing arguments
        "hgtable_file" : None,
        "normalidx_file" : None,
        "filtergenelist_file" : None,
        "rdrbinsize" : None,
        "bafonly" : True,
        # phase switch probability
        "nu" : 1,
        "logphase_shift" : 0,
        # HMRF configurations
        "n_clones" : None,
        "max_iter_outer" : 20,
        "nodepotential" : "max", # max or weighted_sum
        "num_hmrf_initialization" : 10, 
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
        "gmm_random_state" : 0
    }

    argument_type = {
        "spaceranger_dir" : "str",
        "snp_dir" : "str",
        "output_dir" : "str",
        # supporting files and preprocessing arguments
        "hgtable_file" : "str",
        "normalidx_file" : "str",
        "filtergenelist_file" : "str",
        "rdrbinsize" : "int",
        "bafonly" : "bool",
        # phase switch probability
        "nu" : "float",
        "logphase_shift" : "float",
        # HMRF configurations
        "n_clones" : "int",
        "max_iter_outer" : "int",
        "nodepotential" : "str",
        "num_hmrf_initialization" : "int",
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
        "gmm_random_state" : "int"
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
    assert not config["spaceranger_dir"] is None, "No spaceranger directory!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    return config


def main(configuration_file):
    config = read_configuration_file(configuration_file)
    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids = load_data(config["spaceranger_dir"], config["snp_dir"], config["filtergenelist_file"])
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos = convert_to_hmm_input(adata, cell_snp_Aallele, cell_snp_Ballele, \
        snp_gene_list, unique_snp_ids, config["rdrbinsize"], config["nu"], config["logphase_shift"], config["hgtable_file"], config["normalidx_file"])
    coords = adata.obsm["X_pos"]
    unique_chrs = np.arange(1, 23)
    if config["bafonly"]:
        single_X[:,0,:] = 0
        single_base_nb_mean[:,:] = 0
    # run HMRF
    for r_hmrf_initialization in range(config["num_hmrf_initialization"]):
        outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
        initial_clone_index = rectangle_initialize_initial_clone(coords, config["n_clones"], random_state=r_hmrf_initialization)
        # create directory
        p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = p.communicate()
        # run HMRF + HMM
        hmrf_pipeline(outdir, single_X, lengths, single_base_nb_mean, single_total_bb_RD, initial_clone_index, n_states=config["n_states"], \
            log_sitewise_transmat=log_sitewise_transmat, coords=coords, max_iter_outer=config["max_iter_outer"], nodepotential=config["nodepotential"], \
            params=config["params"], t=config["t"], random_state=config["gmm_random_state"], \
            fix_NB_dispersion=config["fix_NB_dispersion"], shared_NB_dispersion=config["shared_NB_dispersion"], \
            fix_BB_dispersion=config["fix_BB_dispersion"], shared_BB_dispersion=config["shared_BB_dispersion"], \
            is_diag=True, max_iter=config["max_iter"], tol=config["tol"], spatial_weight=config["spatial_weight"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])