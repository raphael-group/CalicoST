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
from utils_distribution_fitting import *
from hmrf import *
from utils_IO import *


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
        "min_avgumi_per_clone" : 10,
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
        "t_phaseing" : 1-1e-4,
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
        "min_avgumi_per_clone" : "int",
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
        "t_phaseing" : "eval",
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



def write_joint_config_file(outputfilename, config):
    list_argument_io = ["input_filelist",
        "snp_dir",
        "output_dir"]
    list_argument_sup = ["hgtable_file",
        "normalidx_file",
        "tumorprop_file",
        "supervision_clone_file",
        "alignment_files",
        "filtergenelist_file",
        "filterregion_file",
        "binsize",
        "rdrbinsize",
        # "secondbinning_min_umi",
        "max_nbins",
        "avg_umi_perbinspot",
        "bafonly"]
    list_argument_phase = ["nu",
        "logphase_shift",
        "npart_phasing"]
    list_argument_hmrf = ["n_clones",
        "n_clones_rdr",
        "min_spots_per_clone",
        "min_avgumi_per_clone",
        "maxspots_pooling",
        "tumorprop_threshold",
        "max_iter_outer",
        "nodepotential",
        "initialization_method",
        "num_hmrf_initialization_start", 
        "num_hmrf_initialization_end",
        "spatial_weight",
        "construct_adjacency_method",
        "construct_adjacency_w"]
    list_argument_hmm = ["n_states",
        "params",
        "t",
        "t_phaseing",
        "fix_NB_dispersion",
        "shared_NB_dispersion",
        "fix_BB_dispersion",
        "shared_BB_dispersion",
        "max_iter",
        "tol",
        "gmm_random_state",
        "np_threshold",
        "np_eventminlen"]
    with open(outputfilename, 'w') as fp:
        #
        for k in list_argument_io:
            fp.write(f"{k} : {config[k]}\n")
        #
        fp.write("\n")
        fp.write("# supporting files and preprocessing arguments\n")
        for k in list_argument_sup:
            if not isinstance(config[k], list):
                fp.write(f"{k} : {config[k]}\n")
            else:
                fp.write(f"{k} : " + " ".join(config[k]) + "\n")
        #
        fp.write("\n")
        fp.write("# phase switch probability\n")
        for k in list_argument_phase:
            fp.write(f"{k} : {config[k]}\n")
        #
        fp.write("\n")
        fp.write("# HMRF configurations\n")
        for k in list_argument_hmrf:
            fp.write(f"{k} : {config[k]}\n")
        #
        fp.write("\n")
        fp.write("# HMM configurations\n")
        for k in list_argument_hmm:
            fp.write(f"{k} : {config[k]}\n")


def main(argv):
    template_configuration_file = argv[1]
    outputdir = argv[2]
    hmrf_seed_s = int(argv[3])
    hmrf_seed_t = int(argv[4])
    config = read_joint_configuration_file(template_configuration_file)
    for r in range(hmrf_seed_s, hmrf_seed_t):
        config["num_hmrf_initialization_start"] = r
        config["num_hmrf_initialization_end"] = r+1
        write_joint_config_file(f"{outputdir}/configfile{r}", config)
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python joint_allele_generateconfig.py <template_configuration_file> <outputdir> <hmrf_seed_s> <hmrf_seed_t>")
    if len(sys.argv) > 1:
        main(sys.argv)