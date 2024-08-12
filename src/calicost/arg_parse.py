import sys
import numpy as np
import scipy
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def load_default_config():
    config_joint = {"input_filelist": None, "alignment_files": []}
    config_single = {"spaceranger_dir": None}
    config_shared = {
        "snp_dir": None,
        "output_dir": None,
        # supporting files and preprocessing arguments
        "geneticmap_file": None,
        "hgtable_file": None,
        "normalidx_file": None,
        "tumorprop_file": None,
        "supervision_clone_file": None,
        "filtergenelist_file": None,
        "filterregion_file": None,
        "secondary_min_umi": 300,
        "min_snpumi_perspot": 50,
        "min_percent_expressed_spots": 0.005,
        "bafonly": False,
        # phase switch probability
        "nu": 1.0,
        "logphase_shift": -2.0,
        "npart_phasing": 3,
        # HMRF configurations
        "n_clones": None,
        "n_clones_rdr": 2,
        "min_spots_per_clone": 100,
        "min_avgumi_per_clone": 10,
        "maxspots_pooling": 7,
        "tumorprop_threshold": 0.5,
        "max_iter_outer_initial" : 20,
        "max_iter_outer": 10,
        "nodepotential": "weighted_sum",  # max or weighted_sum
        "initialization_method": "rectangle",  # rectangle or datadrive
        "num_hmrf_initialization_start": 0,
        "num_hmrf_initialization_end": 10,
        "spatial_weight": 1.0,
        "construct_adjacency_method": "hexagon",
        "construct_adjacency_w": 1.0,
        # HMM configurations
        "n_states": None,
        "params": "smp",
        "t": 1 - 1e-5,
        "t_phaseing": 1 - 1e-4,
        "fix_NB_dispersion": False,
        "shared_NB_dispersion": True,
        "fix_BB_dispersion": False,
        "shared_BB_dispersion": True,
        "max_iter": 30,
        "tol": 1e-4,
        "gmm_random_state": 0,
        "np_threshold": 1.0,
        "np_eventminlen": 10,
        # integer copy number
        "nonbalance_bafdist": 1.0,
        "nondiploid_rdrdist": 10.0,
    }

    argtype_joint = {"input_filelist": "str", "alignment_files": "list_str"}
    argtype_single = {"spaceranger_dir": "str"}
    argtype_shared = {
        "snp_dir": "str",
        "output_dir": "str",
        # supporting files and preprocessing arguments
        "geneticmap_file": "str",
        "hgtable_file": "str",
        "normalidx_file": "str",
        "tumorprop_file": "str",
        "supervision_clone_file": "str",
        "filtergenelist_file": "str",
        "filterregion_file": "str",
        "secondary_min_umi": "int",
        "min_snpumi_perspot": "int",
        "min_percent_expressed_spots": "float",
        "bafonly": "bool",
        # phase switch probability
        "nu": "float",
        "logphase_shift": "float",
        "npart_phasing": "int",
        # HMRF configurations
        "n_clones": "int",
        "n_clones_rdr": "int",
        "min_spots_per_clone": "int",
        "min_avgumi_per_clone": "int",
        "maxspots_pooling": "int",
        "tumorprop_threshold": "float",
        "max_iter_outer_initial" : "int",
        "max_iter_outer": "int",
        "nodepotential": "str",
        "initialization_method": "str",
        "num_hmrf_initialization_start": "int",
        "num_hmrf_initialization_end": "int",
        "spatial_weight": "float",
        "construct_adjacency_method": "str",
        "construct_adjacency_w": "float",
        # HMM configurations
        "n_states": "int",
        "params": "str",
        "t": "eval",
        "t_phaseing": "eval",
        "fix_NB_dispersion": "bool",
        "shared_NB_dispersion": "bool",
        "fix_BB_dispersion": "bool",
        "shared_BB_dispersion": "bool",
        "max_iter": "int",
        "tol": "float",
        "gmm_random_state": "int",
        "np_threshold": "float",
        "np_eventminlen": "int",
        # integer copy number
        "nonbalance_bafdist": "float",
        "nondiploid_rdrdist": "float",
    }

    category_names = [
        "",
        "# supporting files and preprocessing arguments",
        "# phase switch probability",
        "# HMRF configurations",
        "# HMM configurations",
        "# integer copy number",
    ]
    category_elements = [
        ["input_filelist", "spaceranger_dir", "snp_dir", "output_dir"],
        [
            "geneticmap_file",
            "hgtable_file",
            "normalidx_file",
            "tumorprop_file",
            "alignment_files",
            "supervision_clone_file",
            "filtergenelist_file",
            "filterregion_file",
            "secondary_min_umi",
            "min_snpumi_perspot",
            "min_percent_expressed_spots",
            "bafonly",
        ],
        ["nu", "logphase_shift", "npart_phasing"],
        [
            "n_clones",
            "n_clones_rdr",
            "min_spots_per_clone",
            "min_avgumi_per_clone",
            "maxspots_pooling",
            "tumorprop_threshold",
            "max_iter_outer_initial",
            "max_iter_outer",
            "nodepotential",
            "initialization_method",
            "num_hmrf_initialization_start",
            "num_hmrf_initialization_end",
            "spatial_weight",
            "construct_adjacency_method",
            "construct_adjacency_w",
        ],
        [
            "n_states",
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
            "np_eventminlen",
        ],
        ["nonbalance_bafdist", "nondiploid_rdrdist"],
    ]
    return (
        config_shared,
        config_joint,
        config_single,
        argtype_shared,
        argtype_joint,
        argtype_single,
        category_names,
        category_elements,
    )


def read_configuration_file(filename):
    ##### [Default settings] #####
    (
        config_shared,
        config_joint,
        config_single,
        argtype_shared,
        argtype_joint,
        argtype_single,
        _,
        _,
    ) = load_default_config()
    config = {**config_shared, **config_single}
    argument_type = {**argtype_shared, **argtype_single}

    ##### [ read configuration file to update settings ] #####
    with open(filename, "r") as fp:
        for line in fp:
            if line.strip() == "" or line[0] == "#":
                continue
            strs = [x.strip() for x in line.strip().split(":") if x != ""]
            # assert strs[0] in config.keys(), f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
            if (not strs[0] in config.keys()) and (not strs[0] in config_joint.keys()):
                # warning that the argument is not a valid configuration parameter and continue
                logger.warning(
                    f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
                )
                continue
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
                config[strs[0]] = strs[1].upper() == "TRUE"
            elif argument_type[strs[0]] == "list_str":
                config[strs[0]] = strs[1].split(" ")
    # assertions
    assert not config["spaceranger_dir"] is None, "No spaceranger directory!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    return config


def read_joint_configuration_file(filename):
    ##### [Default settings] #####
    (
        config_shared,
        config_joint,
        config_single,
        argtype_shared,
        argtype_joint,
        argtype_single,
        _,
        _,
    ) = load_default_config()
    config = {**config_shared, **config_joint}
    argument_type = {**argtype_shared, **argtype_joint}

    ##### [ read configuration file to update settings ] #####
    with open(filename, "r") as fp:
        for line in fp:
            if line.strip() == "" or line[0] == "#":
                continue
            strs = [x.strip() for x in line.strip().split(":") if x != ""]
            # assert strs[0] in config.keys(), f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
            if (not strs[0] in config.keys()) and (not strs[0] in config_single.keys()):
                # warning that the argument is not a valid configuration parameter and continue
                logger.warning(
                    f"{strs[0]} is not a valid configuration parameter! Configuration parameters are: {list(config.keys())}"
                )
                continue
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
                config[strs[0]] = strs[1].upper() == "TRUE"
            elif argument_type[strs[0]] == "list_str":
                config[strs[0]] = strs[1].split(" ")
    # assertions
    assert not config["input_filelist"] is None, "No input file list!"
    assert not config["snp_dir"] is None, "No SNP directory!"
    assert not config["output_dir"] is None, "No output directory!"

    return config


def write_config_file(outputfilename, config):
    (
        _,
        _,
        _,
        argtype_shared,
        argtype_joint,
        argtype_single,
        category_names,
        category_elements,
    ) = load_default_config()
    argument_type = {**argtype_shared, **argtype_joint, **argtype_single}
    with open(outputfilename, "w") as fp:
        for i in range(len(category_names)):
            fp.write(f"{category_names[i]}\n")
            for k in category_elements[i]:
                if k in config:
                    if argument_type[k] == "list_str":
                        fp.write(f"{k} : {' '.join(config[k])}\n")
                    else:
                        fp.write(f"{k} : {config[k]}\n")
            fp.write("\n")


def get_default_config_single():
    (
        config_shared,
        config_joint,
        config_single,
        argtype_shared,
        argtype_joint,
        argtype_single,
        _,
        _,
    ) = load_default_config()
    config = {**config_shared, **config_single}
    return config


def get_default_config_joint():
    (
        config_shared,
        config_joint,
        config_single,
        argtype_shared,
        argtype_joint,
        argtype_single,
        _,
        _,
    ) = load_default_config()
    config = {**config_shared, **config_joint}
    return config


def main(argv):
    template_configuration_file = argv[1]
    outputdir = argv[2]
    hmrf_seed_s = int(argv[3])
    hmrf_seed_t = int(argv[4])
    try:
        config = read_configuration_file(template_configuration_file)
    except:
        config = read_joint_configuration_file(template_configuration_file)

    for r in range(hmrf_seed_s, hmrf_seed_t):
        config["num_hmrf_initialization_start"] = r
        config["num_hmrf_initialization_end"] = r + 1
        write_config_file(f"{outputdir}/configfile{r}", config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv)
