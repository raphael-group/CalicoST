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
from utils_IO import *
from phasing import *


def parse_visium(config):
    """
    Read multiple 10X Visium SRT samples and SNP data and generate tables with counts and meta info.
    
    Attributes:
    ----------
    config : dictionary
        Dictionary containing configuration parameters. Output from read_joint_configuration_file.

    Returns:
    ----------
    table_bininfo : DataFrame
        DataFrame with columns [chr, arm, start, end, log_phase_transition, included_genes, normal count, n_snps].

    table_rdrbaf : DataFrame
        DataFrame with columns [barcodes, exp_count, tot_count, b_count].

    meta_info : DataFrame
        DataFrame with columns [barcodes, sample, x, y, tumor_proportion]

    expression : sparse matrix, (n_spots, n_genes)
        Gene expression UMI count matrix.

    adjacency_mat : array, (n_spots, n_spots)
        Adjacency matrix for evaluating label coherence in HMRF.

    smooth_mat : array, (n_spots, n_spots)
        KNN smoothing matrix.
    """
    if "input_filelist" in config:
        adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, across_slice_adjacency_mat = load_joint_data(config["input_filelist"], config["snp_dir"], config["alignment_files"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"])
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
        adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids = load_data(config["spaceranger_dir"], config["snp_dir"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"])
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
    
    # read original data
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps = convert_to_hmm_input_new(adata, \
        cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, config["hgtable_file"], config["nu"], config["logphase_shift"])
    # infer an initial phase using pseudobulk
    if not Path(f"{config['output_dir']}/initial_phase.npz").exists():
        initial_clone_for_phasing = perform_partition(coords, sample_ids, x_part=config["npart_phasing"], y_part=config["npart_phasing"], single_tumor_prop=single_tumor_prop, threshold=config["tumorprop_threshold"])
        phase_indicator, refined_lengths = initial_phase_given_partition(single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_for_phasing, 5, log_sitewise_transmat, \
            "sp", 1-1e-4, config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], 30, 1e-3)
        np.savez(f"{config['output_dir']}/initial_phase.npz", phase_indicator=phase_indicator, refined_lengths=refined_lengths)
    else:
        tmp = dict(np.load(f"{config['output_dir']}/initial_phase.npz"))
        phase_indicator, refined_lengths = tmp["phase_indicator"], tmp["refined_lengths"]
    # binning with inferred phase
    expected_nbins = min(config["max_nbins"], np.median(np.sum(single_total_bb_RD, axis=0)) * config["maxspots_pooling"] / config["avg_umi_perbinspot"] )
    expected_nbins = int(expected_nbins)
    secondary_min_umi = choose_umithreshold_given_nbins(single_total_bb_RD, refined_lengths, expected_nbins)
    print(f"Secondary_min_umi = {secondary_min_umi}")
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps = perform_binning_new(lengths, single_X, \
        single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps, phase_indicator, refined_lengths, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"], secondary_min_umi=secondary_min_umi)
    
    # remove bins where normal spots have imbalanced SNPs
    if not config["tumorprop_file"] is None:
        index_normal = np.where(single_tumor_prop < 0.01)[0]
        lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, index_remaining = bin_selection_basedon_normal(single_X, \
                single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, config["nu"], config["logphase_shift"], index_normal)
        n_snps = [n_snps[i] for i in index_remaining]

    # create RDR-BAF table
    table_bininfo = pd.DataFrame({"CHR":[x[0] for x in sorted_chr_pos], \
                                 "ARM":".", \
                                 "START":[x[1] for x in sorted_chr_pos], \
                                 "END":[x[1] for x in sorted_chr_pos_last], \
                                 "LOG_PHASE_TRANSITION": log_sitewise_transmat, \
                                 "INCLUDED_GENES":x_gene_list, \
                                 "NORMAL_COUNT":1e6 * np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean), \
                                 "N_SNPS":n_snps})
    table_rdrbaf = []
    for i in range(single_X.shape[2]):
        table_rdrbaf.append( pd.DataFrame({"BARCODES":adata.obs.index[i], "EXP":single_X[:,0,i], "TOT":single_total_bb_RD[:,i], "B":single_X[:,1,i]}) )
    table_rdrbaf = pd.concat(table_rdrbaf, ignore_index=True)

    # create meta info table
    # note that table_meta.BARCODES is equal to the unique ones of table_rdrbaf.BARCODES in the original order
    table_meta = pd.DataFrame({"BARCODES":adata.obs.index, "SAMPLE":adata.obs["sample"], "X":coords[:,0], "Y":coords[:,1]})
    if not single_tumor_prop is None:
        table_meta["TUMOR_PROPORTION"] = single_tumor_prop

    # expression count dataframe
    exp_counts = pd.DataFrame.sparse.from_spmatrix( scipy.sparse.csc_matrix(adata.layers["count"]), index=adata.obs.index, columns=adata.var.index)

    # smooth and adjacency matrix for each sample
    adjacency_mat = []
    smooth_mat = []
    for sname in sample_list:
        index = np.where(adata.obs["sample"] == sname)[0]
        this_coords = np.array(coords[index,:])
        if config["construct_adjacency_method"] == "hexagon":
            tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts(this_coords, single_total_bb_RD[:,index], maxspots_pooling=config["maxspots_pooling"])
        elif config["construct_adjacency_method"] == "KNN":
            tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_KNN(this_coords, exp_counts.iloc[index,:], w=config["construct_adjacency_w"], maxspots_pooling=config["maxspots_pooling"])
        else:
            logging.error("Unknown adjacency construction method")
        # tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts_slidedna(this_coords, maxspots_pooling=config["maxspots_pooling"])
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
    
    return table_bininfo, table_rdrbaf, table_meta, exp_counts, adjacency_mat, smooth_mat


def load_tables_to_matrices(config):
    """
    Load tables and adjacency from parse_visium_joint or parse_visium_single, and convert to HMM input matrices.
    """
    table_bininfo = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_bininfo.csv.gz", header=0, index_col=None, sep="\t")
    table_rdrbaf = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_rdrbaf.csv.gz", header=0, index_col=None, sep="\t")
    table_meta = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_meta.csv.gz", header=0, index_col=None, sep="\t")
    adjacency_mat = scipy.sparse.load_npz( f"{config['output_dir']}/parsed_inputs/adjacency_mat.npz" )
    smooth_mat = scipy.sparse.load_npz( f"{config['output_dir']}/parsed_inputs/smooth_mat.npz" )

    n_spots = table_meta.shape[0]
    n_bins = table_bininfo.shape[0]

    # construct single_X
    single_X = np.zeros((n_bins, 2, n_spots), dtype=int)
    single_X[:, 0, :] = table_rdrbaf["EXP"].values.reshape((n_bins, n_spots), order="F")
    single_X[:, 1, :] = table_rdrbaf["B"].values.reshape((n_bins, n_spots), order="F")

    # construct single_base_nb_mean, lengths
    single_base_nb_mean = table_bininfo["NORMAL_COUNT"].values.reshape(-1,1) / np.sum(table_bininfo["NORMAL_COUNT"].values) @ np.sum(single_X[:,0,:], axis=0).reshape(1,-1)

    # construct single_total_bb_RD
    single_total_bb_RD = table_rdrbaf["TOT"].values.reshape((n_bins, n_spots), order="F")

    # construct log_sitewise_transmat
    log_sitewise_transmat = table_bininfo["LOG_PHASE_TRANSITION"].values

    # construct bin info and lengths and x_gene_list
    df_bininfo = table_bininfo
    lengths = np.array([ np.sum(table_bininfo.CHR == c) for c in range(1, 23) ])
    x_gene_list = np.where(table_bininfo["INCLUDED_GENES"].isnull(), "", table_bininfo["INCLUDED_GENES"].values).astype(str)

    # construct barcodes
    barcodes = table_meta["BARCODES"]

    # construct coords
    coords = table_meta[["X", "Y"]].values

    # construct single_tumor_prop
    single_tumor_prop = table_meta["TUMOR_PROPORTION"].values if "TUMOR_PROPORTION" in table_meta.columns else None

    # construct sample_list and sample_ids
    sample_list = [table_meta["SAMPLE"].values[0]]
    for i in range(1, table_meta.shape[0]):
        if table_meta["SAMPLE"].values[i] != sample_list[-1]:
            sample_list.append( table_meta["SAMPLE"].values[i] )
    sample_ids = np.zeros(table_meta.shape[0], dtype=int)
    for s,sname in enumerate(sample_list):
        index = np.where(table_meta["SAMPLE"].values == sname)[0]
        sample_ids[index] = s

    # expression UMI count matrix
    exp_counts = pd.read_pickle( f"{config['output_dir']}/parsed_inputs/exp_counts.pkl" )

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, x_gene_list, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts


def run_parse_n_load(config):
    file_exists = np.array([ Path(f"{config['output_dir']}/parsed_inputs/table_bininfo.csv.gz").exists(), \
                             Path(f"{config['output_dir']}/parsed_inputs/table_rdrbaf.csv.gz").exists(), \
                             Path(f"{config['output_dir']}/parsed_inputs/table_meta.csv.gz").exists(), \
                             Path(f"{config['output_dir']}/parsed_inputs/adjacency_mat.npz").exists(), \
                             Path(f"{config['output_dir']}/parsed_inputs/smooth_mat.npz").exists(), \
                             Path(f"{config['output_dir']}/parsed_inputs/exp_counts.pkl").exists() ])
    if not np.all(file_exists):
        # process to tables
        table_bininfo, table_rdrbaf, table_meta, exp_counts, adjacency_mat, smooth_mat = parse_visium(config)

        # save file
        p = subprocess.Popen(f"mkdir -p {config['output_dir']}/parsed_inputs", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = p.communicate()
        
        table_bininfo.to_csv( f"{config['output_dir']}/parsed_inputs/table_bininfo.csv.gz", header=True, index=False, sep="\t" )
        table_rdrbaf.to_csv( f"{config['output_dir']}/parsed_inputs/table_rdrbaf.csv.gz", header=True, index=False, sep="\t" )
        table_meta.to_csv( f"{config['output_dir']}/parsed_inputs/table_meta.csv.gz", header=True, index=False, sep="\t" )
        exp_counts.to_pickle( f"{config['output_dir']}/parsed_inputs/exp_counts.pkl" )
        scipy.sparse.save_npz( f"{config['output_dir']}/parsed_inputs/adjacency_mat.npz", adjacency_mat )
        scipy.sparse.save_npz( f"{config['output_dir']}/parsed_inputs/smooth_mat.npz", smooth_mat )

    # load and parse data
    return load_tables_to_matrices(config)