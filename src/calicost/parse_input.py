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
import functools
import subprocess
import argparse
from calicost.utils_IO import *
from calicost.phasing import *
from calicost.arg_parse import *

logger = logging.getLogger(__name__)

def genesnp_to_bininfo(df_gene_snp):
    table_bininfo = df_gene_snp[~df_gene_snp.bin_id.isnull()].groupby('bin_id').agg({"CHR":'first', 'START':'first', 'END':'last', 'gene':set, 'snp_id':set}).reset_index()
    table_bininfo['ARM'] = '.'
    table_bininfo['INCLUDED_GENES'] = [ " ".join([x for x in y if not x is None]) for y in table_bininfo.gene.values ]
    table_bininfo['INCLUDED_SNP_IDS'] = [ " ".join([x for x in y if not x is None]) for y in table_bininfo.snp_id.values ]
    table_bininfo['NORMAL_COUNT'] = np.nan
    table_bininfo['N_SNPS'] = [ len([x for x in y if not x is None]) for y in table_bininfo.snp_id.values ]
    # drop the set columns
    table_bininfo.drop(columns=['gene', 'snp_id'], inplace=True)
    return table_bininfo


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
        adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids, across_slice_adjacency_mat = load_joint_data(config["input_filelist"], config["snp_dir"], config["alignment_files"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"], config['min_snpumi_perspot'], config['min_percent_expressed_spots'])
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
        adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids = load_data(config["spaceranger_dir"], config["snp_dir"], config["filtergenelist_file"], config["filterregion_file"], config["normalidx_file"], config['min_snpumi_perspot'], config['min_percent_expressed_spots'])
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
    df_gene_snp = combine_gene_snps(unique_snp_ids, config['hgtable_file'], adata)
    df_gene_snp = create_haplotype_block_ranges(df_gene_snp, adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids)
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat = summarize_counts_for_blocks(df_gene_snp, \
            adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids, nu=config['nu'], logphase_shift=config['logphase_shift'], geneticmap_file=config['geneticmap_file'])
    # infer an initial phase using pseudobulk
    if not Path(f"{config['output_dir']}/initial_phase.npz").exists():
        initial_clone_for_phasing = perform_partition(coords, sample_ids, x_part=config["npart_phasing"], y_part=config["npart_phasing"], single_tumor_prop=single_tumor_prop, threshold=config["tumorprop_threshold"])
        phase_indicator, refined_lengths = initial_phase_given_partition(single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_for_phasing, 5, log_sitewise_transmat, \
            "sp", config["t_phaseing"], config["gmm_random_state"], config["fix_NB_dispersion"], config["shared_NB_dispersion"], config["fix_BB_dispersion"], config["shared_BB_dispersion"], 30, 1e-3, threshold=config["tumorprop_threshold"])
        np.savez(f"{config['output_dir']}/initial_phase.npz", phase_indicator=phase_indicator, refined_lengths=refined_lengths)
        # map phase indicator to individual snps
        df_gene_snp['phase'] = np.where(df_gene_snp.snp_id.isnull(), None, df_gene_snp.block_id.map({i:x for i,x in enumerate(phase_indicator)}) )
    else:
        tmp = dict(np.load(f"{config['output_dir']}/initial_phase.npz"))
        phase_indicator, refined_lengths = tmp["phase_indicator"], tmp["refined_lengths"]

    # binning
    df_gene_snp = create_bin_ranges(df_gene_snp, single_total_bb_RD, refined_lengths, config['secondary_min_umi'])
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat = summarize_counts_for_bins(df_gene_snp, \
            adata, single_X, single_total_bb_RD, phase_indicator, nu=config['nu'], logphase_shift=config['logphase_shift'], geneticmap_file=config['geneticmap_file'])
    # lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps = perform_binning_new(lengths, single_X, \
    #     single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps, phase_indicator, refined_lengths, config["binsize"], config["rdrbinsize"], config["nu"], config["logphase_shift"], secondary_min_umi=secondary_min_umi)
        
    # # remove bins where normal spots have imbalanced SNPs
    # if not config["tumorprop_file"] is None:
    #     for prop_threshold in np.arange(0, 0.6, 0.05):
    #         normal_candidate = (single_tumor_prop <= prop_threshold)
    #         if np.sum(single_X[:, 0, (normal_candidate==True)]) > single_X.shape[0] * 200:
    #             break
    #     index_normal = np.where(normal_candidate)[0]
    #     lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_gene_snp = bin_selection_basedon_normal(df_gene_snp, \
    #             single_X, single_base_nb_mean, single_total_bb_RD, config["nu"], config["logphase_shift"], index_normal, config['geneticmap_file'])
    #     assert np.sum(lengths) == single_X.shape[0] 
    #     assert single_X.shape[0] == single_total_bb_RD.shape[0]
    #     assert single_X.shape[0] == len(log_sitewise_transmat)

    # expression count dataframe
    exp_counts = pd.DataFrame.sparse.from_spmatrix( scipy.sparse.csc_matrix(adata.layers["count"]), index=adata.obs.index, columns=adata.var.index)

    # smooth and adjacency matrix for each sample
    adjacency_mat, smooth_mat = multislice_adjacency(sample_ids, sample_list, coords, single_total_bb_RD, exp_counts, 
                                                     across_slice_adjacency_mat, construct_adjacency_method=config['construct_adjacency_method'], 
                                                     maxspots_pooling=config['maxspots_pooling'], construct_adjacency_w=config['construct_adjacency_w'])
    n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())
    print(f"Set up number of spots to pool in HMRF: {n_pooled}")

    # If adjacency matrix is only constructed using gene expression similarity (e.g. scRNA-seq data)
    # Then, directly replace coords by the umap of gene expression, to avoid potential inconsistency in HMRF initialization
    if config["construct_adjacency_method"] == "KNN" and config["construct_adjacency_w"] == 0:
        sc.pp.normalize_total(adata, target_sum=np.median(np.sum(exp_counts.values,axis=1)) )
        sc.pp.log1p(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        coords = adata.obsm["X_umap"]

    # create RDR-BAF table
    table_bininfo = genesnp_to_bininfo(df_gene_snp)
    table_bininfo['LOG_PHASE_TRANSITION'] = log_sitewise_transmat

    table_rdrbaf = []
    for i in range(single_X.shape[2]):
        table_rdrbaf.append( pd.DataFrame({"BARCODES":adata.obs.index[i], "EXP":single_X[:,0,i], "TOT":single_total_bb_RD[:,i], "B":single_X[:,1,i]}) )
    table_rdrbaf = pd.concat(table_rdrbaf, ignore_index=True)

    # create meta info table
    # note that table_meta.BARCODES is equal to the unique ones of table_rdrbaf.BARCODES in the original order
    table_meta = pd.DataFrame({"BARCODES":adata.obs.index, "SAMPLE":adata.obs["sample"], "X":coords[:,0], "Y":coords[:,1]})
    if not single_tumor_prop is None:
        table_meta["TUMOR_PROPORTION"] = single_tumor_prop
    
    return table_bininfo, table_rdrbaf, table_meta, exp_counts, adjacency_mat, smooth_mat, df_gene_snp


def load_tables_to_matrices(config):
    """
    Load tables and adjacency from parse_visium_joint or parse_visium_single, and convert to HMM input matrices.
    """
    table_bininfo = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_bininfo.csv.gz", header=0, index_col=None, sep="\t")
    table_rdrbaf = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_rdrbaf.csv.gz", header=0, index_col=None, sep="\t")
    table_meta = pd.read_csv(f"{config['output_dir']}/parsed_inputs/table_meta.csv.gz", header=0, index_col=None, sep="\t")
    adjacency_mat = scipy.sparse.load_npz( f"{config['output_dir']}/parsed_inputs/adjacency_mat.npz" )
    smooth_mat = scipy.sparse.load_npz( f"{config['output_dir']}/parsed_inputs/smooth_mat.npz" )
    #
    df_gene_snp = pd.read_csv(f"{config['output_dir']}/parsed_inputs/gene_snp_info.csv.gz", header=0, index_col=None, sep="\t")
    df_gene_snp = df_gene_snp.replace(np.nan, None)
    
    n_spots = table_meta.shape[0]
    n_bins = table_bininfo.shape[0]

    # construct single_X
    # single_X = np.zeros((n_bins, 2, n_spots), dtype=int)
    single_X = np.zeros((n_bins, 2, n_spots))
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
    lengths = np.array([ np.sum(table_bininfo.CHR == c) for c in df_bininfo.CHR.unique() ])
    
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

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
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
        table_bininfo, table_rdrbaf, table_meta, exp_counts, adjacency_mat, smooth_mat, df_gene_snp = parse_visium(config)
        # table_bininfo, table_rdrbaf, table_meta, exp_counts, adjacency_mat, smooth_mat = parse_hatchetblock(config, cellsnplite_dir, bb_file)

        # save file
        p = subprocess.Popen(f"mkdir -p {config['output_dir']}/parsed_inputs", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = p.communicate()
        
        table_bininfo.to_csv( f"{config['output_dir']}/parsed_inputs/table_bininfo.csv.gz", header=True, index=False, sep="\t" )
        table_rdrbaf.to_csv( f"{config['output_dir']}/parsed_inputs/table_rdrbaf.csv.gz", header=True, index=False, sep="\t" )
        table_meta.to_csv( f"{config['output_dir']}/parsed_inputs/table_meta.csv.gz", header=True, index=False, sep="\t" )
        exp_counts.to_pickle( f"{config['output_dir']}/parsed_inputs/exp_counts.pkl" )
        scipy.sparse.save_npz( f"{config['output_dir']}/parsed_inputs/adjacency_mat.npz", adjacency_mat )
        scipy.sparse.save_npz( f"{config['output_dir']}/parsed_inputs/smooth_mat.npz", smooth_mat )
        #
        df_gene_snp.to_csv( f"{config['output_dir']}/parsed_inputs/gene_snp_info.csv.gz", header=True, index=False, sep="\t" )

    # load and parse data
    return load_tables_to_matrices(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    args = parser.parse_args()

    try:
        config = read_configuration_file(args.configfile)
    except:
        config = read_joint_configuration_file(args.configfile)

    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")

    _ = run_parse_n_load(config)
