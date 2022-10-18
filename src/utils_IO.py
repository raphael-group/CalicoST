import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import scanpy as sc
import anndata
from utils_phase_switch import *


def load_data(spaceranger_dir, snp_dir, filtergenelist_file):
    ##### read raw UMI count matrix #####
    adata = sc.read_10x_h5(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5")
    adata.layers["count"] = adata.X.A.astype(int)
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/round3-arrays/cell_snp_Aallele.npz")
    cell_snp_Aallele = cell_snp_Aallele.A
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/round3-arrays/cell_snp_Ballele.npz")
    cell_snp_Ballele = cell_snp_Ballele.A
    snp_gene_list = np.load(f"{snp_dir}/round3-arrays/snp_gene_list.npy", allow_pickle=True)
    unique_snp_ids = np.load(f"{snp_dir}/round3-arrays/unique_snp_ids.npy", allow_pickle=True)

    # filter out genes that are expressed in <0.5% cells
    indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    indicator = np.array([ (x in genenames) for x in snp_gene_list ])
    cell_snp_Aallele = cell_snp_Aallele[:, indicator]
    cell_snp_Ballele = cell_snp_Ballele[:, indicator]
    snp_gene_list = snp_gene_list[indicator]
    unique_snp_ids = unique_snp_ids[indicator]
    print("median UMI after filtering out genes < 0.5% of cells = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    # remove genes in filtergenelist_file
    # ig_gene_list = pd.read_csv("/n/fs/ragr-data/users/congma/references/cellranger_refdata-gex-GRCh38-2020-A/genes/ig_gene_list.txt", header=None)
    if not filtergenelist_file is None:
        filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
        filter_gene_list = set(list( filter_gene_list.iloc[:,0] ))
        indicator_fulter = np.array([ (not x in filter_gene_list) for x in adata.var.index ])
        adata = adata[:, indicator_fulter]
        indicator_fulter = np.array([ (x not in filter_gene_list) for x in snp_gene_list ])
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_fulter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_fulter]
        snp_gene_list = snp_gene_list[indicator_fulter]
        unique_snp_ids = unique_snp_ids[indicator_fulter]
        print("median UMI after filtering out genes in filtergenelist_file = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    # add position
    df_pos = pd.read_csv(f"{spaceranger_dir}/spatial/tissue_positions_list.csv", sep=",", header=None, \
                    names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    df_pos = df_pos[df_pos.in_tissue == True]
    assert set(list(df_pos.barcode)) == set(list(adata.obs.index))
    df_pos.barcode = pd.Categorical(df_pos.barcode, categories=list(adata.obs.index), ordered=True)
    df_pos.sort_values(by="barcode", inplace=True)
    adata.obsm["X_pos"] = np.vstack([df_pos.x, df_pos.y]).T

    # add position
    df_pos = pd.read_csv(f"{spaceranger_dir}/spatial/tissue_positions_list.csv", sep=",", header=None, \
                    names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    df_pos = df_pos[df_pos.in_tissue == True]
    assert set(list(df_pos.barcode)) == set(list(adata.obs.index))
    df_pos.barcode = pd.Categorical(df_pos.barcode, categories=list(adata.obs.index), ordered=True)
    df_pos.sort_values(by="barcode", inplace=True)
    adata.obsm["X_pos"] = np.vstack([df_pos.x, df_pos.y]).T

    return adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids


def convert_to_hmm_input(adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, binsize, nu, logphase_shift, hgtable_file, normalidx_file):
    uncovered_genes = set(list(adata.var.index)) - set(list(snp_gene_list))
    # df_hgtable = pd.read_csv("/u/congma/ragr-data/users/congma/Codes/STARCH_crazydev/hgTables_hg38_gencode.txt", header=0, index_col=0, sep="\t")
    df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
    df_hgtable = df_hgtable[df_hgtable.name2.isin(uncovered_genes)]
    df_hgtable = df_hgtable[df_hgtable.chrom.isin( [f"chr{i}" for i in range(1, 23)] )]
    uncovered_genes = df_hgtable.name2.values
    uncovered_genes_chr = [int(x[3:]) for x in df_hgtable.chrom]
    uncovered_genes_pos = df_hgtable.cdsStart.values

    # combining and sorting all SNPs and uncovered genes
    map_id = {(uncovered_genes_chr[i], uncovered_genes_pos[i]):uncovered_genes[i] for i in range(len(uncovered_genes_chr))}
    map_id.update( {(int(x.split("_")[0]), int(x.split("_")[1])):x for x in unique_snp_ids} )
    sorted_chr_pos = sorted( list(map_id.keys()) )
    gene_mapper = {adata.var.index[i]:i for i in range(adata.shape[1])}

    ##### fill-in HMM input matrices #####
    unique_chrs = np.arange(1, 23)
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    single_X = np.zeros((len(sorted_chr_pos), 2, adata.shape[0]), dtype=int)
    single_base_nb_mean = np.zeros((len(sorted_chr_pos), adata.shape[0]))
    single_total_bb_RD = np.zeros((len(sorted_chr_pos), adata.shape[0]), dtype=int)
    # BAF
    map_unique_snp_ids = {x:i for i,x in enumerate(unique_snp_ids)}
    idx_snps = np.array([i for i,x in enumerate(sorted_chr_pos) if map_id[x] in map_unique_snp_ids])
    single_X[idx_snps, 1, :] = cell_snp_Aallele.T
    single_total_bb_RD[idx_snps, :] = (cell_snp_Aallele + cell_snp_Ballele).T
    # RDR
    combined_snp_gene_list = [snp_gene_list[map_unique_snp_ids[map_id[x]]] if map_id[x] in map_unique_snp_ids else map_id[x] for i,x in enumerate(sorted_chr_pos)]
    rdridx_X = []
    rdridx_gene = []
    for i,g in enumerate(combined_snp_gene_list):
        if g != "":
            if i == 0 or g != combined_snp_gene_list[i-1]:
                rdridx_X.append(i)
                rdridx_gene.append( gene_mapper[g] )
    rdridx_X = np.array(rdridx_X)
    rdridx_gene = np.array(rdridx_gene)
    single_X[rdridx_X,0,:] = adata.layers["count"][:, rdridx_gene].T

    # diploid baseline
    if not normalidx_file is None:
        idx_normal = pd.read_csv(normalidx_file, header=None).iloc[:,0].values
        single_base_nb_mean = np.sum(single_X[:,0,:], axis=0, keepdims=True) * np.sum(single_X[:,0,idx_normal], axis=1, keepdims=True) / np.sum(single_X[:,0,idx_normal])
        assert np.sum(np.abs( np.sum(single_X[:,0,:],axis=0) - np.sum(single_base_nb_mean,axis=0) )) < 1e-4

    # phase switch probability from genetic distance
    position_cM = get_position_cM_table( sorted_chr_pos )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    # bin RDR
    for i in range(int(np.ceil(single_X.shape[0] / binsize))):
        single_X[(i*binsize):(i*binsize+binsize), 0, :] = np.sum(single_X[(i*binsize):(i*binsize+binsize), 0, :], axis=0)
        single_X[(i*binsize+1):(i*binsize+binsize), 0, :] = 0
        single_base_nb_mean[(i*binsize):(i*binsize+binsize), :] = np.sum(single_base_nb_mean[(i*binsize):(i*binsize+binsize), :], axis=0)
        single_base_nb_mean[(i*binsize+1):(i*binsize+binsize), :] = 0

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos
