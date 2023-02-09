import sys
import numpy as np
import scipy
import copy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import scanpy as sc
import anndata
from utils_phase_switch import *
import subprocess


def load_data(spaceranger_dir, snp_dir, filtergenelist_file, normalidx_file):
    ##### read raw UMI count matrix #####
    if Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5").exists():
        adata = sc.read_10x_h5(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5")
    elif Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad").exists():
        adata = sc.read_h5ad(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad")
    adata.layers["count"] = adata.X.A.astype(int)
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Aallele = cell_snp_Aallele.A
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    cell_snp_Ballele = cell_snp_Ballele.A
    snp_gene_list = np.load(f"{snp_dir}/snp_gene_list.npy", allow_pickle=True)
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)

    # add position
    df_pos = pd.read_csv(f"{spaceranger_dir}/spatial/tissue_positions_list.csv", sep=",", header=None, \
                    names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    df_pos = df_pos[df_pos.in_tissue == True]
    assert set(list(df_pos.barcode)) == set(list(adata.obs.index))
    df_pos.barcode = pd.Categorical(df_pos.barcode, categories=list(adata.obs.index), ordered=True)
    df_pos.sort_values(by="barcode", inplace=True)
    adata.obsm["X_pos"] = np.vstack([df_pos.x, df_pos.y]).T

    # filter out spots with too small number of UMIs
    indicator = (np.sum(adata.layers["count"], axis=1) > 100)
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out genes that are expressed in <0.5% cells
    indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    # indicator = np.array([ (x in genenames) for x in snp_gene_list ])
    # cell_snp_Aallele = cell_snp_Aallele[:, indicator]
    # cell_snp_Ballele = cell_snp_Ballele[:, indicator]
    # snp_gene_list = snp_gene_list[indicator]
    # unique_snp_ids = unique_snp_ids[indicator]
    print(adata)
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
    
    clf = LocalOutlierFactor(n_neighbors=200)
    label = clf.fit_predict( np.sum(adata.layers["count"], axis=0).reshape(-1,1) )
    adata.layers["count"][:, np.where(label==-1)[0]] = 0
    print("filter out {} outlier genes.".format( np.sum(label==-1) ))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:,0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        print( adata.obs["tumor_annotation"].value_counts() )
    
    return adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids


def load_joint_data(input_filelist, snp_dir, filtergenelist_file, normalidx_file):
    ##### read meta sample info #####
    df_meta = pd.read_csv(input_filelist, sep="\t", header=None, names=["bam", "sample_id", "spaceranger_dir"])
    df_barcode = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, names=["combined_barcode"])
    df_barcode["sample_id"] = [x.split("_")[-1] for x in df_barcode.combined_barcode.values]
    df_barcode["barcode"] = [x.split("_")[0] for x in df_barcode.combined_barcode.values]
    ##### read SNP count #####
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Aallele = cell_snp_Aallele.A
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    cell_snp_Ballele = cell_snp_Ballele.A
    snp_gene_list = np.load(f"{snp_dir}/snp_gene_list.npy", allow_pickle=True)
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    ##### read anndata and coordinate #####
    # add position
    adata = None
    for i,sname in enumerate(df_meta.sample_id.values):
        # locate the corresponding rows in df_meta
        index = np.where(df_barcode["sample_id"] == sname)[0]
        df_this_barcode = copy.copy(df_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode
        # read adata count info
        adatatmp = sc.read_10x_h5(f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5")
        adatatmp.layers["count"] = adatatmp.X.A
        # reorder anndata spots to have the same order as df_this_barcode
        idx_argsort = pd.Categorical(adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True).argsort()
        adatatmp = adatatmp[idx_argsort, :]
        # read position info
        df_this_pos = pd.read_csv(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv", sep=",", header=0, \
                    names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
        df_this_pos = df_this_pos[df_this_pos.in_tissue == True]
        df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(df_this_barcode.barcode), ordered=True)
        df_this_pos.sort_values(by="barcode", inplace=True)
        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]
        adatatmp.var_names_make_unique()
        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")
    
    # # filter out spots with too small number of UMIs
    # indicator = (np.sum(adata.layers["count"], axis=1) > 100)
    # adata = adata[indicator, :]
    # cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    # cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out genes that are expressed in <0.5% cells
    indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    # indicator = np.array([ (x in genenames) for x in snp_gene_list ])
    # cell_snp_Aallele = cell_snp_Aallele[:, indicator]
    # cell_snp_Ballele = cell_snp_Ballele[:, indicator]
    # snp_gene_list = snp_gene_list[indicator]
    # unique_snp_ids = unique_snp_ids[indicator]
    print(adata)
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

    clf = LocalOutlierFactor(n_neighbors=200)
    label = clf.fit_predict( np.sum(adata.layers["count"], axis=0).reshape(-1,1) )
    adata.layers["count"][:, np.where(label==-1)[0]] = 0
    print("filter out {} outlier genes.".format( np.sum(label==-1) ))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:,0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        print( adata.obs["tumor_annotation"].value_counts() )

    return adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids


def load_slidedna_data(snp_dir, bead_file, filterregion_bedfile):
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    barcodes = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, index_col=None)
    barcodes = barcodes.iloc[:,0].values
    # add spatial position
    df_pos = pd.read_csv(bead_file, header=0, sep=",", index_col=None)
    coords = np.vstack([df_pos.xcoord, df_pos.ycoord]).T
    # remove SNPs within filterregion_bedfile
    if not filterregion_bedfile is None:
        df_filter = pd.read_csv(filterregion_bedfile, header=None, sep="\t", names=["chrname", "start", "end"])
        df_filter = df_filter[df_filter.chrname.isin( [f"chr{i}" for i in range(1,23)] )]
        df_filter["CHR"] = [int(x[3:]) for x in df_filter.chrname]
        df_filter.sort_values(by=["CHR", "start"])
        # check whether unique_snp_ids are within the regions in df_filter
        snp_chrs = [int(x.split("_")[0]) for x in unique_snp_ids]
        snp_pos = [int(x.split("_")[1]) for x in unique_snp_ids]
        filter_chrs = df_filter.CHR.values
        filter_start = df_filter.start.values
        filter_end = df_filter.end.values
        is_within_filterregion = []
        j = 0
        for i in range(len(unique_snp_ids)):
            while (filter_chrs[j] < snp_chrs[i]) or ((filter_chrs[j] == snp_chrs[i]) and (filter_end[j] < snp_pos[i])):
                j += 1
            if filter_chrs[j] == snp_chrs[i] and filter_start[j] <= snp_pos[i] and filter_end[j] >= snp_pos[i]:
                is_within_filterregion.append(True)
            else:
                is_within_filterregion.append(False)
        is_within_filterregion = np.array(is_within_filterregion)
        # remove SNPs based on is_within_filterregion
        cell_snp_Aallele = cell_snp_Aallele[:, ~is_within_filterregion]
        cell_snp_Ballele = cell_snp_Ballele[:, ~is_within_filterregion]
        unique_snp_ids = unique_snp_ids[~is_within_filterregion]
    return coords, cell_snp_Aallele, cell_snp_Ballele, barcodes, unique_snp_ids


# def filter_slidedna_spot_by_adjacency(coords, cell_snp_Aallele, cell_snp_Ballele, barcodes):
#     # pairwise distance
#     x_dist = coords[:,0][None,:] - coords[:,0][:,None]
#     y_dist = coords[:,1][None,:] - coords[:,1][:,None]
#     pairwise_squared_dist = x_dist**2 + y_dist**2
#     np.fill_diagonal(pairwise_squared_dist, np.max(pairwise_squared_dist))
#     # radius to include 10 nearest neighbors
#     idx = np.argpartition(pairwise_squared_dist, kth=10, axis=1)[:,10]
#     radius = pairwise_squared_dist[(np.arange(pairwise_squared_dist.shape[0]), idx)]
#     idx_keep = (radius < np.mean(radius) + np.std(radius))
#     # remove spots
#     coords = coords[idx_keep, :]
#     cell_snp_Aallele = cell_snp_Aallele[idx_keep, :]
#     cell_snp_Ballele = cell_snp_Ballele[idx_keep, :]
#     barcodes = barcodes[idx_keep]
#     return coords, cell_snp_Aallele, cell_snp_Ballele, barcodes


def filter_slidedna_spot_by_adjacency(coords, cell_snp_Aallele, cell_snp_Ballele, barcodes):
    # distance to center
    dist = np.sqrt(np.sum(np.square(coords - np.median(coords, axis=0, keepdims=True)), axis=1))
    idx_keep = np.where(dist < 2500)[0]
    # remove spots
    coords = coords[idx_keep, :]
    cell_snp_Aallele = cell_snp_Aallele[idx_keep, :]
    cell_snp_Ballele = cell_snp_Ballele[idx_keep, :]
    barcodes = barcodes[idx_keep]
    return coords, cell_snp_Aallele, cell_snp_Ballele, barcodes


def convert_to_hmm_input(adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, binsize, rdrbinsize, nu, logphase_shift, hgtable_file, genome_build="hg38"):
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
    if "tumor_annotation" in adata.obs:
        idx_normal = np.where(adata.obs["tumor_annotation"] == "tumor")[0]
        idx_normal = idx_normal[np.arange(0, len(idx_normal), 2)] # TBD
        single_base_nb_mean = np.sum(single_X[:,0,:], axis=0, keepdims=True) * np.sum(single_X[:,0,idx_normal], axis=1, keepdims=True) / np.sum(single_X[:,0,idx_normal])
        assert np.sum(np.abs( np.sum(single_X[:,0,:],axis=0) - np.sum(single_base_nb_mean,axis=0) )) < 1e-4

    # bin both RDR and BAF
    bin_single_X = np.zeros((int(single_X.shape[0] / binsize), 2, single_X.shape[2]), dtype=int)
    bin_single_base_nb_mean = np.zeros((int(single_base_nb_mean.shape[0] / binsize), single_base_nb_mean.shape[1]))
    bin_single_total_bb_RD = np.zeros((int(single_total_bb_RD.shape[0] / binsize), single_total_bb_RD.shape[1]), dtype=int)
    bin_sorted_chr_pos = []
    for i in range(bin_single_X.shape[0]):
        bin_single_X[i,:,:] = np.sum(single_X[(i*binsize):(i*binsize+binsize), :, :], axis=0)
        bin_single_base_nb_mean[i,:] = np.sum(single_base_nb_mean[(i*binsize):(i*binsize+binsize), :], axis=0)
        bin_single_total_bb_RD[i,:] = np.sum(single_total_bb_RD[(i*binsize):(i*binsize+binsize), :], axis=0)
        bin_sorted_chr_pos.append( sorted_chr_pos[(i*binsize)] )
    single_X = bin_single_X
    single_base_nb_mean = bin_single_base_nb_mean
    single_total_bb_RD = bin_single_total_bb_RD
    sorted_chr_pos = bin_sorted_chr_pos

    # phase switch probability from genetic distance
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    # bin RDR
    for i in range(int(np.ceil(single_X.shape[0] / rdrbinsize))):
        single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :] = np.sum(single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :], axis=0)
        single_X[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), 0, :] = 0
        single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :] = np.sum(single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :], axis=0)
        single_base_nb_mean[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), :] = 0

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos


def convert_to_hmm_input_v2(adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, hgtable_file, nu, logphase_shift, genome_build="hg38"):
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
    single_X = np.zeros((len(sorted_chr_pos), 2, adata.shape[0]), dtype=int)
    single_base_nb_mean = np.zeros((len(sorted_chr_pos), adata.shape[0]))
    single_total_bb_RD = np.zeros((len(sorted_chr_pos), adata.shape[0]), dtype=int)
    # BAF
    map_unique_snp_ids = {x:i for i,x in enumerate(unique_snp_ids)}
    idx_snps = np.array([i for i,x in enumerate(sorted_chr_pos) if map_id[x] in map_unique_snp_ids])
    single_X[idx_snps, 1, :] = cell_snp_Aallele.T
    single_total_bb_RD[idx_snps, :] = (cell_snp_Aallele + cell_snp_Ballele).T
    # RDR
    x_gene_list = [""] * len(sorted_chr_pos) #!
    combined_snp_gene_list = [snp_gene_list[map_unique_snp_ids[map_id[x]]] if map_id[x] in map_unique_snp_ids else map_id[x] for i,x in enumerate(sorted_chr_pos)]
    rdridx_X = []
    rdridx_gene = []
    for i,g in enumerate(combined_snp_gene_list):
        if g != "" and g in gene_mapper:
            if i == 0 or g != combined_snp_gene_list[i-1]:
                rdridx_X.append(i)
                rdridx_gene.append( gene_mapper[g] )
                x_gene_list[i] = g #!
    rdridx_X = np.array(rdridx_X)
    rdridx_gene = np.array(rdridx_gene)
    single_X[rdridx_X,0,:] = adata.layers["count"][:, rdridx_gene].T
    # diploid baseline
    if "tumor_annotation" in adata.obs:
        idx_normal = np.where(adata.obs["tumor_annotation"] == "tumor")[0]
        idx_normal = idx_normal[np.arange(0, len(idx_normal), 2)] # TBD
        single_base_nb_mean = np.sum(single_X[:,0,:], axis=0, keepdims=True) * np.sum(single_X[:,0,idx_normal], axis=1, keepdims=True) / np.sum(single_X[:,0,idx_normal])
        assert np.sum(np.abs( np.sum(single_X[:,0,:],axis=0) - np.sum(single_base_nb_mean,axis=0) )) < 1e-4

    # bin both RDR and BAF
    tmpbinsize = 2
    bin_single_X = np.zeros((int(single_X.shape[0] / tmpbinsize), 2, single_X.shape[2]), dtype=int)
    bin_single_base_nb_mean = np.zeros((int(single_base_nb_mean.shape[0] / tmpbinsize), single_base_nb_mean.shape[1]))
    bin_single_total_bb_RD = np.zeros((int(single_total_bb_RD.shape[0] / tmpbinsize), single_total_bb_RD.shape[1]), dtype=int)
    bin_sorted_chr_pos = []
    bin_x_gene_list = []
    for i in range(bin_single_X.shape[0]):
        t = i*tmpbinsize+tmpbinsize if i + 1 < bin_single_X.shape[0] else single_X.shape[0]
        bin_single_X[i,:,:] = np.sum(single_X[(i*tmpbinsize):t, :, :], axis=0)
        bin_single_base_nb_mean[i,:] = np.sum(single_base_nb_mean[(i*tmpbinsize):t, :], axis=0)
        bin_single_total_bb_RD[i,:] = np.sum(single_total_bb_RD[(i*tmpbinsize):t, :], axis=0)
        bin_sorted_chr_pos.append( sorted_chr_pos[(i*tmpbinsize)] )
        this_genes = [g for g in x_gene_list[(i*tmpbinsize):t] if g != ""]
        bin_x_gene_list.append( " ".join(this_genes) )
    single_X = bin_single_X
    single_base_nb_mean = bin_single_base_nb_mean
    single_total_bb_RD = bin_single_total_bb_RD
    sorted_chr_pos = bin_sorted_chr_pos
    x_gene_list = bin_x_gene_list

    # phase switch probability from genetic distance
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list


def convert_to_hmm_input_slidedna(cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids, normalidx_file, nu, logphase_shift, snp_readcount_threshold=10, genome_build="hg38"):
    # choose reference-based phasing binsize
    tmpbinsize = snp_readcount_threshold / np.median(np.sum(cell_snp_Aallele, axis=0).A.flatten() + np.sum(cell_snp_Ballele, axis=0).A.flatten())
    tmpbinsize = max(tmpbinsize, 1.0)
    tmpbinsize = int(np.ceil(tmpbinsize))
    n_obs = cell_snp_Aallele.shape[1]
    n_spots = cell_snp_Aallele.shape[0]
    # get binned matrices
    row_ind = np.arange(n_obs)
    col_ind = np.zeros(n_obs, dtype=int)
    for i in range(int(n_obs / tmpbinsize)):
        col_ind[(i*tmpbinsize):(i*tmpbinsize + tmpbinsize)] = i
    multiplier = scipy.sparse.csr_matrix(( np.ones(len(row_ind),dtype=int), (row_ind, col_ind) ))
    bin_single_X = np.zeros((int(n_obs / tmpbinsize), 2, n_spots), dtype=int)
    bin_single_X[:,1,:] = (cell_snp_Aallele @ multiplier).T.A
    bin_single_base_nb_mean = np.zeros((int(n_obs / tmpbinsize), n_spots), dtype=int)
    bin_single_total_bb_RD = ((cell_snp_Aallele + cell_snp_Ballele) @ multiplier).T.A
    bin_sorted_chr_pos = []
    for i in range(int(n_obs / tmpbinsize)):
        bin_sorted_chr_pos.append( (int(unique_snp_ids[(i*tmpbinsize)].split("_")[0]), int(unique_snp_ids[(i*tmpbinsize)].split("_")[1])) )
    single_X = bin_single_X
    single_base_nb_mean = bin_single_base_nb_mean
    single_total_bb_RD = bin_single_total_bb_RD
    sorted_chr_pos = bin_sorted_chr_pos

    # phase switch probability from genetic distance
    unique_chrs = np.arange(1, 23)
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos


# def perform_binning(lengths, single_X, single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, x_gene_list, phase_prob, binsize, rdrbinsize, nu, logphase_shift, min_genes_perbin=5, genome_build="hg38"):
#     # bin both RDR and BAF
#     bin_single_X_rdr = []
#     bin_single_X_baf = []
#     bin_single_base_nb_mean = []
#     bin_single_total_bb_RD = []
#     bin_sorted_chr_pos = []
#     bin_x_gene_list = []
#     cumlen = 0
#     s = 0
#     for le in lengths:
#         while s < cumlen + le:
#             # initial bin with certain number of SNPs
#             t = min(s + binsize, cumlen + le)
#             # expand binsize by minimum number of genes
#             this_genes = set( sum([ x_gene_list[i].split(" ") for i in range(s,t) ], []) )
#             # while (t < cumlen + le) and (len(this_genes) < min_genes_perbin):
#             #     t += 1
#             #     this_genes = this_genes | set(x_gene_list[t].split(" "))
#             idx_A = np.where(phase_prob[s:t] >= 0.5)[0]
#             idx_B = np.where(phase_prob[s:t] < 0.5)[0]
#             bin_single_X_rdr.append( np.sum(single_X[s:t, 0, :], axis=0) )
#             bin_single_X_baf.append( np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0) )
#             bin_single_base_nb_mean.append( np.sum(single_base_nb_mean[s:t, :], axis=0) )
#             bin_single_total_bb_RD.append( np.sum(single_total_bb_RD[s:t, :], axis=0) )
#             bin_sorted_chr_pos.append( sorted_chr_pos[s] )
#             # this_genes = sum([ x_gene_list[i].split(" ") for i in range(s,t) ], [])
#             bin_x_gene_list.append( " ".join(this_genes) )
#             s = t
#         cumlen += le
#     single_X = np.stack([ np.vstack([bin_single_X_rdr[i], bin_single_X_baf[i]]) for i in range(len(bin_single_X_rdr)) ])
#     single_base_nb_mean = np.vstack(bin_single_base_nb_mean)
#     single_total_bb_RD = np.vstack(bin_single_total_bb_RD)
#     sorted_chr_pos = bin_sorted_chr_pos
#     x_gene_list = bin_x_gene_list

#     # phase switch probability from genetic distance
#     unique_chrs = np.arange(1, 23)
#     sorted_chr = np.array([x[0] for x in sorted_chr_pos])
#     lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
#     position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
#     phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
#     log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

#     # bin RDR
#     for i in range(int(np.ceil(single_X.shape[0] / rdrbinsize))):
#         single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :] = np.sum(single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :], axis=0)
#         single_X[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), 0, :] = 0
#         single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :] = np.sum(single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :], axis=0)
#         single_base_nb_mean[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), :] = 0

#     return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list


def perform_binning(lengths, single_X, single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, x_gene_list, phase_indicator, refined_lengths, binsize, rdrbinsize, nu, logphase_shift, min_genes_perbin=5, genome_build="hg38"):
    # bin both RDR and BAF
    bin_single_X_rdr = []
    bin_single_X_baf = []
    bin_single_base_nb_mean = []
    bin_single_total_bb_RD = []
    bin_sorted_chr_pos = []
    bin_x_gene_list = []
    cumlen = 0
    s = 0
    for le in refined_lengths:
        while s < cumlen + le:
            # initial bin with certain number of SNPs
            t = min(s + binsize, cumlen + le)
            # expand binsize by minimum number of genes
            this_genes = sum([ x_gene_list[i].split(" ") for i in range(s,t) ], [])
            this_genes = [z for z in this_genes if z!=""]
            # while (t < cumlen + le) and (len(this_genes) < min_genes_perbin):
            #     t += 1
            #     this_genes = this_genes | set(x_gene_list[t].split(" "))
            idx_A = np.where(phase_indicator[s:t])[0]
            idx_B = np.where(~phase_indicator[s:t])[0]
            bin_single_X_rdr.append( np.sum(single_X[s:t, 0, :], axis=0) )
            bin_single_X_baf.append( np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0) )
            bin_single_base_nb_mean.append( np.sum(single_base_nb_mean[s:t, :], axis=0) )
            bin_single_total_bb_RD.append( np.sum(single_total_bb_RD[s:t, :], axis=0) )
            bin_sorted_chr_pos.append( sorted_chr_pos[s] )
            # this_genes = sum([ x_gene_list[i].split(" ") for i in range(s,t) ], [])
            bin_x_gene_list.append( " ".join(this_genes) )
            s = t
        cumlen += le
    single_X = np.stack([ np.vstack([bin_single_X_rdr[i], bin_single_X_baf[i]]) for i in range(len(bin_single_X_rdr)) ])
    single_base_nb_mean = np.vstack(bin_single_base_nb_mean)
    single_total_bb_RD = np.vstack(bin_single_total_bb_RD)
    sorted_chr_pos = bin_sorted_chr_pos
    x_gene_list = bin_x_gene_list

    # phase switch probability from genetic distance
    unique_chrs = np.arange(1, 23)
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    # bin RDR
    for i in range(int(np.ceil(single_X.shape[0] / rdrbinsize))):
        single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :] = np.sum(single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :], axis=0)
        single_X[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), 0, :] = 0
        single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :] = np.sum(single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :], axis=0)
        single_base_nb_mean[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), :] = 0

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list


def filter_de_genes(adata, x_gene_list, logfcthreshold=4, quantile_threshold=80):
    assert "normal_candidate" in adata.obs
    # clone = np.array(["normal"] * adata.shape[0])
    # clone[~adata.obs["normal_candidate"]] = "tumor"
    tmpadata = adata.copy()
    # tmpadata.obs["clone"] = clone
    #
    map_gene_adatavar = {}
    map_gene_umi = {}
    for i,x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = np.sum(adata.layers["count"][:,i])
    #
    umi_threshold = np.percentile( np.sum(adata.layers["count"], axis=0), quantile_threshold )
    #
    sc.pp.filter_cells(tmpadata, min_genes=200)
    sc.pp.filter_genes(tmpadata, min_cells=10)
    #sc.pp.filter_genes(tmpadata, min_counts=200)
    sc.pp.normalize_total(tmpadata, target_sum=1e4)
    sc.pp.log1p(tmpadata)
    # new added
    sc.pp.pca(tmpadata, n_comps=4)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.X)
    kmeans_labels = kmeans.predict(tmpadata.X)
    idx_kmeans_label = np.argmax(np.bincount( kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2 ))
    clone = np.array(["normal"] * tmpadata.shape[0])
    clone[ (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"]) ] = "tumor"
    tmpadata.obs["clone"] = clone
    # end added
    sc.tl.rank_genes_groups(tmpadata, 'clone', groups=["tumor"], reference="normal", method='wilcoxon')
    genenames = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["names"] ])
    logfc = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["logfoldchanges"] ])
    geneumis = np.array([ map_gene_umi[x] for x in genenames])
    # filtered_out_set = set(list(genenames[ np.abs(logfc) > logfcthreshold ]))
    filtered_out_set = set(list(genenames[ (np.abs(logfc) > logfcthreshold) & (geneumis > umi_threshold) ]))
    print(f"Filter out {len(filtered_out_set)} DE genes")
    #
    new_single_X_rdr = np.zeros((len(x_gene_list), adata.shape[0]))
    for i,x in enumerate(x_gene_list):
        g_list = [z for z in x.split() if z != ""]
        idx_genes = np.array([ map_gene_adatavar[g] for g in g_list if (not g in filtered_out_set) and (g in map_gene_adatavar)])
        if len(idx_genes) > 0:
            new_single_X_rdr[i, :] = np.sum(adata.layers["count"][:, idx_genes], axis=1)
    return new_single_X_rdr


def get_lengths_by_arm(sorted_chr_pos, centromere_file):
    """
    centromere_file for hg38: /u/congma/ragr-data/datasets/ref-genomes/centromeres/hg38.centromeres.txt
    """
    # read and process centromere file
    unique_chrs = [f"chr{i}" for i in range(1, 23)]
    df = pd.read_csv(centromere_file, sep="\t", header=None, index_col=None, names=["CHRNAME", "START", "END", "LABEL", "SOURCE"])
    df = df[df.CHRNAME.isin(unique_chrs)]
    df["CHR"] = [int(x[3:]) for x in df.CHRNAME]
    df = df.groupby("CHR").agg({"CHRNAME":"first", "START":"min", "END":"min", "LABEL":"first", "SOURCE":"first"})
    df.sort_index(inplace=True)
    # count lengths
    mat_chr_pos = np.vstack([ np.array([x[0] for x in sorted_chr_pos]), np.array([x[1] for x in sorted_chr_pos]) ]).T
    armlengths = sum([ [np.sum((mat_chr_pos[:,0] == df.index[i]) & (mat_chr_pos[:,1] <= df.END.iloc[i])), \
                        np.sum((mat_chr_pos[:,0] == df.index[i]) & (mat_chr_pos[:,1] > df.END.iloc[i]))] for i in range(df.shape[0])], [])
    armlengths = np.array(armlengths, dtype=int)
    return armlengths


# def count_reads_from_bam(sorted_chr_pos, barcodes, bamfile):
#     dic_counts = {}
#     map_barcodes = {barcodes[i]:i for i in range(len(barcodes))}
#     map_snp = {f"{x[0]}_{x[1]}":i for i,x in enumerate(sorted_chr_pos)}
#     # partition genome such that each region contains ont SNP in sorted_chr_pos
#     # the region boundary is set to the position right before each SNP
#     for i,x in enumerate(sorted_chr_pos):
#         if i > 0 and sorted_chr_pos[i-1][0] == sorted_chr_pos[i][0] and i+1 < len(sorted_chr_pos) and sorted_chr_pos[i+1][0] == sorted_chr_pos[i][0]:
#             cmd_samtools = f"samtools view -F 1796 -q 13 {bamfile} chr{x[0]}:{x[1]}-{sorted_chr_pos[i+1][1]-1}"
#         elif i+1 < len(sorted_chr_pos) and sorted_chr_pos[i+1][0] == sorted_chr_pos[i][0]:
#             cmd_samtools = f"samtools view -F 1796 -q 13 {bamfile} chr{x[0]}:0-{sorted_chr_pos[i+1][1]-1}"
#         else:
#             cmd_samtools = f"samtools view -F 1796 -q 13 {bamfile} chr{x[0]}:{x[1]}"
#         p = subprocess.Popen(cmd_samtools, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         out,err = p.communicate()
#         # split by barcodes and count the number of reads
#     return NotImplemented


import pysam
def count_reads_from_bam(sorted_chr_pos, barcodes, bamfile, barcodetag="BX"):
    def get_reference(refname):
        if refname[:3] == "chr":
            if refname[3] >= '0' and refname[3] <= '9':
                return int(refname[3:])
            else:
                return None
        else:
            if refname[0] >= '0' and refname[0] <= '9':
                return int(refname)
            else:
                return None
    dic_counts = {}
    map_barcodes = {barcodes[i]:i for i in range(len(barcodes))}
    # separate chr info and pos info of SNPs
    snp_chrs = np.array([x[0] for x in sorted_chr_pos])
    chr_ranges = [(np.where(snp_chrs==c)[0][0], np.where(snp_chrs==c)[0][-1]) for c in range(1,23)]
    snp_pos = np.array([x[1] for x in sorted_chr_pos])
    # parse bam file
    last_chr = 0
    idx = 0
    last_ranges = (-1,-1)
    fp = pysam.AlignmentFile(bamfile, "rb")
    for read in fp:
        if read.is_unmapped or read.is_secondary or read.is_duplicate or read.is_qcfail:
            continue
        this_chr = get_reference(read.reference_name)
        if (not this_chr is None) and (read.has_tag(barcodetag)) and (read.get_tag(barcodetag) in map_barcodes):
            idx_barcode = map_barcodes[ read.get_tag(barcodetag) ]
            if this_chr != last_chr:
                last_ranges = chr_ranges[this_chr-1]
                idx = last_ranges[0]
                last_chr = this_chr
            # find the bin index of the read
            while idx + 1 <= last_ranges[1] and snp_pos[idx+1] <= read.reference_start:
                idx += 1
            if (idx_barcode, idx) in dic_counts:
                dic_counts[(idx_barcode, idx)] += 1
            else:
                dic_counts[(idx_barcode, idx)] = 1
    fp.close()
    # convert dic_counts to count matrix
    list_keys = list(dic_counts.keys())
    row_ind = np.array([k[0] for k in list_keys]).astype(int)
    col_ind = np.array([k[1] for k in list_keys]).astype(int)
    counts = scipy.sparse.csr_matrix(( [dic_counts[k] for k in list_keys], (row_ind, col_ind) ))
    return counts


def generate_bedfile(sorted_chr_pos, outputfile):
    last_chr = 1
    bed_regions = []
    for i,x in enumerate(sorted_chr_pos):
        if i + 1 < len(sorted_chr_pos) and sorted_chr_pos[i+1][0] != last_chr:
            bed_regions.append( (f"chr{x[0]}", x[1], chrlengths[last_chr-1]) )
            last_chr = sorted_chr_pos[i+1][0]
        elif i + 1 < len(sorted_chr_pos):
            bed_regions.append( (f"chr{x[0]}", x[1], sorted_chr_pos[i+1][1]) )
        else:
            bed_regions.append( (f"chr{x[0]}", x[1], chrlengths[-1]) )
    with open(outputfile, 'w') as fp:
        for x in bed_regions:
            fp.write(f"{x[0]}\t{x[1]}\t{x[2]}\n")

# def simple_loop(bamfile, map_barcodes):
#     barcodetag = "BX"
#     fp = pysam.AlignmentFile(bamfile, "rb")
#     for read in fp:
#         if read.is_unmapped or read.is_secondary or read.is_duplicate or read.is_qcfail:
#             continue
#         this_chr = get_reference(read.reference_name)
#         if (not this_chr is None) and (read.has_tag(barcodetag)) and (read.get_tag(barcodetag) in map_barcodes):
#             idx_barcode = map_barcodes[ read.get_tag(barcodetag) ]
#     fp.close()

# import timeit
# timeit.timeit(f'simple_loop(\"{bamfile}\", map_barcodes)', number=1, globals=globals())
# pickle.dump([single_X, lengths, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, coords], open("/u/congma/ragr-data/datasets/SlideDNAseq/allele_starch_results/human_colon_cancer_dna_4x_201027_12/hmminput.pkl", 'wb'))