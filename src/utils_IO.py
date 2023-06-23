import sys
import numpy as np
import scipy
import copy
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
import scanpy as sc
import anndata
from utils_phase_switch import *
from utils_distribution_fitting import *
import subprocess


def load_data(spaceranger_dir, snp_dir, filtergenelist_file, filterregion_file, normalidx_file):
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
    snp_barcodes = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"])

    # add position
    if Path(f"{spaceranger_dir}/spatial/tissue_positions.csv").exists():
        df_pos = pd.read_csv(f"{spaceranger_dir}/spatial/tissue_positions.csv", sep=",", header=0, \
                        names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_pos = pd.read_csv(f"{spaceranger_dir}/spatial/tissue_positions_list.csv", sep=",", header=None, \
                        names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    else:
        raise Exception("No spatial coordinate file!")
    df_pos = df_pos[df_pos.in_tissue == True]
    # assert set(list(df_pos.barcode)) == set(list(adata.obs.index))
    # only keep shared barcodes
    shared_barcodes = set(list(df_pos.barcode)) & set(list(adata.obs.index))
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    df_pos = df_pos[df_pos.barcode.isin(shared_barcodes)]
    # sort and match
    df_pos.barcode = pd.Categorical(df_pos.barcode, categories=list(adata.obs.index), ordered=True)
    df_pos.sort_values(by="barcode", inplace=True)
    adata.obsm["X_pos"] = np.vstack([df_pos.x, df_pos.y]).T

    # shared barcodes between adata and SNPs
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[ pd.Categorical(adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True).argsort(), : ]

    # filter out spots with too small number of UMIs
    indicator = (np.sum(adata.layers["count"], axis=1) > 100)
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out genes that are expressed in <0.5% cells
    indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print("median UMI after filtering out genes < 0.5% of cells = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    # remove genes in filtergenelist_file
    # ig_gene_list = pd.read_csv("/n/fs/ragr-data/users/congma/references/cellranger_refdata-gex-GRCh38-2020-A/genes/ig_gene_list.txt", header=None)
    if not filtergenelist_file is None:
        filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
        filter_gene_list = set(list( filter_gene_list.iloc[:,0] ))
        indicator_filter = np.array([ (not x in filter_gene_list) for x in adata.var.index ])
        adata = adata[:, indicator_filter]
        print("median UMI after filtering out genes in filtergenelist_file = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    if not filterregion_file is None:
        regions = pd.read_csv(filterregion_file, header=None, sep="\t", names=["Chrname", "Start", "End"])
        if "chr" in regions.Chrname.iloc[0]:
            regions["CHR"] = [int(x[3:]) for x in regions.Chrname.values]
        else:
            regions.rename(columns={'Chrname':'CHR'}, inplace=True)
        regions.sort_values(by=["CHR", "Start"], inplace=True)
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])
            while j < regions.shape[0] and ( (regions.CHR.values[j] < this_chr) or ((regions.CHR.values[j] == this_chr) and (regions.End.values[j] <= this_pos)) ):
                j += 1
            if j < regions.shape[0] and (regions.CHR.values[j] == this_chr) and (regions.Start.values[j] <= this_pos) and (regions.End.values[j] > this_pos):
                indicator_filter[i] = False
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        snp_gene_list = snp_gene_list[indicator_filter]
        unique_snp_ids = unique_snp_ids[indicator_filter]

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


# def load_joint_data(input_filelist, snp_dir, filtergenelist_file, normalidx_file):
#     ##### read meta sample info #####
#     df_meta = pd.read_csv(input_filelist, sep="\t", header=None, names=["bam", "sample_id", "spaceranger_dir"])
#     df_barcode = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, names=["combined_barcode"])
#     df_barcode["sample_id"] = [x.split("_")[-1] for x in df_barcode.combined_barcode.values]
#     df_barcode["barcode"] = [x.split("_")[0] for x in df_barcode.combined_barcode.values]
#     ##### read SNP count #####
#     cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
#     cell_snp_Aallele = cell_snp_Aallele.A
#     cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
#     cell_snp_Ballele = cell_snp_Ballele.A
#     snp_gene_list = np.load(f"{snp_dir}/snp_gene_list.npy", allow_pickle=True)
#     unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
#     ##### read anndata and coordinate #####
#     # add position
#     adata = None
#     for i,sname in enumerate(df_meta.sample_id.values):
#         # locate the corresponding rows in df_meta
#         index = np.where(df_barcode["sample_id"] == sname)[0]
#         df_this_barcode = copy.copy(df_barcode.iloc[index, :])
#         df_this_barcode.index = df_this_barcode.barcode
#         # read adata count info
#         adatatmp = sc.read_10x_h5(f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5")
#         adatatmp.layers["count"] = adatatmp.X.A
#         # reorder anndata spots to have the same order as df_this_barcode
#         idx_argsort = pd.Categorical(adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True).argsort()
#         adatatmp = adatatmp[idx_argsort, :]
#         # read position info
#         df_this_pos = pd.read_csv(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv", sep=",", header=0, \
#                     names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
#         df_this_pos = df_this_pos[df_this_pos.in_tissue == True]
#         df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(df_this_barcode.barcode), ordered=True)
#         df_this_pos.sort_values(by="barcode", inplace=True)
#         adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
#         adatatmp.obs["sample"] = sname
#         adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]
#         adatatmp.var_names_make_unique()
#         if adata is None:
#             adata = adatatmp
#         else:
#             adata = anndata.concat([adata, adatatmp], join="outer")
    
#     # # filter out spots with too small number of UMIs
#     # indicator = (np.sum(adata.layers["count"], axis=1) > 100)
#     # adata = adata[indicator, :]
#     # cell_snp_Aallele = cell_snp_Aallele[indicator, :]
#     # cell_snp_Ballele = cell_snp_Ballele[indicator, :]

#     # filter out genes that are expressed in <0.5% cells
#     indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
#     genenames = set(list(adata.var.index[indicator]))
#     adata = adata[:, indicator]
#     print(adata)
#     print("median UMI after filtering out genes < 0.5% of cells = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

#     # remove genes in filtergenelist_file
#     # ig_gene_list = pd.read_csv("/n/fs/ragr-data/users/congma/references/cellranger_refdata-gex-GRCh38-2020-A/genes/ig_gene_list.txt", header=None)
#     if not filtergenelist_file is None:
#         filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
#         filter_gene_list = set(list( filter_gene_list.iloc[:,0] ))
#         indicator_fulter = np.array([ (not x in filter_gene_list) for x in adata.var.index ])
#         adata = adata[:, indicator_fulter]
#         indicator_fulter = np.array([ (x not in filter_gene_list) for x in snp_gene_list ])
#         cell_snp_Aallele = cell_snp_Aallele[:, indicator_fulter]
#         cell_snp_Ballele = cell_snp_Ballele[:, indicator_fulter]
#         snp_gene_list = snp_gene_list[indicator_fulter]
#         unique_snp_ids = unique_snp_ids[indicator_fulter]
#         print("median UMI after filtering out genes in filtergenelist_file = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

#     clf = LocalOutlierFactor(n_neighbors=200)
#     label = clf.fit_predict( np.sum(adata.layers["count"], axis=0).reshape(-1,1) )
#     adata.layers["count"][:, np.where(label==-1)[0]] = 0
#     print("filter out {} outlier genes.".format( np.sum(label==-1) ))

#     if not normalidx_file is None:
#         normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:,0].values
#         adata.obs["tumor_annotation"] = "tumor"
#         adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
#         print( adata.obs["tumor_annotation"].value_counts() )

#     return adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids


def load_joint_data(input_filelist, snp_dir, alignment_files, filtergenelist_file, filterregion_file, normalidx_file):
    ##### read meta sample info #####
    df_meta = pd.read_csv(input_filelist, sep="\t", header=None)
    df_meta.rename(columns=dict(zip( df_meta.columns[:3], ["bam", "sample_id", "spaceranger_dir"] )), inplace=True)
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
    snp_barcodes = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"])

    assert (len(alignment_files) == 0) or (len(alignment_files) + 1 == df_meta.shape[0])

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
        if Path(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv").exists():
            df_this_pos = pd.read_csv(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv", sep=",", header=0, \
                        names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
        elif Path(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv").exists():
            df_this_pos = pd.read_csv(f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv", sep=",", header=None, \
                        names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
        else:
            raise Exception("No spatial coordinate file!")
        df_this_pos = df_this_pos[df_this_pos.in_tissue == True]
        # only keep shared barcodes
        shared_barcodes = set(list(df_this_pos.barcode)) & set(list(adatatmp.obs.index))
        adatatmp = adatatmp[adatatmp.obs.index.isin(shared_barcodes), :]
        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]
        #
        # df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(df_this_barcode.barcode), ordered=True)
        df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True)
        df_this_pos.sort_values(by="barcode", inplace=True)
        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]
        adatatmp.var_names_make_unique()
        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")

    # shared barcodes between adata and SNPs
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[ pd.Categorical(adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True).argsort(), : ]

    ##### load pairwise alignments #####
    # TBD: directly convert to big "adjacency" matrix
    across_slice_adjacency_mat = None
    if len(alignment_files) > 0:
        EPS = 1e-6
        row_ind = []
        col_ind = []
        dat = []
        offset = 0
        for i,f in enumerate(alignment_files):
            pi = np.load(f)
            # normalize p such that max( rowsum(pi), colsum(pi) ) = 1, max alignment weight = 1
            pi = pi / np.max( np.append(np.sum(pi,axis=0), np.sum(pi,axis=1)) )
            sname1 = df_meta.sample_id.values[i]
            sname2 = df_meta.sample_id.values[i+1]
            assert pi.shape[0] == np.sum(df_barcode["sample_id"] == sname1) # double check whether this is correct
            assert pi.shape[1] == np.sum(df_barcode["sample_id"] == sname2) # or the dimension should be flipped
            # for each spot s in sname1, select {t: spot t in sname2 and pi[s,t] >= np.max(pi[s,:])} as the corresponding spot in the other slice
            for row in range(pi.shape[0]):
                cutoff = np.max(pi[row,:]) if np.max(pi[row,:]) > EPS else 1+EPS
                list_cols = np.where(pi[row, :] >= cutoff - EPS)[0]
                row_ind += [offset + row] * len(list_cols)
                col_ind += list( offset + pi.shape[0] + list_cols )
                dat += list(pi[row, list_cols])
            offset += pi.shape[0]
        across_slice_adjacency_mat = scipy.sparse.csr_matrix((dat, (row_ind, col_ind) ), shape=(adata.shape[0], adata.shape[0]))
        across_slice_adjacency_mat += across_slice_adjacency_mat.T
    
    # filter out spots with too small number of UMIs
    indicator = (np.sum(adata.layers["count"], axis=1) > 100)
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator,:][:,indicator]

    # filter out genes that are expressed in <0.5% cells
    indicator = (np.sum(adata.X > 0, axis=0) >= 0.005 * adata.shape[0]).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print("median UMI after filtering out genes < 0.5% of cells = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    if not filtergenelist_file is None:
        filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
        filter_gene_list = set(list( filter_gene_list.iloc[:,0] ))
        indicator_filter = np.array([ (not x in filter_gene_list) for x in adata.var.index ])
        adata = adata[:, indicator_filter]
        print("median UMI after filtering out genes in filtergenelist_file = {}".format( np.median(np.sum(adata.layers["count"], axis=1)) ))

    if not filterregion_file is None:
        regions = pd.read_csv(filterregion_file, header=None, sep="\t", names=["Chrname", "Start", "End"])
        if "chr" in regions.Chrname.iloc[0]:
            regions["CHR"] = [int(x[3:]) for x in regions.Chrname.values]
        else:
            regions.rename(columns={'Chrname':'CHR'}, inplace=True)
        regions.sort_values(by=["CHR", "Start"], inplace=True)
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])
            while j < regions.shape[0] and ( (regions.CHR.values[j] < this_chr) or ((regions.CHR.values[j] == this_chr) and (regions.End.values[j] <= this_pos)) ):
                j += 1
            if j < regions.shape[0] and (regions.CHR.values[j] == this_chr) and (regions.Start.values[j] <= this_pos) and (regions.End.values[j] > this_pos):
                indicator_filter[i] = False
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        snp_gene_list = snp_gene_list[indicator_filter]
        unique_snp_ids = unique_snp_ids[indicator_filter]
        
    clf = LocalOutlierFactor(n_neighbors=200)
    label = clf.fit_predict( np.sum(adata.layers["count"], axis=0).reshape(-1,1) )
    adata.layers["count"][:, np.where(label==-1)[0]] = 0
    print("filter out {} outlier genes.".format( np.sum(label==-1) ))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:,0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        print( adata.obs["tumor_annotation"].value_counts() )

    return adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, across_slice_adjacency_mat


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


def read_bias_correction_info(bc_file):
    try:
        df_info = pd.read_csv(bc_file, header=None, sep="\t")
    except:
        df_info = pd.read_csv(bc_file, header=0, sep="\t")
    return df_info.iloc[:,-1].values


def binning_readcount_using_SNP(df_bins, sorted_chr_pos_first):
    """
    Returns
    ----------
    multiplier : array, (n_bins, n_snp_bins)
        Binary matrix to indicate which RDR bins belong to which SNP bins
    """
    idx = 0
    multiplier = np.zeros((df_bins.shape[0], len(sorted_chr_pos_first)))
    for i in range(df_bins.shape[0]):
        this_chr = df_bins.CHR.values[i]
        this_s = df_bins.START.values[i]
        this_t = df_bins.END.values[i]
        mid = (this_s + this_t) / 2
        # move the cursort on sorted_chr_pos_first such that the chr matches that in df_bins
        while this_chr != sorted_chr_pos_first[idx][0]:
            idx += 1
        while idx + 1 < len(sorted_chr_pos_first) and this_chr == sorted_chr_pos_first[idx+1][0] and mid > sorted_chr_pos_first[idx+1][1]:
            idx += 1
        multiplier[i, idx] = 1
    return multiplier
    

def load_slidedna_readcount(countfile, bead_file, binfile, normalfile, bias_correction_filelist, retained_barcodes, retain_chr_list=np.arange(1,23)):
    # load counts and the corresponding barcodes per spot in counts
    tmpcounts = np.loadtxt(countfile)
    counts = scipy.sparse.csr_matrix(( tmpcounts[:,2], (tmpcounts[:,0].astype(int)-1, tmpcounts[:,1].astype(int)-1) ))
    tmpdf = pd.read_csv(bead_file, header=0, sep=",", index_col=0)
    tmpdf = tmpdf.join( pd.DataFrame(counts.A, index=tmpdf.index))
    # keep only the spots in retained_barcodes
    tmpdf = tmpdf[tmpdf.index.isin(retained_barcodes)]
    # reorder by retained_barcodes
    tmpdf.index = pd.Categorical(tmpdf.index, categories=retained_barcodes, ordered=True)
    tmpdf.sort_index(inplace=True)
    counts = tmpdf.values[:, 2:]

    # load normal counts
    normal_cov = pd.read_csv(normalfile, header=None, sep="\t").values[:,-1].astype(float)

    # load bin info
    df_bins = pd.read_csv(binfile, comment="#", header=None, index_col=None, sep="\t")
    old_names = df_bins.columns[:3]
    df_bins.rename(columns=dict(zip(old_names, ["CHR", "START", "END"])), inplace=True)
    
    # select bins according to retain_chr_list
    retain_chr_list_append = list(retain_chr_list) + [str(x) for x in retain_chr_list] + [f"chr{x}" for x in retain_chr_list]
    bidx = np.where(df_bins.CHR.isin(retain_chr_list_append))[0]
    df_bins = df_bins.iloc[bidx,:]
    counts = counts[:, bidx]
    normal_cov = normal_cov[bidx]

    # sort bins
    df_bins.CHR = [int(x[3:]) if "chr" in x else int(x) for x in df_bins.CHR]
    idx_sort = np.lexsort((df_bins.START, df_bins.CHR))
    df_bins = df_bins.iloc[idx_sort, :]
    counts = counts[:, idx_sort]
    normal_cov = normal_cov[idx_sort]

    # bias correction
    bias_features = []
    for f in bias_correction_filelist:
        this_feature = read_bias_correction_info(f)
        bias_features.append( this_feature[bidx] )
    bias_features = np.array(bias_features).T
    # kernel ridge regression to predict the read count per bin
    # the prediction serves as a baseline of the expected read count, and plays a role in base_nb_mean
    krr = KernelRidge(alpha=0.2)
    krr.fit( bias_features, np.sum(counts, axis=0) / np.sum(counts) )
    pred = krr.predict( bias_features )

    # single_base_nb_mean from bias correction + expected normal
    single_base_nb_mean = (pred * normal_cov).reshape(-1,1) / np.sum(pred * normal_cov) * np.sum(counts, axis=1).reshape(1,-1)
    # single_base_nb_mean = pred.reshape(-1,1) / np.sum(pred) * np.sum(counts, axis=1).reshape(1,-1)

    # remove too low baseline
    threshold = np.median( np.sum(single_base_nb_mean, axis=1) / df_bins.iloc[:,3].values.astype(float) ) * 0.5
    idx_filter = np.where( np.sum(single_base_nb_mean, axis=1) / df_bins.iloc[:,3].values.astype(float) < threshold )[0]
    single_base_nb_mean[idx_filter, :] = 0
    counts[:, idx_filter] = 0

    return counts, single_base_nb_mean, df_bins, normal_cov

    
def get_slidednaseq_rdr(countfile, bead_file, binfile, normalfile, bias_correction_filelist, retained_barcodes, sorted_chr_pos_first, single_X, single_base_nb_mean, retain_chr_list=np.arange(1,23)):
    counts, single_base_nb_mean, df_bins, _ = load_slidedna_readcount(countfile, bead_file, binfile, normalfile, bias_correction_filelist, retained_barcodes)
    # remove bins with low-coverage single_base_nb_mean
    
    multiplier = binning_readcount_using_SNP(df_bins, sorted_chr_pos_first)
    single_X[:,0,:] = multiplier.T @ counts.T
    single_base_nb_mean = multiplier.T @ single_base_nb_mean
    return single_X, single_base_nb_mean


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


# def convert_to_hmm_input_new(adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, hgtable_file, nu, logphase_shift, initial_percentile=75, genome_build="hg38"):
def convert_to_hmm_input_new(adata, cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids, hgtable_file, nu, logphase_shift, initial_min_umi=15, genome_build="hg38"):
    """
    Coverage-based binning.
    """
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
    
    # bin both RDR and BAF
    # initial_min_umi = np.percentile(np.sum(single_total_bb_RD, axis=1), initial_percentile)
    bin_single_X = []
    bin_single_base_nb_mean = []
    bin_single_total_bb_RD = []
    # bin_sorted_chr_pos = []
    bin_sorted_chr_pos_first = []
    bin_sorted_chr_pos_last = []
    bin_x_gene_list = []
    n_snps = []
    i = 0
    per_snp_umis = np.sum(single_total_bb_RD, axis=1)
    while i < single_X.shape[0]:
        t = i + 1
        while t < len(per_snp_umis) and sorted_chr_pos[i][0] == sorted_chr_pos[t][0] and np.sum(per_snp_umis[i:t]) < initial_min_umi:
            t += 1
        if np.sum(per_snp_umis[i:t]) >= initial_min_umi:
            bin_single_X.append( np.sum(single_X[i:t, :, :], axis=0) )
            bin_single_base_nb_mean.append( np.sum(single_base_nb_mean[i:t, :], axis=0) )
            bin_single_total_bb_RD.append( np.sum(single_total_bb_RD[i:t, :], axis=0) )
            # bin_sorted_chr_pos.append( sorted_chr_pos[i] )
            bin_sorted_chr_pos_first.append( sorted_chr_pos[i] )
            bin_sorted_chr_pos_last.append( sorted_chr_pos[t-1] )
            this_genes = [g for g in x_gene_list[i:t] if g != ""]
            bin_x_gene_list.append( " ".join(this_genes) )
            n_snps.append( t - i )
        else:
            bin_single_X[-1] += np.sum(single_X[i:t, :, :], axis=0)
            bin_single_base_nb_mean[-1] += np.sum(single_base_nb_mean[i:t, :], axis=0)
            bin_single_total_bb_RD[-1] += np.sum(single_total_bb_RD[i:t, :], axis=0)
            this_genes = [g for g in x_gene_list[i:t] if g != ""]
            if len(this_genes) > 0:
                bin_x_gene_list[-1] += " " + " ".join(this_genes)
            n_snps[-1] += t - i
        i = t
    single_X = np.stack( bin_single_X )
    single_base_nb_mean = np.vstack(bin_single_base_nb_mean)
    single_total_bb_RD = np.vstack(bin_single_total_bb_RD)
    sorted_chr_pos_first = bin_sorted_chr_pos_first
    sorted_chr_pos_last = bin_sorted_chr_pos_last
    x_gene_list = bin_x_gene_list

    # phase switch probability from genetic distance
    tmp_sorted_chr_pos = [val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair]
    sorted_chr = np.array([x[0] for x in tmp_sorted_chr_pos])
    position_cM = get_position_cM_table( tmp_sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, tmp_sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[np.arange(1, len(log_sitewise_transmat), 2)]

    sorted_chr = np.array([x[0] for x in sorted_chr_pos_first])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos_first, sorted_chr_pos_last, x_gene_list, n_snps


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
    # chr and pos vector from unique_snp_ids
    snp_chr = np.array([ int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([ int(x.split("_")[1]) for x in unique_snp_ids])
    # get binned matrices by matrix multiplication
    # multiplier matrix of size num_snps * num_bins, each entry is a binary indicator of whether the SNP belongs to the bin
    row_ind = np.arange(n_obs)
    col_ind = np.zeros(n_obs, dtype=int)
    bin_sorted_chr_pos_first = []
    bin_sorted_chr_pos_last = []
    n_snps = []
    s = 0
    i = 0
    while s < n_obs:
        t = min(s + tmpbinsize, np.where(snp_chr == snp_chr[s])[0][-1] + 1 )
        col_ind[s:t] = i
        bin_sorted_chr_pos_first.append( (snp_chr[s], snp_pos[s]) )
        # bin_sorted_chr_pos_last.append( (int(unique_snp_ids[t-1].split("_")[0]), int(unique_snp_ids[t-1].split("_")[1])) )
        ##### testing not using exact last SNP position within a bin because it leads to very large penalty in phase switch #####
        bin_sorted_chr_pos_last.append( (snp_chr[s], snp_pos[s]) )
        n_snps.append( t-s )
        s = t
        i += 1
    multiplier = scipy.sparse.csr_matrix(( np.ones(len(row_ind),dtype=int), (row_ind, col_ind) ))
    single_X = np.zeros((multiplier.shape[1], 2, n_spots), dtype=int)
    single_X[:,1,:] = (cell_snp_Aallele @ multiplier).T.A
    single_base_nb_mean = np.zeros((multiplier.shape[1], n_spots), dtype=int)
    single_total_bb_RD = ((cell_snp_Aallele + cell_snp_Ballele) @ multiplier).T.A
    sorted_chr_pos_first = bin_sorted_chr_pos_first
    sorted_chr_pos_last = bin_sorted_chr_pos_last

    # phase switch probability from genetic distance
    unique_chrs = np.arange(1, 23)
    tmp_sorted_chr_pos = [val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair]
    position_cM = get_position_cM_table( tmp_sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, tmp_sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift
    log_sitewise_transmat = log_sitewise_transmat[np.arange(1, len(log_sitewise_transmat), 2)]

    sorted_chr = np.array([x[0] for x in sorted_chr_pos_first])
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos_first, sorted_chr_pos_last, n_snps


def choose_umithreshold_given_nbins(single_total_bb_RD, refined_lengths, expected_nbins):
    def count_num_bins(per_snp_umi, refined_lengths, secondary_min_umi):
        cumlen = 0
        s = 0
        bin_counter = 0
        for le in refined_lengths:
            while s < cumlen + le:
                # initial bin with certain number of SNPs
                t = s + 1
                while t < cumlen + le and np.sum(per_snp_umi[s:t]) < secondary_min_umi:
                    t += 1
                if np.sum(per_snp_umi[s:t]) >= secondary_min_umi:
                    bin_counter += 1
                s = t
            cumlen += le
        return bin_counter
    per_snp_umi = np.sum(single_total_bb_RD, axis=1)
    # candicate range
    lo = np.sort(per_snp_umi)[-expected_nbins]
    # hi = np.sort(per_snp_umi)[-int(expected_nbins/3)]
    hi = int(np.ceil(np.sum(per_snp_umi) / expected_nbins))
    # binary search
    while lo < hi:
        mid = int((hi + lo) / 2)
        bin_counter = count_num_bins(per_snp_umi, refined_lengths, mid)
        if bin_counter == expected_nbins:
            return mid
        elif bin_counter < expected_nbins:
            hi = mid - 1
        else:
            lo = mid + 1
    return mid


# def perform_binning_new(lengths, single_X, single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, phase_indicator, refined_lengths, binsize, rdrbinsize, nu, logphase_shift, secondary_percentile=90, genome_build="hg38"):
def perform_binning_new(lengths, single_X, single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, n_snps, phase_indicator, refined_lengths, binsize, rdrbinsize, nu, logphase_shift, secondary_min_umi=1000, genome_build="hg38"):
    per_snp_umi = np.sum(single_total_bb_RD, axis=1)
    # secondary_min_umi = np.percentile(per_snp_umi, secondary_percentile)
    # bin both RDR and BAF
    bin_single_X_rdr = []
    bin_single_X_baf = []
    bin_single_base_nb_mean = []
    bin_single_total_bb_RD = []
    bin_sorted_chr_pos_first = []
    bin_sorted_chr_pos_last = []
    bin_x_gene_list = []
    bin_n_snps = []
    cumlen = 0
    s = 0
    for le in refined_lengths:
        while s < cumlen + le:
            # initial bin with certain number of SNPs
            t = s + 1
            while t < cumlen + le and np.sum(per_snp_umi[s:t]) < secondary_min_umi:
                t += 1
            # expand binsize by minimum number of genes
            this_genes = sum([ x_gene_list[i].split(" ") for i in range(s,t) ], [])
            this_genes = [z for z in this_genes if z!=""]
            idx_A = np.where(phase_indicator[s:t])[0]
            idx_B = np.where(~phase_indicator[s:t])[0]
            if np.sum(per_snp_umi[s:t]) >= secondary_min_umi:
                bin_single_X_rdr.append( np.sum(single_X[s:t, 0, :], axis=0) )
                bin_single_X_baf.append( np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0) )
                bin_single_base_nb_mean.append( np.sum(single_base_nb_mean[s:t, :], axis=0) )
                bin_single_total_bb_RD.append( np.sum(single_total_bb_RD[s:t, :], axis=0) )
                bin_sorted_chr_pos_first.append( sorted_chr_pos[s] )
                bin_sorted_chr_pos_last.append( sorted_chr_pos_last[t-1] )
                bin_x_gene_list.append( " ".join(this_genes) )
                bin_n_snps.append( np.sum(n_snps[s:t]) )
            else:
                bin_single_X_rdr[-1] += np.sum(single_X[s:t, 0, :], axis=0) 
                bin_single_X_baf[-1] += np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0)
                bin_single_base_nb_mean[-1] += np.sum(single_base_nb_mean[s:t, :], axis=0)
                bin_single_total_bb_RD[-1] += np.sum(single_total_bb_RD[s:t, :], axis=0)
                bin_sorted_chr_pos_last[-1] = sorted_chr_pos_last[t-1]
                if len(this_genes) > 0:
                    bin_x_gene_list[-1] += " " +  " ".join(this_genes)
                bin_n_snps[-1] += np.sum(n_snps[s:t])
            s = t
        cumlen += le
    single_X = np.stack([ np.vstack([bin_single_X_rdr[i], bin_single_X_baf[i]]) for i in range(len(bin_single_X_rdr)) ])
    single_base_nb_mean = np.vstack(bin_single_base_nb_mean)
    single_total_bb_RD = np.vstack(bin_single_total_bb_RD)
    sorted_chr_pos_first = bin_sorted_chr_pos_first
    sorted_chr_pos_last = bin_sorted_chr_pos_last
    x_gene_list = bin_x_gene_list
    n_snps = bin_n_snps

    # phase switch probability from genetic distance
    tmp_sorted_chr_pos = [val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair]
    sorted_chr = np.array([x[0] for x in tmp_sorted_chr_pos])
    position_cM = get_position_cM_table( tmp_sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, tmp_sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[np.arange(1, len(log_sitewise_transmat), 2)]

    sorted_chr = np.array([x[0] for x in sorted_chr_pos_first])
    unique_chrs = [sorted_chr[0]]
    for x in sorted_chr[1:]:
        if x != unique_chrs[-1]:
            unique_chrs.append( x )
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    
    # bin RDR
    s = 0
    while s < single_X.shape[0]:
        t = s+1
        this_genes = sum([ x_gene_list[i].split(" ") for i in range(s,t) ], [])
        this_genes = [z for z in this_genes if z!=""]
        while t < single_X.shape[0] and len(this_genes) < rdrbinsize:
            t += 1
            this_genes += x_gene_list[t-1].split(" ")
            this_genes = [z for z in this_genes if z!=""]
        single_X[s, 0, :] = np.sum(single_X[s:t, 0, :], axis=0)
        single_X[(s+1):t, 0, :] = 0
        single_base_nb_mean[s, :] = np.sum(single_base_nb_mean[s:t, :], axis=0)
        single_base_nb_mean[(s+1):t, :] = 0
        x_gene_list[s] = " ".join(this_genes)
        for k in range(s+1,t):
            x_gene_list[k] = ""
        s = t

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos_first, sorted_chr_pos_last, x_gene_list, n_snps


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
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    position_cM = get_position_cM_table( sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift

    unique_chrs = np.unique(sorted_chr)
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])

    # bin RDR
    for i in range(int(np.ceil(single_X.shape[0] / rdrbinsize))):
        single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :] = np.sum(single_X[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), 0, :], axis=0)
        single_X[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), 0, :] = 0
        single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :] = np.sum(single_base_nb_mean[(i*rdrbinsize):(i*rdrbinsize+rdrbinsize), :], axis=0)
        single_base_nb_mean[(i*rdrbinsize+1):(i*rdrbinsize+rdrbinsize), :] = 0

    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, x_gene_list


def bin_selection_basedon_normal(single_X, single_base_nb_mean, single_total_bb_RD, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, nu, logphase_shift, index_normal, genome_build="hg38", confidence_interval=[0.05, 0.95]):
    """
    Filter out bins that potential contain somatic mutations based on BAF of normal spots.
    """
    # pool B allele counts for each bin across all normal spots
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)
    model = Weighted_BetaBinom(tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD)
    tmpres = model.fit(disp=0)
    # remove bins if normal B allele frequencies fall out of 5%-95% probability range
    removal_indicator1 = (tmpX < scipy.stats.betabinom.ppf(confidence_interval[0], tmptotal_bb_RD, tmpres.params[0] * tmpres.params[1], (1-tmpres.params[0]) * tmpres.params[1]))
    removal_indicator2 = (tmpX > scipy.stats.betabinom.ppf(confidence_interval[1], tmptotal_bb_RD, tmpres.params[0] * tmpres.params[1], (1-tmpres.params[0]) * tmpres.params[1]))
    index_remaining = np.where(~(removal_indicator1 | removal_indicator2))[0]
    #
    # change the related data matrices
    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]
    sorted_chr_pos = [sorted_chr_pos[i] for i in index_remaining]
    sorted_chr_pos_last = [sorted_chr_pos_last[i] for i in index_remaining]
    x_gene_list = [x_gene_list[i] for i in index_remaining]
    # re-estimating phase switch probability
    tmp_sorted_chr_pos = [val for pair in zip(sorted_chr_pos, sorted_chr_pos_last) for val in pair]
    sorted_chr = np.array([x[0] for x in tmp_sorted_chr_pos])
    position_cM = get_position_cM_table( tmp_sorted_chr_pos, genome_build=genome_build )
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, tmp_sorted_chr_pos, nu)
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[np.arange(1, len(log_sitewise_transmat), 2)]
    #
    sorted_chr = np.array([x[0] for x in sorted_chr_pos])
    unique_chrs = [sorted_chr[0]]
    for x in sorted_chr[1:]:
        if x != unique_chrs[-1]:
            unique_chrs.append( x )
    lengths = np.array([ np.sum(sorted_chr == chrname) for chrname in unique_chrs ])
    #
    return lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, sorted_chr_pos, sorted_chr_pos_last, x_gene_list, index_remaining


# def filter_de_genes(adata, x_gene_list, sample_list=None, logfcthreshold=4, quantile_threshold=80):
#     assert "normal_candidate" in adata.obs
def filter_de_genes(exp_counts, x_gene_list, normal_candidate, sample_list=None, sample_ids=None, logfcthreshold=4, quantile_threshold=80):
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate
    #
    map_gene_adatavar = {}
    map_gene_umi = {}
    list_gene_umi = np.sum(adata.layers["count"], axis=0)
    for i,x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]
    #
    if sample_list is None:
        sample_list = [None]
    #
    filtered_out_set = set()
    for s,sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]
        tmpadata = adata[index, :].copy()
        #
        umi_threshold = np.percentile( np.sum(tmpadata.layers["count"], axis=0), quantile_threshold )
        #
        sc.pp.filter_cells(tmpadata, min_genes=200)
        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median( np.sum(tmpadata.layers["count"], axis=1) )
        # sc.pp.normalize_total(tmpadata, target_sum=1e4)
        sc.pp.normalize_total(tmpadata, target_sum=med)
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
        this_filtered_out_set = set(list(genenames[ (np.abs(logfc) > logfcthreshold) & (geneumis > umi_threshold) ]))
        filtered_out_set = filtered_out_set | this_filtered_out_set
        print(f"Filter out {len(filtered_out_set)} DE genes")
    #
    new_single_X_rdr = np.zeros((len(x_gene_list), adata.shape[0]))
    for i,x in enumerate(x_gene_list):
        g_list = [z for z in x.split() if z != ""]
        idx_genes = np.array([ map_gene_adatavar[g] for g in g_list if (not g in filtered_out_set) and (g in map_gene_adatavar)])
        if len(idx_genes) > 0:
            new_single_X_rdr[i, :] = np.sum(adata.layers["count"][:, idx_genes], axis=1)
    return new_single_X_rdr, filtered_out_set


# def filter_de_genes(adata, x_gene_list, sample_list=None, logfcthreshold=4, quantile_threshold=80):
#     assert "normal_candidate" in adata.obs
#     #
#     map_gene_adatavar = {}
#     map_gene_umi = {}
#     list_gene_umi = np.sum(adata.layers["count"], axis=0).A.flatten()
#     for i,x in enumerate(adata.var.index):
#         map_gene_adatavar[x] = i
#         map_gene_umi[x] = list_gene_umi[i]
#     #
#     if sample_list is None:
#         sample_list = [None]
#     #
#     filtered_out_set = set()
#     for s,sname in enumerate(sample_list):
#         if sname is None:
#             index = np.arange(adata.shape[0])
#         else:
#             index = np.where(sample_ids == s)[0]
#         tmpadata = adata[index, :].copy()
#         #
#         umi_threshold = np.percentile( np.sum(tmpadata.layers["count"], axis=0), quantile_threshold )
#         #
#         sc.pp.filter_cells(tmpadata, min_genes=200)
#         sc.pp.filter_genes(tmpadata, min_cells=10)
#         med = np.median( np.sum(tmpadata.layers["count"], axis=1) )
#         # sc.pp.normalize_total(tmpadata, target_sum=1e4)
#         sc.pp.normalize_total(tmpadata, target_sum=med)
#         sc.pp.log1p(tmpadata)
#         # new added
#         sc.pp.pca(tmpadata, n_comps=4)
#         kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.X)
#         kmeans_labels = kmeans.predict(tmpadata.X)
#         idx_kmeans_label = np.argmax(np.bincount( kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2 ))
#         clone = np.array(["normal"] * tmpadata.shape[0])
#         clone[ (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"]) ] = "tumor"
#         tmpadata.obs["clone"] = clone
#         # end added
#         sc.tl.rank_genes_groups(tmpadata, 'clone', groups=["tumor"], reference="normal", method='wilcoxon')
#         genenames = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["names"] ])
#         logfc = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["logfoldchanges"] ])
#         geneumis = np.array([ map_gene_umi[x] for x in genenames])
#         this_filtered_out_set = set(list(genenames[ (np.abs(logfc) > logfcthreshold) & (geneumis > umi_threshold) ]))
#         filtered_out_set = filtered_out_set | this_filtered_out_set
#         print(f"Filter out {len(filtered_out_set)} DE genes")
#     #
#     new_single_X_rdr = np.zeros((len(x_gene_list), adata.shape[0]))
#     for i,x in enumerate(x_gene_list):
#         g_list = [z for z in x.split() if z != ""]
#         idx_genes = np.array([ map_gene_adatavar[g] for g in g_list if (not g in filtered_out_set) and (g in map_gene_adatavar)])
#         if len(idx_genes) > 0:
#             new_single_X_rdr[i, :] = np.sum(adata.layers["count"][:, idx_genes], axis=1)
#     return new_single_X_rdr, filtered_out_set


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


def count_reads_from_bam_bulk(sorted_chr_pos, bamfile):
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
    bulk_counts = np.zeros(len(sorted_chr_pos))
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
        if (not this_chr is None):
            if this_chr != last_chr:
                last_ranges = chr_ranges[this_chr-1]
                idx = last_ranges[0]
                last_chr = this_chr
            # find the bin index of the read
            while idx + 1 <= last_ranges[1] and snp_pos[idx+1] <= read.reference_start:
                idx += 1
            bulk_counts[idx] += 1
    fp.close()
    return bulk_counts


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