import copy
import logging
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""
logger = logging.getLogger(__name__)

import subprocess

from calicost.utils_distribution_fitting import *
from calicost.utils_phase_switch import *


def load_data(
    spaceranger_dir,
    snp_dir,
    filtergenelist_file,
    filterregion_file,
    normalidx_file,
    min_snpumis=50,
    min_percent_expressed_spots=0.005,
):
    ##### read raw UMI count matrix #####
    if Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5").exists():
        adata = sc.read_10x_h5(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5")
    elif Path(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad").exists():
        adata = sc.read_h5ad(f"{spaceranger_dir}/filtered_feature_bc_matrix.h5ad")
    else:
        logging.error(
            f"{spaceranger_dir} directory doesn't have a filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix.h5ad file!"
        )

        raise RuntimeError()

    adata.layers["count"] = adata.X.A.astype(int)
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    snp_barcodes = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"]
    )

    # add position
    if Path(f"{spaceranger_dir}/spatial/tissue_positions.csv").exists():
        df_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions.csv",
            sep=",",
            header=0,
            names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
        )
    elif Path(f"{spaceranger_dir}/spatial/tissue_positions_list.csv").exists():
        df_pos = pd.read_csv(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
        )
    else:
        raise Exception("No spatial coordinate file!")
    df_pos = df_pos[df_pos.in_tissue == True]
    # assert set(list(df_pos.barcode)) == set(list(adata.obs.index))
    # only keep shared barcodes
    shared_barcodes = set(list(df_pos.barcode)) & set(list(adata.obs.index))
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    df_pos = df_pos[df_pos.barcode.isin(shared_barcodes)]
    # sort and match
    df_pos.barcode = pd.Categorical(
        df_pos.barcode, categories=list(adata.obs.index), ordered=True
    )
    df_pos.sort_values(by="barcode", inplace=True)
    adata.obsm["X_pos"] = np.vstack([df_pos.x, df_pos.y]).T

    # shared barcodes between adata and SNPs
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]

    # filter out spots with too small number of UMIs
    indicator = np.sum(adata.layers["count"], axis=1) > min_snpumis
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out spots with too small number of SNP-covering UMIs
    indicator = (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snpumis
    )
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out genes that are expressed in <0.5% cells
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print(
        "median UMI after filtering out genes < 0.5% of cells = {}".format(
            np.median(np.sum(adata.layers["count"], axis=1))
        )
    )

    # remove genes in filtergenelist_file
    # ig_gene_list = pd.read_csv("/n/fs/ragr-data/users/congma/references/cellranger_refdata-gex-GRCh38-2020-A/genes/ig_gene_list.txt", header=None)
    if not filtergenelist_file is None:
        filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
        filter_gene_list = set(list(filter_gene_list.iloc[:, 0]))
        indicator_filter = np.array(
            [(not x in filter_gene_list) for x in adata.var.index]
        )
        adata = adata[:, indicator_filter]
        print(
            "median UMI after filtering out genes in filtergenelist_file = {}".format(
                np.median(np.sum(adata.layers["count"], axis=1))
            )
        )

    if not filterregion_file is None:
        regions = pd.read_csv(
            filterregion_file, header=None, sep="\t", names=["Chrname", "Start", "End"]
        )
        if "chr" in regions.Chrname.iloc[0]:
            regions["CHR"] = [int(x[3:]) for x in regions.Chrname.values]
        else:
            regions.rename(columns={"Chrname": "CHR"}, inplace=True)
        regions.sort_values(by=["CHR", "Start"], inplace=True)
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])
            while j < regions.shape[0] and (
                (regions.CHR.values[j] < this_chr)
                or (
                    (regions.CHR.values[j] == this_chr)
                    and (regions.End.values[j] <= this_pos)
                )
            ):
                j += 1
            if (
                j < regions.shape[0]
                and (regions.CHR.values[j] == this_chr)
                and (regions.Start.values[j] <= this_pos)
                and (regions.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        unique_snp_ids = unique_snp_ids[indicator_filter]

    clf = LocalOutlierFactor(n_neighbors=200)
    label = clf.fit_predict(np.sum(adata.layers["count"], axis=0).reshape(-1, 1))
    adata.layers["count"][:, np.where(label == -1)[0]] = 0
    print("filter out {} outlier genes.".format(np.sum(label == -1)))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:, 0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        print(adata.obs["tumor_annotation"].value_counts())

    return adata, cell_snp_Aallele.A, cell_snp_Ballele.A, unique_snp_ids


def load_joint_data(
    input_filelist,
    snp_dir,
    alignment_files,
    filtergenelist_file,
    filterregion_file,
    normalidx_file,
    min_snpumis=50,
    min_percent_expressed_spots=0.005,
):
    ##### read meta sample info #####
    df_meta = pd.read_csv(input_filelist, sep="\t", header=None)
    df_meta.rename(
        columns=dict(zip(df_meta.columns[:3], ["bam", "sample_id", "spaceranger_dir"])),
        inplace=True,
    )
    logger.info(f"Input spaceranger file list {input_filelist} contains:")
    logger.info(df_meta)
    df_barcode = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["combined_barcode"]
    )
    df_barcode["sample_id"] = [
        x.split("_")[-1] for x in df_barcode.combined_barcode.values
    ]
    df_barcode["barcode"] = [
        x.split("_")[0] for x in df_barcode.combined_barcode.values
    ]
    ##### read SNP count #####
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    snp_barcodes = pd.read_csv(
        f"{snp_dir}/barcodes.txt", header=None, names=["barcodes"]
    )

    assert (len(alignment_files) == 0) or (len(alignment_files) + 1 == df_meta.shape[0])

    ##### read anndata and coordinate #####
    # add position
    adata = None
    for i, sname in enumerate(df_meta.sample_id.values):
        # locate the corresponding rows in df_meta
        index = np.where(df_barcode["sample_id"] == sname)[0]
        df_this_barcode = copy.copy(df_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode
        # read adata count info
        if Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5"
        ).exists():
            adatatmp = sc.read_10x_h5(
                f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5"
            )
        elif Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5ad"
        ).exists():
            adatatmp = sc.read_h5ad(
                f"{df_meta['spaceranger_dir'].iloc[i]}/filtered_feature_bc_matrix.h5ad"
            )
        else:
            logging.error(
                f"{df_meta['spaceranger_dir'].iloc[i]} directory doesn't have a filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix.h5ad file!"
            )
            raise RuntimeError()

        adatatmp.layers["count"] = adatatmp.X.A
        # reorder anndata spots to have the same order as df_this_barcode
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()
        adatatmp = adatatmp[idx_argsort, :]
        # read position info
        if Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv"
        ).exists():
            df_this_pos = pd.read_csv(
                f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions.csv",
                sep=",",
                header=0,
                names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
            )
        elif Path(
            f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv"
        ).exists():
            df_this_pos = pd.read_csv(
                f"{df_meta['spaceranger_dir'].iloc[i]}/spatial/tissue_positions_list.csv",
                sep=",",
                header=None,
                names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"],
            )
        else:
            raise Exception("No spatial coordinate file!")
        df_this_pos = df_this_pos[df_this_pos.in_tissue == True]
        # only keep shared barcodes
        shared_barcodes = set(list(df_this_pos.barcode)) & set(list(adatatmp.obs.index))
        adatatmp = adatatmp[adatatmp.obs.index.isin(shared_barcodes), :]
        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]
        #
        # df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(df_this_barcode.barcode), ordered=True)
        df_this_pos.barcode = pd.Categorical(
            df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True
        )
        df_this_pos.sort_values(by="barcode", inplace=True)
        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T
        adatatmp.obs["sample"] = sname
        adatatmp.obs.index = [f"{x}_{sname}" for x in adatatmp.obs.index]
        adatatmp.var_names_make_unique()
        if adata is None:
            adata = adatatmp
        else:
            adata = anndata.concat([adata, adatatmp], join="outer")
    # replace nan with 0
    adata.layers["count"][np.isnan(adata.layers["count"])] = 0
    adata.layers["count"] = adata.layers["count"].astype(int)

    # shared barcodes between adata and SNPs
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]

    ##### load pairwise alignments #####
    # TBD: directly convert to big "adjacency" matrix
    across_slice_adjacency_mat = None
    if len(alignment_files) > 0:
        EPS = 1e-6
        row_ind = []
        col_ind = []
        dat = []
        offset = 0
        for i, f in enumerate(alignment_files):
            pi = np.load(f)
            # normalize p such that max( rowsum(pi), colsum(pi) ) = 1, max alignment weight = 1
            pi = pi / np.max(np.append(np.sum(pi, axis=0), np.sum(pi, axis=1)))
            sname1 = df_meta.sample_id.values[i]
            sname2 = df_meta.sample_id.values[i + 1]
            assert pi.shape[0] == np.sum(
                df_barcode["sample_id"] == sname1
            )  # double check whether this is correct
            assert pi.shape[1] == np.sum(
                df_barcode["sample_id"] == sname2
            )  # or the dimension should be flipped
            # for each spot s in sname1, select {t: spot t in sname2 and pi[s,t] >= np.max(pi[s,:])} as the corresponding spot in the other slice
            for row in range(pi.shape[0]):
                cutoff = np.max(pi[row, :]) if np.max(pi[row, :]) > EPS else 1 + EPS
                list_cols = np.where(pi[row, :] >= cutoff - EPS)[0]
                row_ind += [offset + row] * len(list_cols)
                col_ind += list(offset + pi.shape[0] + list_cols)
                dat += list(pi[row, list_cols])
            offset += pi.shape[0]
        across_slice_adjacency_mat = scipy.sparse.csr_matrix(
            (dat, (row_ind, col_ind)), shape=(adata.shape[0], adata.shape[0])
        )
        across_slice_adjacency_mat += across_slice_adjacency_mat.T

    # filter out spots with too small number of UMIs
    indicator = np.sum(adata.layers["count"], axis=1) >= min_snpumis
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # filter out spots with too small number of SNP-covering UMIs
    indicator = (
        np.sum(cell_snp_Aallele, axis=1).A.flatten()
        + np.sum(cell_snp_Ballele, axis=1).A.flatten()
        >= min_snpumis
    )
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]
    if not (across_slice_adjacency_mat is None):
        across_slice_adjacency_mat = across_slice_adjacency_mat[indicator, :][
            :, indicator
        ]

    # filter out genes that are expressed in <min_percent_expressed_spots cells
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_percent_expressed_spots * adata.shape[0]
    ).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print(
        "median UMI after filtering out genes < 0.5% of cells = {}".format(
            np.median(np.sum(adata.layers["count"], axis=1))
        )
    )

    if not filtergenelist_file is None:
        filter_gene_list = pd.read_csv(filtergenelist_file, header=None)
        filter_gene_list = set(list(filter_gene_list.iloc[:, 0]))
        indicator_filter = np.array(
            [(not x in filter_gene_list) for x in adata.var.index]
        )
        adata = adata[:, indicator_filter]
        print(
            "median UMI after filtering out genes in filtergenelist_file = {}".format(
                np.median(np.sum(adata.layers["count"], axis=1))
            )
        )

    if not filterregion_file is None:
        regions = pd.read_csv(
            filterregion_file, header=None, sep="\t", names=["Chrname", "Start", "End"]
        )
        if "chr" in regions.Chrname.iloc[0]:
            regions["CHR"] = [int(x[3:]) for x in regions.Chrname.values]
        else:
            regions.rename(columns={"Chrname": "CHR"}, inplace=True)
        regions.sort_values(by=["CHR", "Start"], inplace=True)
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])
            while j < regions.shape[0] and (
                (regions.CHR.values[j] < this_chr)
                or (
                    (regions.CHR.values[j] == this_chr)
                    and (regions.End.values[j] <= this_pos)
                )
            ):
                j += 1
            if (
                j < regions.shape[0]
                and (regions.CHR.values[j] == this_chr)
                and (regions.Start.values[j] <= this_pos)
                and (regions.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        unique_snp_ids = unique_snp_ids[indicator_filter]

    clf = LocalOutlierFactor(n_neighbors=200)
    label = clf.fit_predict(np.sum(adata.layers["count"], axis=0).reshape(-1, 1))
    adata.layers["count"][:, np.where(label == -1)[0]] = 0
    print("filter out {} outlier genes.".format(np.sum(label == -1)))

    if not normalidx_file is None:
        normal_barcodes = pd.read_csv(normalidx_file, header=None).iloc[:, 0].values
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"
        print(adata.obs["tumor_annotation"].value_counts())

    return (
        adata,
        cell_snp_Aallele.A,
        cell_snp_Ballele.A,
        unique_snp_ids,
        across_slice_adjacency_mat,
    )


def load_slidedna_data(snp_dir, bead_file, filterregion_bedfile):
    cell_snp_Aallele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(f"{snp_dir}/cell_snp_Ballele.npz")
    unique_snp_ids = np.load(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    barcodes = pd.read_csv(f"{snp_dir}/barcodes.txt", header=None, index_col=None)
    barcodes = barcodes.iloc[:, 0].values
    # add spatial position
    df_pos = pd.read_csv(bead_file, header=0, sep=",", index_col=None)
    coords = np.vstack([df_pos.xcoord, df_pos.ycoord]).T
    # remove SNPs within filterregion_bedfile
    if not filterregion_bedfile is None:
        df_filter = pd.read_csv(
            filterregion_bedfile,
            header=None,
            sep="\t",
            names=["chrname", "start", "end"],
        )
        df_filter = df_filter[df_filter.chrname.isin([f"chr{i}" for i in range(1, 23)])]
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
            while (filter_chrs[j] < snp_chrs[i]) or (
                (filter_chrs[j] == snp_chrs[i]) and (filter_end[j] < snp_pos[i])
            ):
                j += 1
            if (
                filter_chrs[j] == snp_chrs[i]
                and filter_start[j] <= snp_pos[i]
                and filter_end[j] >= snp_pos[i]
            ):
                is_within_filterregion.append(True)
            else:
                is_within_filterregion.append(False)
        is_within_filterregion = np.array(is_within_filterregion)
        # remove SNPs based on is_within_filterregion
        cell_snp_Aallele = cell_snp_Aallele[:, ~is_within_filterregion]
        cell_snp_Ballele = cell_snp_Ballele[:, ~is_within_filterregion]
        unique_snp_ids = unique_snp_ids[~is_within_filterregion]
    return coords, cell_snp_Aallele, cell_snp_Ballele, barcodes, unique_snp_ids


def taking_shared_barcodes(
    snp_barcodes, cell_snp_Aallele, cell_snp_Ballele, adata, df_pos
):
    # shared barcodes between adata and SNPs
    shared_barcodes = (
        set(list(snp_barcodes.barcodes))
        & set(list(adata.obs.index))
        & set(list(df_pos.barcode))
    )
    cell_snp_Aallele = cell_snp_Aallele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    cell_snp_Ballele = cell_snp_Ballele[snp_barcodes.barcodes.isin(shared_barcodes), :]
    snp_barcodes = snp_barcodes[snp_barcodes.barcodes.isin(shared_barcodes)]
    adata = adata[adata.obs.index.isin(shared_barcodes), :]
    adata = adata[
        pd.Categorical(
            adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]
    df_pos = df_pos[df_pos.barcode.isin(shared_barcodes)]
    df_pos = df_pos.iloc[
        pd.Categorical(
            df_pos.barcode, categories=list(snp_barcodes.barcodes), ordered=True
        ).argsort(),
        :,
    ]
    return snp_barcodes, cell_snp_Aallele, cell_snp_Ballele, adata, df_pos


def filter_genes_barcodes_hatchetblock(
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    snp_barcodes,
    unique_snp_ids,
    config,
    min_umi=100,
    min_spot_percent=0.005,
    ordered_chr=[str(c) for c in range(1, 23)],
):
    # filter out spots with too small number of UMIs
    indicator = np.sum(adata.layers["count"], axis=1) > min_umi
    adata = adata[indicator, :]
    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    # filter out genes that are expressed in <0.5% cells
    indicator = (
        np.sum(adata.X > 0, axis=0) >= min_spot_percent * adata.shape[0]
    ).A.flatten()
    genenames = set(list(adata.var.index[indicator]))
    adata = adata[:, indicator]
    print(adata)
    print(
        "median UMI after filtering out genes < 0.5% of cells = {}".format(
            np.median(np.sum(adata.layers["count"], axis=1))
        )
    )

    if not config["filtergenelist_file"] is None:
        filter_gene_list = pd.read_csv(config["filtergenelist_file"], header=None)
        filter_gene_list = set(list(filter_gene_list.iloc[:, 0]))
        indicator_filter = np.array(
            [(not x in filter_gene_list) for x in adata.var.index]
        )
        adata = adata[:, indicator_filter]
        print(
            "median UMI after filtering out genes in filtergenelist_file = {}".format(
                np.median(np.sum(adata.layers["count"], axis=1))
            )
        )

    if not config["filterregion_file"] is None:
        regions = pd.read_csv(
            config["filterregion_file"],
            header=None,
            sep="\t",
            names=["Chrname", "Start", "End"],
        )
        ordered_chr_map = {ordered_chr[i]: i for i in range(len(ordered_chr))}
        # retain only chromosomes in ordered_chr
        if ~np.any(regions.Chrname.isin(ordered_chr)):
            regions["Chrname"] = regions.Chrname.map(lambda x: x.replace("chr", ""))
        regions = regions[regions.Chrname.isin(ordered_chr)]
        regions["int_chrom"] = regions.Chrname.map(ordered_chr_map)
        regions.sort_values(by=["int_chrom", "Start"], inplace=True)
        indicator_filter = np.array([True] * cell_snp_Aallele.shape[1])
        j = 0
        for i in range(cell_snp_Aallele.shape[1]):
            this_chr = int(unique_snp_ids[i].split("_")[0])
            this_pos = int(unique_snp_ids[i].split("_")[1])
            while j < regions.shape[0] and (
                (regions.int_chrom.values[j] < this_chr)
                or (
                    (regions.int_chrom.values[j] == this_chr)
                    and (regions.End.values[j] <= this_pos)
                )
            ):
                j += 1
            if (
                j < regions.shape[0]
                and (regions.int_chrom.values[j] == this_chr)
                and (regions.Start.values[j] <= this_pos)
                and (regions.End.values[j] > this_pos)
            ):
                indicator_filter[i] = False
        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]
        unique_snp_ids = unique_snp_ids[indicator_filter]

    return adata, cell_snp_Aallele, cell_snp_Ballele, snp_barcodes, unique_snp_ids


def read_bias_correction_info(bc_file):
    try:
        df_info = pd.read_csv(bc_file, header=None, sep="\t")
    except:
        df_info = pd.read_csv(bc_file, header=0, sep="\t")
    return df_info.iloc[:, -1].values


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
        while (
            idx + 1 < len(sorted_chr_pos_first)
            and this_chr == sorted_chr_pos_first[idx + 1][0]
            and mid > sorted_chr_pos_first[idx + 1][1]
        ):
            idx += 1
        multiplier[i, idx] = 1
    return multiplier


def load_slidedna_readcount(
    countfile,
    bead_file,
    binfile,
    normalfile,
    bias_correction_filelist,
    retained_barcodes,
    retain_chr_list=np.arange(1, 23),
):
    # load counts and the corresponding barcodes per spot in counts
    tmpcounts = np.loadtxt(countfile)
    counts = scipy.sparse.csr_matrix(
        (
            tmpcounts[:, 2],
            (tmpcounts[:, 0].astype(int) - 1, tmpcounts[:, 1].astype(int) - 1),
        )
    )
    tmpdf = pd.read_csv(bead_file, header=0, sep=",", index_col=0)
    tmpdf = tmpdf.join(pd.DataFrame(counts.A, index=tmpdf.index))
    # keep only the spots in retained_barcodes
    tmpdf = tmpdf[tmpdf.index.isin(retained_barcodes)]
    # reorder by retained_barcodes
    tmpdf.index = pd.Categorical(
        tmpdf.index, categories=retained_barcodes, ordered=True
    )
    tmpdf.sort_index(inplace=True)
    counts = tmpdf.values[:, 2:]

    # load normal counts
    normal_cov = (
        pd.read_csv(normalfile, header=None, sep="\t").values[:, -1].astype(float)
    )

    # load bin info
    df_bins = pd.read_csv(binfile, comment="#", header=None, index_col=None, sep="\t")
    old_names = df_bins.columns[:3]
    df_bins.rename(columns=dict(zip(old_names, ["CHR", "START", "END"])), inplace=True)

    # select bins according to retain_chr_list
    retain_chr_list_append = (
        list(retain_chr_list)
        + [str(x) for x in retain_chr_list]
        + [f"chr{x}" for x in retain_chr_list]
    )
    bidx = np.where(df_bins.CHR.isin(retain_chr_list_append))[0]
    df_bins = df_bins.iloc[bidx, :]
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
        bias_features.append(this_feature[bidx])
    bias_features = np.array(bias_features).T
    # kernel ridge regression to predict the read count per bin
    # the prediction serves as a baseline of the expected read count, and plays a role in base_nb_mean
    krr = KernelRidge(alpha=0.2)
    krr.fit(bias_features, np.sum(counts, axis=0) / np.sum(counts))
    pred = krr.predict(bias_features)

    # single_base_nb_mean from bias correction + expected normal
    single_base_nb_mean = (
        (pred * normal_cov).reshape(-1, 1)
        / np.sum(pred * normal_cov)
        * np.sum(counts, axis=1).reshape(1, -1)
    )
    # single_base_nb_mean = pred.reshape(-1,1) / np.sum(pred) * np.sum(counts, axis=1).reshape(1,-1)

    # remove too low baseline
    threshold = (
        np.median(
            np.sum(single_base_nb_mean, axis=1)
            / df_bins.iloc[:, 3].values.astype(float)
        )
        * 0.5
    )
    idx_filter = np.where(
        np.sum(single_base_nb_mean, axis=1) / df_bins.iloc[:, 3].values.astype(float)
        < threshold
    )[0]
    single_base_nb_mean[idx_filter, :] = 0
    counts[:, idx_filter] = 0

    return counts, single_base_nb_mean, df_bins, normal_cov


def get_slidednaseq_rdr(
    countfile,
    bead_file,
    binfile,
    normalfile,
    bias_correction_filelist,
    retained_barcodes,
    sorted_chr_pos_first,
    single_X,
    single_base_nb_mean,
    retain_chr_list=np.arange(1, 23),
):
    counts, single_base_nb_mean, df_bins, _ = load_slidedna_readcount(
        countfile,
        bead_file,
        binfile,
        normalfile,
        bias_correction_filelist,
        retained_barcodes,
    )
    # remove bins with low-coverage single_base_nb_mean

    multiplier = binning_readcount_using_SNP(df_bins, sorted_chr_pos_first)
    single_X[:, 0, :] = multiplier.T @ counts.T
    single_base_nb_mean = multiplier.T @ single_base_nb_mean
    return single_X, single_base_nb_mean


def filter_slidedna_spot_by_adjacency(
    coords, cell_snp_Aallele, cell_snp_Ballele, barcodes
):
    # distance to center
    dist = np.sqrt(
        np.sum(np.square(coords - np.median(coords, axis=0, keepdims=True)), axis=1)
    )
    idx_keep = np.where(dist < 2500)[0]
    # remove spots
    coords = coords[idx_keep, :]
    cell_snp_Aallele = cell_snp_Aallele[idx_keep, :]
    cell_snp_Ballele = cell_snp_Ballele[idx_keep, :]
    barcodes = barcodes[idx_keep]
    return coords, cell_snp_Aallele, cell_snp_Ballele, barcodes


def combine_gene_snps(unique_snp_ids, hgtable_file, adata):
    # read gene info and keep only chr1-chr22 and genes appearing in adata
    df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
    df_hgtable = df_hgtable[df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])]
    df_hgtable = df_hgtable[df_hgtable.name2.isin(adata.var.index)]
    # a data frame including both gene and SNP info: CHR, START, END, snp_id, gene, is_interval
    df_gene_snp = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.chrom.values],
            "START": df_hgtable.cdsStart.values,
            "END": df_hgtable.cdsEnd.values,
            "snp_id": None,
            "gene": df_hgtable.name2.values,
            "is_interval": True,
        }
    )
    # add SNP info
    snp_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids])
    df_gene_snp = pd.concat(
        [
            df_gene_snp,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_pos + 1,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )
    df_gene_snp.sort_values(by=["CHR", "START"], inplace=True)

    # check the what gene each SNP belongs to
    # for each SNP (with not null snp_id), find the previous gene (is_interval == True) such that the SNP start position is within the gene start and end interval
    vec_is_interval = df_gene_snp.is_interval.values
    vec_chr = df_gene_snp.CHR.values
    vec_start = df_gene_snp.START.values
    vec_end = df_gene_snp.END.values
    for i in np.where(df_gene_snp.gene.isnull())[0]:
        if i == 0:
            continue
        this_pos = vec_start[i]
        j = i - 1
        while j >= 0 and j >= i - 50 and vec_chr[i] == vec_chr[j]:
            if (
                vec_is_interval[j]
                and vec_start[j] <= this_pos
                and vec_end[j] > this_pos
            ):
                df_gene_snp.iloc[i, 4] = df_gene_snp.iloc[j]["gene"]
                break
            j -= 1

    # remove SNPs that have no corresponding genes
    df_gene_snp = df_gene_snp[~df_gene_snp.gene.isnull()]
    return df_gene_snp


def create_haplotype_block_ranges(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi=15,
):
    """
    Initially block SNPs along genome.

    Returns
    ----------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id)
        Gene and SNP info combined into a single data frame sorted by genomic positions. "is_interval" suggest whether the entry is a gene or a SNP. "gene" column either contain gene name if the entry is a gene, or the gene a SNP belongs to if the entry is a SNP.
    """
    # first level: partition of genome: by gene regions (if two genes overlap, they are grouped to one region)
    tmp_block_genome_intervals = list(
        zip(
            df_gene_snp[df_gene_snp.is_interval].CHR.values,
            df_gene_snp[df_gene_snp.is_interval].START.values,
            df_gene_snp[df_gene_snp.is_interval].END.values,
        )
    )
    block_genome_intervals = [tmp_block_genome_intervals[0]]
    for x in tmp_block_genome_intervals[1:]:
        # check whether overlap with previous block
        if x[0] == block_genome_intervals[-1][0] and max(
            x[1], block_genome_intervals[-1][1]
        ) < min(x[2], block_genome_intervals[-1][2]):
            block_genome_intervals[-1] = (
                x[0],
                min(x[1], block_genome_intervals[-1][1]),
                max(x[2], block_genome_intervals[-1][2]),
            )
        else:
            block_genome_intervals.append(x)
    # get block_ranges in the index of df_gene_snp
    block_ranges = []
    for x in block_genome_intervals:
        indexes = np.where(
            (df_gene_snp.CHR.values == x[0])
            & (
                np.maximum(df_gene_snp.START.values, x[1])
                < np.minimum(df_gene_snp.END.values, x[2])
            )
        )[0]
        block_ranges.append((indexes[0], indexes[-1] + 1))
    assert np.all(
        np.array(np.array([x[1] for x in block_ranges[:-1]]))
        == np.array(np.array([x[0] for x in block_ranges[1:]]))
    )
    # record the initial block id in df_gene_snps
    df_gene_snp["initial_block_id"] = 0
    for i, x in enumerate(block_ranges):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    # second level: group the first level blocks into haplotype blocks such that the minimum SNP-covering UMI counts >= initial_min_umi
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}
    initial_block_chr = df_gene_snp.CHR.values[np.array([x[0] for x in block_ranges])]
    block_ranges_new = []
    s = 0
    while s < len(block_ranges):
        t = s
        while t <= len(block_ranges):
            t += 1
            reach_end = t == len(block_ranges)
            change_chr = initial_block_chr[s] != initial_block_chr[t - 1]
            # count SNP-covering UMI
            involved_snps_ids = df_gene_snp[
                (df_gene_snp.initial_block_id >= s) & (df_gene_snp.initial_block_id < t)
            ].snp_id
            involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()].values
            involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])
            this_snp_umis = (
                0
                if len(involved_snp_idx) == 0
                else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                + np.sum(cell_snp_Ballele[:, involved_snp_idx])
            )
            if reach_end:
                break
            if change_chr:
                t -= 1
                # re-count SNP-covering UMIs
                involved_snps_ids = df_gene_snp.snp_id.iloc[
                    block_ranges[s][0] : block_ranges[t - 1][1]
                ]
                involved_snps_ids = involved_snps_ids[
                    ~involved_snps_ids.isnull()
                ].values
                involved_snp_idx = np.array(
                    [map_snp_index[x] for x in involved_snps_ids]
                )
                this_snp_umis = (
                    0
                    if len(involved_snp_idx) == 0
                    else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                    + np.sum(cell_snp_Ballele[:, involved_snp_idx])
                )
                break
            if this_snp_umis >= initial_min_umi:
                break
        #
        if (
            this_snp_umis < initial_min_umi
            and s > 0
            and initial_block_chr[s - 1] == initial_block_chr[s]
        ):
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new[-1] = (block_ranges_new[-1][0], indexes[-1] + 1)
        else:
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new.append((indexes[0], indexes[-1] + 1))
        s = t

    # record the block id in df_gene_snps
    df_gene_snp["block_id"] = 0
    for i, x in enumerate(block_ranges_new):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    df_gene_snp = df_gene_snp.drop(columns=["initial_block_id"])
    return df_gene_snp


def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    nu,
    logphase_shift,
    geneticmap_file,
):
    """
    Attributes:
    ----------
    df_gene_snp : pd.DataFrame
        Contain "block_id" column to indicate which genes/snps belong to which block.

    Returns
    ----------
    lengths : array, (n_chromosomes,)
        Number of blocks per chromosome.

    single_X : array, (n_blocks, 2, n_spots)
        Transcript counts and B allele count per block per cell.

    single_base_nb_mean : array, (n_blocks, n_spots)
        Baseline transcript counts in normal diploid per block per cell.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total allele count per block per cell.

    log_sitewise_transmat : array, (n_blocks,)
        Log phase switch probability between each pair of adjacent blocks.
    """
    blocks = df_gene_snp.block_id.unique()
    single_X = np.zeros((len(blocks), 2, adata.shape[0]), dtype=int)
    single_base_nb_mean = np.zeros((len(blocks), adata.shape[0]))
    single_total_bb_RD = np.zeros((len(blocks), adata.shape[0]), dtype=int)
    # summarize counts of involved genes and SNPs within each block
    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}
    df_block_contents = df_gene_snp.groupby("block_id").agg(
        {"snp_id": list, "gene": list}
    )
    for b in range(df_block_contents.shape[0]):
        # BAF (SNPs)
        involved_snps_ids = [
            x for x in df_block_contents.snp_id.values[b] if not x is None
        ]
        involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])
        if len(involved_snp_idx) > 0:
            single_X[b, 1, :] = np.sum(cell_snp_Aallele[:, involved_snp_idx], axis=1)
            single_total_bb_RD[b, :] = np.sum(
                cell_snp_Aallele[:, involved_snp_idx], axis=1
            ) + np.sum(cell_snp_Ballele[:, involved_snp_idx], axis=1)
        # RDR (genes)
        involved_genes = list(
            set([x for x in df_block_contents.gene.values[b] if not x is None])
        )
        if len(involved_genes) > 0:
            single_X[b, 0, :] = np.sum(
                adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
            )

    # lengths
    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)
    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(df_gene_snp[df_gene_snp.CHR == c].block_id.unique())

    # phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )
    sorted_chr_pos_last = df_gene_snp.groupby("block_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )
    #
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]
    position_cM = get_position_cM_table(tmp_sorted_chr_pos, geneticmap_file)
    phase_switch_prob = compute_phase_switch_probability_position(
        position_cM, tmp_sorted_chr_pos, nu
    )
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
    )


def choose_umithreshold_given_nbins(
    single_total_bb_RD, refined_lengths, expected_nbins
):
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


def perform_binning_new(
    lengths,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    sorted_chr_pos,
    sorted_chr_pos_last,
    x_gene_list,
    n_snps,
    phase_indicator,
    refined_lengths,
    binsize,
    rdrbinsize,
    nu,
    logphase_shift,
    geneticmap_file,
    secondary_min_umi=1000,
    max_binlength=5e6,
):
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
                if (
                    sorted_chr_pos_last[t - 1][1] - sorted_chr_pos[s][1]
                    >= max_binlength
                ):
                    t = max(t - 1, s + 1)
                    break
            # expand binsize by minimum number of genes
            this_genes = sum([x_gene_list[i].split(" ") for i in range(s, t)], [])
            this_genes = [z for z in this_genes if z != ""]
            idx_A = np.where(phase_indicator[s:t])[0]
            idx_B = np.where(~phase_indicator[s:t])[0]
            # if np.sum(per_snp_umi[s:t]) >= secondary_min_umi or sorted_chr_pos[s][0] != bin_sorted_chr_pos_last[-1][0]:
            #     bin_single_X_rdr.append( np.sum(single_X[s:t, 0, :], axis=0) )
            #     bin_single_X_baf.append( np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0) )
            #     bin_single_base_nb_mean.append( np.sum(single_base_nb_mean[s:t, :], axis=0) )
            #     bin_single_total_bb_RD.append( np.sum(single_total_bb_RD[s:t, :], axis=0) )
            #     bin_sorted_chr_pos_first.append( sorted_chr_pos[s] )
            #     bin_sorted_chr_pos_last.append( sorted_chr_pos_last[t-1] )
            #     bin_x_gene_list.append( " ".join(this_genes) )
            #     bin_n_snps.append( np.sum(n_snps[s:t]) )
            # else:
            #     bin_single_X_rdr[-1] += np.sum(single_X[s:t, 0, :], axis=0)
            #     bin_single_X_baf[-1] += np.sum(single_X[s:t, 1, :][idx_A,:], axis=0) + np.sum(single_total_bb_RD[s:t, :][idx_B,:] - single_X[s:t, 1, :][idx_B,:], axis=0)
            #     bin_single_base_nb_mean[-1] += np.sum(single_base_nb_mean[s:t, :], axis=0)
            #     bin_single_total_bb_RD[-1] += np.sum(single_total_bb_RD[s:t, :], axis=0)
            #     bin_sorted_chr_pos_last[-1] = sorted_chr_pos_last[t-1]
            #     if len(this_genes) > 0:
            #         bin_x_gene_list[-1] += " " +  " ".join(this_genes)
            #     bin_n_snps[-1] += np.sum(n_snps[s:t])
            if (
                len(bin_sorted_chr_pos_last) > 0
                and sorted_chr_pos[s][0] == bin_sorted_chr_pos_last[-1][0]
                and np.sum(per_snp_umi[s:t]) < 0.5 * secondary_min_umi
                and sorted_chr_pos_last[t - 1][1] - sorted_chr_pos[s][1]
                < 0.5 * max_binlength
            ):
                bin_single_X_rdr[-1] += np.sum(single_X[s:t, 0, :], axis=0)
                bin_single_X_baf[-1] += np.sum(
                    single_X[s:t, 1, :][idx_A, :], axis=0
                ) + np.sum(
                    single_total_bb_RD[s:t, :][idx_B, :]
                    - single_X[s:t, 1, :][idx_B, :],
                    axis=0,
                )
                bin_single_base_nb_mean[-1] += np.sum(
                    single_base_nb_mean[s:t, :], axis=0
                )
                bin_single_total_bb_RD[-1] += np.sum(single_total_bb_RD[s:t, :], axis=0)
                bin_sorted_chr_pos_last[-1] = sorted_chr_pos_last[t - 1]
                if len(this_genes) > 0:
                    bin_x_gene_list[-1] += " " + " ".join(this_genes)
                bin_n_snps[-1] += np.sum(n_snps[s:t])
            else:
                bin_single_X_rdr.append(np.sum(single_X[s:t, 0, :], axis=0))
                bin_single_X_baf.append(
                    np.sum(single_X[s:t, 1, :][idx_A, :], axis=0)
                    + np.sum(
                        single_total_bb_RD[s:t, :][idx_B, :]
                        - single_X[s:t, 1, :][idx_B, :],
                        axis=0,
                    )
                )
                bin_single_base_nb_mean.append(
                    np.sum(single_base_nb_mean[s:t, :], axis=0)
                )
                bin_single_total_bb_RD.append(
                    np.sum(single_total_bb_RD[s:t, :], axis=0)
                )
                bin_sorted_chr_pos_first.append(sorted_chr_pos[s])
                bin_sorted_chr_pos_last.append(sorted_chr_pos_last[t - 1])
                bin_x_gene_list.append(" ".join(this_genes))
                bin_n_snps.append(np.sum(n_snps[s:t]))
            s = t
        cumlen += le
    single_X = np.stack(
        [
            np.vstack([bin_single_X_rdr[i], bin_single_X_baf[i]])
            for i in range(len(bin_single_X_rdr))
        ]
    )
    single_base_nb_mean = np.vstack(bin_single_base_nb_mean)
    single_total_bb_RD = np.vstack(bin_single_total_bb_RD)
    sorted_chr_pos_first = bin_sorted_chr_pos_first
    sorted_chr_pos_last = bin_sorted_chr_pos_last
    x_gene_list = bin_x_gene_list
    n_snps = bin_n_snps

    # phase switch probability from genetic distance
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]
    sorted_chr = np.array([x[0] for x in tmp_sorted_chr_pos])
    position_cM = get_position_cM_table(tmp_sorted_chr_pos, geneticmap_file)
    phase_switch_prob = compute_phase_switch_probability_position(
        position_cM, tmp_sorted_chr_pos, nu
    )
    log_sitewise_transmat = np.log(phase_switch_prob) - logphase_shift
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    sorted_chr = np.array([x[0] for x in sorted_chr_pos_first])
    unique_chrs = [sorted_chr[0]]
    for x in sorted_chr[1:]:
        if x != unique_chrs[-1]:
            unique_chrs.append(x)
    lengths = np.array([np.sum(sorted_chr == chrname) for chrname in unique_chrs])

    # bin RDR
    s = 0
    while s < single_X.shape[0]:
        t = s + 1
        this_genes = sum([x_gene_list[i].split(" ") for i in range(s, t)], [])
        this_genes = [z for z in this_genes if z != ""]
        while t < single_X.shape[0] and len(this_genes) < rdrbinsize:
            t += 1
            this_genes += x_gene_list[t - 1].split(" ")
            this_genes = [z for z in this_genes if z != ""]
        single_X[s, 0, :] = np.sum(single_X[s:t, 0, :], axis=0)
        single_X[(s + 1) : t, 0, :] = 0
        single_base_nb_mean[s, :] = np.sum(single_base_nb_mean[s:t, :], axis=0)
        single_base_nb_mean[(s + 1) : t, :] = 0
        x_gene_list[s] = " ".join(this_genes)
        for k in range(s + 1, t):
            x_gene_list[k] = ""
        s = t

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        sorted_chr_pos_first,
        sorted_chr_pos_last,
        x_gene_list,
        n_snps,
    )


def create_bin_ranges(
    df_gene_snp,
    single_total_bb_RD,
    refined_lengths,
    secondary_min_umi,
    max_binlength=5e6,
):
    """
    Aggregate haplotype blocks to bins

    Attributes
    ----------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id)
        Gene and SNP info combined into a single data frame sorted by genomic positions. "is_interval" suggest whether the entry is a gene or a SNP. "gene" column either contain gene name if the entry is a gene, or the gene a SNP belongs to if the entry is a SNP.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total SNP-covering reads per haplotype block per spot.

    refined_lengths : array
        Number of haplotype blocks before each phase switch. The numbers should sum up to n_blocks.

    Returns
    -------
    df_gene_snp : data frame, (CHR, START, END, snp_id, gene, is_interval, block_id, bin_id)
        The newly added bin_id column indicates which bin each gene or SNP belongs to.
    """

    def greedy_binning_nobreak(
        block_lengths, block_umi, secondary_min_umi, max_binlength
    ):
        """
        Returns
        -------
        bin_ids : array, (n_blocks)
            The bin id of the input blocks. Should have the same size with block_lengths and block_umi.
        """
        assert len(block_lengths) == len(block_umi)
        bin_ranges = []
        s = 0
        while s < len(block_lengths):
            t = s + 1
            while t < len(block_lengths) and np.sum(block_umi[s:t]) < secondary_min_umi:
                t += 1
                if np.sum(block_lengths[s:t]) >= max_binlength:
                    t = max(t - 1, s + 1)
                    break
            # check whether it is a very small bin in the end
            if (
                s > 0
                and t == len(block_lengths)
                and np.sum(block_umi[s:t]) < 0.5 * secondary_min_umi
                and np.sum(block_lengths[s:t]) < 0.5 * max_binlength
            ):
                bin_ranges[-1][1] = t
            else:
                bin_ranges.append([s, t])
            s = t
        bin_ids = np.zeros(len(block_lengths), dtype=int)
        for i, x in enumerate(bin_ranges):
            bin_ids[x[0] : x[1]] = i
        return bin_ids

    # block lengths and block umis
    sorted_chr_pos_both = df_gene_snp.groupby("block_id").agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )
    block_lengths = sorted_chr_pos_both.END.values - sorted_chr_pos_both.START.values
    block_umi = np.sum(single_total_bb_RD, axis=1)
    n_blocks = len(block_lengths)

    # get a list of breakpoints where bin much break
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )
    breakpoints = np.sort(np.unique(breakpoints))
    # append 0 in the front of breakpoints so that each pair of adjacent breakpoints can be an input to greedy_binning_nobreak
    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)
    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # loop over breakpoints and bin each block
    bin_ids = np.zeros(n_blocks, dtype=int)
    offset = 0
    for i in range(len(breakpoints) - 1):
        b1 = breakpoints[i]
        b2 = breakpoints[i + 1]
        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2], block_umi[b1:b2], secondary_min_umi, max_binlength
            )
            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    # append bin_ids to df_gene_snp
    df_gene_snp["bin_id"] = df_gene_snp.block_id.map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    return df_gene_snp


def summarize_counts_for_bins(
    df_gene_snp,
    adata,
    single_X,
    single_total_bb_RD,
    phase_indicator,
    nu,
    logphase_shift,
    geneticmap_file,
):
    """
    Attributes:
    ----------
    df_gene_snp : pd.DataFrame
        Contain "block_id" column to indicate which genes/snps belong to which block.

    Returns
    ----------
    lengths : array, (n_chromosomes,)
        Number of blocks per chromosome.

    single_X : array, (n_blocks, 2, n_spots)
        Transcript counts and B allele count per block per cell.

    single_base_nb_mean : array, (n_blocks, n_spots)
        Baseline transcript counts in normal diploid per block per cell.

    single_total_bb_RD : array, (n_blocks, n_spots)
        Total allele count per block per cell.

    log_sitewise_transmat : array, (n_blocks,)
        Log phase switch probability between each pair of adjacent blocks.
    """
    bins = df_gene_snp.bin_id.unique()
    bin_single_X = np.zeros((len(bins), 2, adata.shape[0]), dtype=int)
    bin_single_base_nb_mean = np.zeros((len(bins), adata.shape[0]))
    bin_single_total_bb_RD = np.zeros((len(bins), adata.shape[0]), dtype=int)
    # summarize counts of involved genes and SNPs within each block
    df_bin_contents = (
        df_gene_snp[~df_gene_snp.bin_id.isnull()]
        .groupby("bin_id")
        .agg({"block_id": set, "gene": set})
    )
    for b in range(df_bin_contents.shape[0]):
        # BAF (SNPs)
        involved_blocks = [
            x for x in df_bin_contents.block_id.values[b] if not x is None
        ]
        this_phased = np.where(
            phase_indicator[involved_blocks].reshape(-1, 1),
            single_X[involved_blocks, 1, :],
            single_total_bb_RD[involved_blocks, :] - single_X[involved_blocks, 1, :],
        )
        bin_single_X[b, 1, :] = np.sum(this_phased, axis=0)
        bin_single_total_bb_RD[b, :] = np.sum(
            single_total_bb_RD[involved_blocks, :], axis=0
        )
        # RDR (genes)
        involved_genes = [x for x in df_bin_contents.gene.values[b] if not x is None]
        bin_single_X[b, 0, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    # lengths
    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)
    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    # phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )
    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )
    #
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]
    position_cM = get_position_cM_table(tmp_sorted_chr_pos, geneticmap_file)
    phase_switch_prob = compute_phase_switch_probability_position(
        position_cM, tmp_sorted_chr_pos, nu
    )
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        bin_single_X,
        bin_single_base_nb_mean,
        bin_single_total_bb_RD,
        log_sitewise_transmat,
    )


def bin_selection_basedon_normal(
    df_gene_snp,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    nu,
    logphase_shift,
    index_normal,
    geneticmap_file,
    confidence_interval=[0.05, 0.95],
    min_betabinom_tau=30,
):
    """
    Filter out bins that potential contain somatic mutations based on BAF of normal spots.
    """
    # pool B allele counts for each bin across all normal spots
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)
    model = Weighted_BetaBinom(
        tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD
    )
    tmpres = model.fit(disp=0)
    tmpres.params[0] = 0.5
    tmpres.params[-1] = max(tmpres.params[-1], min_betabinom_tau)
    # remove bins if normal B allele frequencies fall out of 5%-95% probability range
    removal_indicator1 = tmpX < scipy.stats.betabinom.ppf(
        confidence_interval[0],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1 - tmpres.params[0]) * tmpres.params[1],
    )
    removal_indicator2 = tmpX > scipy.stats.betabinom.ppf(
        confidence_interval[1],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1 - tmpres.params[0]) * tmpres.params[1],
    )
    print(np.sum(removal_indicator1 | removal_indicator2))
    index_removal = np.where(removal_indicator1 | removal_indicator2)[0]
    index_remaining = np.where(~(removal_indicator1 | removal_indicator2))[0]
    #
    # change df_gene_snp
    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None
    # remap bin_id to existing list
    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    # change the related data matrices
    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]

    # lengths
    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)
    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    ## phase switch probability from genetic distance
    sorted_chr_pos_first = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "first", "START": "first"}
    )
    sorted_chr_pos_first = list(
        zip(sorted_chr_pos_first.CHR.values, sorted_chr_pos_first.START.values)
    )
    sorted_chr_pos_last = df_gene_snp.groupby("bin_id").agg(
        {"CHR": "last", "END": "last"}
    )
    sorted_chr_pos_last = list(
        zip(sorted_chr_pos_last.CHR.values, sorted_chr_pos_last.END.values)
    )
    #
    tmp_sorted_chr_pos = [
        val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair
    ]
    position_cM = get_position_cM_table(tmp_sorted_chr_pos, geneticmap_file)
    phase_switch_prob = compute_phase_switch_probability_position(
        position_cM, tmp_sorted_chr_pos, nu
    )
    log_sitewise_transmat = np.minimum(
        np.log(0.5), np.log(phase_switch_prob) - logphase_shift
    )
    # log_sitewise_transmat = log_sitewise_transmat[np.arange(0, len(log_sitewise_transmat), 2)]
    log_sitewise_transmat = log_sitewise_transmat[
        np.arange(1, len(log_sitewise_transmat), 2)
    ]

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        df_gene_snp,
    )


def filter_de_genes(
    exp_counts,
    x_gene_list,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold=4,
    quantile_threshold=80,
):
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate
    #
    map_gene_adatavar = {}
    map_gene_umi = {}
    list_gene_umi = np.sum(adata.layers["count"], axis=0)
    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]
    #
    if sample_list is None:
        sample_list = [None]
    #
    filtered_out_set = set()
    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]
        tmpadata = adata[index, :].copy()
        #
        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )
        #
        sc.pp.filter_cells(tmpadata, min_genes=200)
        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))
        # sc.pp.normalize_total(tmpadata, target_sum=1e4)
        sc.pp.normalize_total(tmpadata, target_sum=med)
        sc.pp.log1p(tmpadata)
        # new added
        sc.pp.pca(tmpadata, n_comps=4)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
        kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])
        idx_kmeans_label = np.argmax(
            np.bincount(kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2)
        )
        clone = np.array(["normal"] * tmpadata.shape[0])
        clone[
            (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "tumor"
        tmpadata.obs["clone"] = clone
        # end added
        sc.tl.rank_genes_groups(
            tmpadata, "clone", groups=["tumor"], reference="normal", method="wilcoxon"
        )
        genenames = np.array([x[0] for x in tmpadata.uns["rank_genes_groups"]["names"]])
        logfc = np.array(
            [x[0] for x in tmpadata.uns["rank_genes_groups"]["logfoldchanges"]]
        )
        geneumis = np.array([map_gene_umi[x] for x in genenames])
        this_filtered_out_set = set(
            list(
                genenames[(np.abs(logfc) > logfcthreshold) & (geneumis > umi_threshold)]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set
        print(f"Filter out {len(filtered_out_set)} DE genes")
    #
    new_single_X_rdr = np.zeros((len(x_gene_list), adata.shape[0]))
    for i, x in enumerate(x_gene_list):
        g_list = [z for z in x.split() if z != ""]
        idx_genes = np.array(
            [
                map_gene_adatavar[g]
                for g in g_list
                if (not g in filtered_out_set) and (g in map_gene_adatavar)
            ]
        )
        if len(idx_genes) > 0:
            new_single_X_rdr[i, :] = np.sum(adata.layers["count"][:, idx_genes], axis=1)
    return new_single_X_rdr, filtered_out_set


def filter_de_genes_tri(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2,
    logfcthreshold_t=4,
    quantile_threshold=80,
):
    """
    Attributes
    ----------
    df_bininfo : pd.DataFrame
        Contains columns ['CHR', 'START', 'END', 'INCLUDED_GENES', 'INCLUDED_SNP_IDS'], 'INCLUDED_GENES' contains space-delimited gene names.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate
    #
    map_gene_adatavar = {}
    map_gene_umi = {}
    list_gene_umi = np.sum(adata.layers["count"], axis=0)
    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]
    #
    if sample_list is None:
        sample_list = [None]
    #
    filtered_out_set = set()
    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]
        tmpadata = adata[index, :].copy()
        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10
        ):
            continue
        #
        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )
        #
        # sc.pp.filter_cells(tmpadata, min_genes=200)
        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))
        # sc.pp.normalize_total(tmpadata, target_sum=1e4)
        sc.pp.normalize_total(tmpadata, target_sum=med)
        sc.pp.log1p(tmpadata)
        # new added
        sc.pp.pca(tmpadata, n_comps=4)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
        kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])
        idx_kmeans_label = np.argmax(
            np.bincount(kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2)
        )
        clone = np.array(["normal"] * tmpadata.shape[0])
        clone[
            (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "tumor"
        ### third part ###
        clone[
            (kmeans_labels == idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "unsure"
        tmpadata.obs["clone"] = clone
        # end added
        # sc.tl.rank_genes_groups(tmpadata, 'clone', groups=["tumor", "unsure"], reference="normal", method='wilcoxon')
        # # DE and log fold change comparing tumor and normal
        # genenames_t = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["names"] ])
        # logfc_t = np.array([ x[0] for x in tmpadata.uns["rank_genes_groups"]["logfoldchanges"] ])
        # geneumis_t = np.array([ map_gene_umi[x] for x in genenames_t])
        # # DE and log fold change comparing unsure and normal
        # genenames_u = np.array([ x[1] for x in tmpadata.uns["rank_genes_groups"]["names"] ])
        # logfc_u = np.array([ x[1] for x in tmpadata.uns["rank_genes_groups"]["logfoldchanges"] ])
        # geneumis_u = np.array([ map_gene_umi[x] for x in genenames_u])
        # this_filtered_out_set = set(list(genenames_t[ (np.abs(logfc_t) > logfcthreshold) & (geneumis_t > umi_threshold) ])) | set(list(genenames_u[ (np.abs(logfc_u) > logfcthreshold) & (geneumis_u > umi_threshold) ]))
        #
        agg_counts = np.vstack(
            [
                np.sum(tmpadata.layers["count"][tmpadata.obs["clone"] == c, :], axis=0)
                for c in ["normal", "unsure", "tumor"]
            ]
        )
        agg_counts = agg_counts / np.sum(agg_counts, axis=1, keepdims=True) * 1e6
        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])
        logfc_u = np.where(
            ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[1, :] / agg_counts[0, :]),
        )
        logfc_t = np.where(
            ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[2, :] / agg_counts[0, :]),
        )
        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set
        print(f"Filter out {len(filtered_out_set)} DE genes")
    #
    # remove genes that are in filtered_out_set
    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        # RDR (genes)
        involved_genes = set(genestr.split(" ")) - filtered_out_set
        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    return new_single_X_rdr, filtered_out_set


def get_lengths_by_arm(sorted_chr_pos, centromere_file):
    """
    centromere_file for hg38: /u/congma/ragr-data/datasets/ref-genomes/centromeres/hg38.centromeres.txt
    """
    # read and process centromere file
    unique_chrs = [f"chr{i}" for i in range(1, 23)]
    df = pd.read_csv(
        centromere_file,
        sep="\t",
        header=None,
        index_col=None,
        names=["CHRNAME", "START", "END", "LABEL", "SOURCE"],
    )
    df = df[df.CHRNAME.isin(unique_chrs)]
    df["CHR"] = [int(x[3:]) for x in df.CHRNAME]
    df = df.groupby("CHR").agg(
        {
            "CHRNAME": "first",
            "START": "min",
            "END": "min",
            "LABEL": "first",
            "SOURCE": "first",
        }
    )
    df.sort_index(inplace=True)
    # count lengths
    mat_chr_pos = np.vstack(
        [
            np.array([x[0] for x in sorted_chr_pos]),
            np.array([x[1] for x in sorted_chr_pos]),
        ]
    ).T
    armlengths = sum(
        [
            [
                np.sum(
                    (mat_chr_pos[:, 0] == df.index[i])
                    & (mat_chr_pos[:, 1] <= df.END.iloc[i])
                ),
                np.sum(
                    (mat_chr_pos[:, 0] == df.index[i])
                    & (mat_chr_pos[:, 1] > df.END.iloc[i])
                ),
            ]
            for i in range(df.shape[0])
        ],
        [],
    )
    armlengths = np.array(armlengths, dtype=int)
    return armlengths


# def expand_df_cnv(df_cnv, binsize=1e6):
#     df_expand = []
#     for i in range(df_cnv.shape[0]):
#         # repeat the row i for int(END - START / binsize) times and save to a new dataframe
#         n_bins = max(1, int(1.0*(df_cnv.iloc[i].END - df_cnv.iloc[i].START) / binsize))
#         tmp = pd.DataFrame(np.repeat(df_cnv.iloc[i:(i+1),:].values, n_bins, axis=0), columns=df_cnv.columns)
#         for k in range(n_bins):
#             tmp.END.iloc[k] = df_cnv.START.iloc[i]+ k*binsize
#         tmp.END.iloc[-1] = df_cnv.END.iloc[i]
#         df_expand.append(tmp)
#     df_expand = pd.concat(df_expand, ignore_index=True)
#     return df_expand


def expand_df_cnv(df_cnv, binsize=2e5, fillmissing=True):
    # get CHR and its END
    df_chr_end = df_cnv.groupby("CHR").agg({"END": "max"}).reset_index()

    # initialize df_expand as a dataframe containing CHR, START, END such that END-START = binsize
    df_expand = []
    for i, c in enumerate(df_chr_end.CHR.values):
        df_expand.append(
            pd.DataFrame(
                {
                    "CHR": c,
                    "START": np.arange(0, df_chr_end.END.values[i], binsize),
                    "END": binsize + np.arange(0, df_chr_end.END.values[i], binsize),
                }
            )
        )
    df_expand = pd.concat(df_expand, ignore_index=True)

    # find the index in df_cnv such that each entry in df_expand overlaps with the largest length
    vec_cnv_chr = df_cnv.CHR.values
    vec_cnv_start = df_cnv.START.values
    vec_cnv_end = df_cnv.END.values

    seg_index = -1 * np.ones(df_expand.shape[0], dtype=int)
    j = 0
    for i, this_chr in enumerate(df_expand.CHR.values):
        this_start = df_expand.START.values[i]
        this_end = df_expand.END.values[i]
        while j < df_cnv.shape[0] and (
            vec_cnv_chr[j] < this_chr
            or (vec_cnv_chr[j] == this_chr and vec_cnv_end[j] <= this_start)
        ):
            j += 1
        # overlap length of the j-th segment to (j+3)-th segment in df_cnv
        overlap_lengths = []
        for k in range(j, min(j + 3, df_cnv.shape[0])):
            if vec_cnv_chr[k] > this_chr or vec_cnv_start[k] > this_end:
                break
            overlap_lengths.append(
                min(vec_cnv_end[k], this_end) - max(vec_cnv_start[k], this_start)
            )
        if len(overlap_lengths) > 0:
            seg_index[i] = j + np.argmax(overlap_lengths)

    for col in df_cnv.columns[df_cnv.columns.str.startswith("clone")]:
        df_expand[col] = np.nan
        df_expand[col].iloc[seg_index >= 0] = df_cnv[col].values[
            seg_index[seg_index >= 0]
        ]
        df_expand[col] = df_expand[col].astype("Int64")

    if fillmissing:
        # for each nan row, fill it with the closest non-nan row
        nan_rows = np.where(df_expand.iloc[:, -1].isnull())[0]
        filled_rows = np.where(~df_expand.iloc[:, -1].isnull())[0]
        for i in nan_rows:
            candidates = np.where(
                (~df_expand.iloc[:, -1].isnull())
                & (df_expand.CHR.values == df_expand.CHR.values[i])
            )[0]
            j = candidates[np.argmin(np.abs(candidates - i))]
            df_expand.iloc[i, 3:] = df_expand.iloc[j, 3:].values

    return df_expand


def summary_events(cnv_segfile, rescombinefile, minlength=10):
    EPS_BAF = 0.07
    # read rescombine file
    res_combine = dict(np.load(rescombinefile, allow_pickle=True))
    pred_cnv = res_combine["pred_cnv"]
    logrdr_profile = np.vstack(
        [res_combine["new_log_mu"][pred_cnv[:, c], c] for c in range(pred_cnv.shape[1])]
    )
    baf_profile = np.vstack(
        [
            res_combine["new_p_binom"][pred_cnv[:, c], c]
            for c in range(pred_cnv.shape[1])
        ]
    )

    # read CNV file
    df_cnv = pd.read_csv(cnv_segfile, header=0, sep="\t")
    # get clone names
    calico_clones = np.array(
        [x.split(" ")[0][5:] for x in df_cnv.columns if x.endswith(" A")]
    )
    # retain only the clones that are not entirely diploid
    calico_clones = [
        c
        for c in calico_clones
        if np.sum(np.abs(baf_profile[int(c), :] - 0.5) > EPS_BAF) > minlength
    ]
    # label CNV states per bin per clone into "neu", "del", "amp", "loh" states
    for c in calico_clones:
        counts = df_cnv.END.values - df_cnv.START.values
        counts = np.maximum(1, counts / 1e4).astype(int)
        tmp = strict_convert_copy_to_states(
            df_cnv[f"clone{c} A"].values, df_cnv[f"clone{c} B"].values, counts=counts
        )
        tmp[tmp == "bdel"] = "del"
        tmp[tmp == "bamp"] = "amp"
        df_cnv[f"srt_cnstate_clone{c}"] = tmp

    # partition the genome into segments such that the allele-specific CN across all clones are the same within each segment
    segments, labs = get_intervals_nd(
        df_cnv[
            ["CHR"]
            + [f"clone{x} A" for x in calico_clones]
            + [f"clone{x} B" for x in calico_clones]
        ].values
    )
    # collect event, that is labs and segments pair such that the cnstate is not normal
    events = []
    for i, seg in enumerate(segments):
        if seg[1] - seg[0] < minlength:
            continue
        if np.all(
            df_cnv[[f"srt_cnstate_clone{x}" for x in calico_clones]]
            .iloc[seg[0], :]
            .values
            == "neu"
        ):
            continue
        acn_list = [
            (
                df_cnv[f"srt_cnstate_clone{c}"].values[seg[0]],
                df_cnv[f"clone{c} A"].values[seg[0]],
                df_cnv[f"clone{c} B"].values[seg[0]],
            )
            for c in calico_clones
        ]
        acn_set = set(acn_list)
        for e in acn_set:
            if e[0] == "neu":
                continue
            involved_clones = [
                calico_clones[i] for i in range(len(calico_clones)) if acn_list[i] == e
            ]
            events.append(
                pd.DataFrame(
                    {
                        "CHR": df_cnv.CHR.values[seg[0]],
                        "START": df_cnv.START.values[seg[0]],
                        "END": df_cnv.END.values[seg[1] - 1],
                        "BinSTART": seg[0],
                        "BinEND": seg[1] - 1,
                        "CN": f"{e[1]}|{e[2]}",
                        "Label": e[0],
                        "involved_clones": ",".join(involved_clones),
                    },
                    index=[0],
                )
            )
    df_events = pd.concat(events, ignore_index=True)

    # merge adjacent events if they have the same involved_clones and same CN
    unique_ic = np.unique(df_events.involved_clones.values)
    concise_events = []
    for ic in unique_ic:
        tmpdf = df_events[df_events.involved_clones == ic]
        # merge adjacent rows in tmpdf if they have the same CN END of the previous row is the same as the START of the next row
        concise_events.append(tmpdf.iloc[0:1, :])
        for i in range(1, tmpdf.shape[0]):
            if (
                tmpdf.CN.values[i] == concise_events[-1].CN.values[0]
                and tmpdf.CHR.values[i] == concise_events[-1].CHR.values[0]
                and tmpdf.START.values[i] == concise_events[-1].END.values[0]
            ):
                concise_events[-1].END.values[0] = tmpdf.END.values[i]
                concise_events[-1].BinEND.values[0] = tmpdf.BinEND.values[i]
            else:
                concise_events.append(tmpdf.iloc[i : (i + 1), :])
    df_concise_events = pd.concat(concise_events, ignore_index=True)

    # add the RDR abd BAF info
    rdr = np.nan * np.ones(df_concise_events.shape[0])
    baf = np.nan * np.ones(df_concise_events.shape[0])
    rdr_diff = np.nan * np.ones(df_concise_events.shape[0])
    baf_diff = np.nan * np.ones(df_concise_events.shape[0])
    for i in range(df_concise_events.shape[0]):
        involved_clones = np.array(
            [int(c) for c in df_concise_events.involved_clones.values[i].split(",")]
        )
        bs = df_concise_events.BinSTART.values[i]
        be = df_concise_events.BinEND.values[i]
        # rdr[i] = np.exp(np.mean(res_combine["new_log_mu"][ (pred_cnv[bs:be,:][:,involved_clones].flatten(), np.tile(involved_clones, be-bs)) ]))
        # baf[i] = np.mean(res_combine["new_p_binom"][ (pred_cnv[bs:be,:][:,involved_clones].flatten(), np.tile(involved_clones, be-bs)) ])
        rdr[i] = np.exp(
            np.mean(np.concatenate([logrdr_profile[i, bs:be] for i in involved_clones]))
        )
        baf[i] = np.mean(
            np.concatenate([baf_profile[i, bs:be] for i in involved_clones])
        )
        # get the uninvolved clones
        uninvolved_clones = np.array(
            [int(c) for c in calico_clones if int(c) not in involved_clones]
        )
        if len(uninvolved_clones) > 0:
            # rdr_diff[i] = np.exp(np.mean(res_combine["new_log_mu"][ (pred_cnv[bs:be,:][:,uninvolved_clones].flatten(), np.tile(uninvolved_clones, be-bs)) ])) - rdr[i]
            # baf_diff[i] = np.mean(res_combine["new_p_binom"][ (pred_cnv[bs:be,:][:,uninvolved_clones].flatten(), np.tile(uninvolved_clones, be-bs)) ]) - baf[i]
            rdr_diff[i] = rdr[i] - np.exp(
                np.mean(
                    np.concatenate(
                        [logrdr_profile[i, bs:be] for i in uninvolved_clones]
                    )
                )
            )
            baf_diff[i] = baf[i] - np.mean(
                np.concatenate([baf_profile[i, bs:be] for i in uninvolved_clones])
            )
    df_concise_events["RDR"] = rdr
    df_concise_events["BAF"] = baf
    df_concise_events["RDR_diff"] = rdr_diff
    df_concise_events["BAF_diff"] = baf_diff

    return df_concise_events[
        [
            "CHR",
            "START",
            "END",
            "BinSTART",
            "BinEND",
            "RDR",
            "BAF",
            "RDR_diff",
            "BAF_diff",
            "CN",
            "Label",
            "involved_clones",
        ]
    ]


def get_best_initialization(output_dir):
    """
    find the best HMRF initialization random seed
    """
    # get a list */rdrbaf_final_nstates*_smp.npz files within output_dir
    rdrbaf_files = [x for x in Path(output_dir).rglob("rdrbaf_final_nstates*_smp.npz")]
    df = []
    for file in rdrbaf_files:
        outdir = file.parent
        res_combine = dict(np.load(str(file)), allow_pickle=True)
        df.append(
            pd.DataFrame(
                {"outdir": str(outdir), "log-likelihood": res_combine["total_llf"]},
                index=[0],
            )
        )
    df = pd.concat(df, ignore_index=True)
    idx = np.argmax(df["log-likelihood"])
    return df["outdir"].iloc[idx]
