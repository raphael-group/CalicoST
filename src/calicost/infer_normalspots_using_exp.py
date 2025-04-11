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
from pathlib import Path
import functools
import subprocess
import argparse

from calicost.arg_parse import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.utils_distribution_fitting import *
from calicost.utils_hmrf import *
from calicost.hmrf import *
from calicost.phasing import *
from calicost.utils_IO import *
from calicost.find_integer_copynumber import *
from calicost.parse_input import *
from calicost.utils_plotting import *


def special_load_data(input_filelist, spaceranger_dir, hgtable_file, min_genes, min_snpumi_perspot, min_percent_expressed_spots, secondary_min_umi):
    if not input_filelist is None:
        adata, across_slice_adjacency_mat = load_joint_data_exp_only(input_filelist, [], None, None, None, min_snpumi_perspot, min_percent_expressed_spots)
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
        adata = load_data_exp_only(spaceranger_dir, None, None, None, min_snpumi_perspot, min_percent_expressed_spots)
        adata.obs["sample"] = "unique_sample"
        sample_list = [adata.obs["sample"][0]]
        sample_ids = np.zeros(adata.shape[0], dtype=int)
        across_slice_adjacency_mat = None

    coords = adata.obsm["X_pos"]

    single_tumor_prop = None

    # binning
    df_gene_snp = create_genetable_exp_only(hgtable_file, adata)
    df_gene_snp = create_bin_ranges_exp_only(df_gene_snp, adata, min_genes, secondary_min_umi=secondary_min_umi)
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat = summarize_counts_for_blocks(df_gene_snp, \
            adata, np.empty(0), np.empty(0), np.empty(0), nu=None, logphase_shift=None, geneticmap_file=None)
    return adata, sample_list, sample_ids, coords, df_gene_snp, lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, single_tumor_prop


def special_plotting_spatial(coords, hue, palette, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], hue=hue, s=15, linewidth=0, ax=ax, palette=palette)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.axis('off')
    fig.tight_layout()
    return fig


def main(input_filelist, spaceranger_dir, hgtable_file, output_prefix, min_genes, min_snpumi_perspot, min_percent_expressed_spots, secondary_min_umi):

    ########## Proprocess input data ##########
    adata, sample_list, sample_ids, coords, df_gene_snp, lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, single_tumor_prop = special_load_data(
        input_filelist,
        spaceranger_dir,
        hgtable_file,
        min_genes,
        min_snpumi_perspot,
        min_percent_expressed_spots,
        secondary_min_umi
        )
        
    ########## infer normal spots ##########
    # preprocessing and leiden clustering
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=np.median(adata.X.sum(axis=1).A1))
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    passed_resol = []
    for resol in np.arange(1.0, 0.4, -0.1):
        sc.tl.leiden(adata, resolution=resol)
        if len(adata.obs.leiden.unique()) >= 3:
            passed_resol.append(resol)

    sc.tl.leiden(adata, resolution=np.min(passed_resol))
    logging.info(f'Using resolution parameter {np.min(passed_resol)} in leiden clustering, generating {len(adata.obs.leiden.unique())} clusters.')

    # summary statistics for each leiden cluster
    sorted_cluster_ids = np.sort(adata.obs.leiden.unique())

    # average log UMI per cluster
    log_umis = [ np.log1p(adata[adata.obs.leiden==c].layers['count'].sum(axis=1)).mean() for c in sorted_cluster_ids ]

    # average log number of expressed genes per cluster
    log_num_expressedgenes = [ np.log1p(np.sum(adata[adata.obs.leiden==c].layers['count']>0, axis=1)).mean() for c in sorted_cluster_ids ]

    # entropy of gene expression across genes per cluster
    cluster_counts = np.array([ adata[adata.obs.leiden==c].layers['count'].sum(axis=0) for c in sorted_cluster_ids ]) # shape: (n_clusters, n_genes)
    proportion_genes = cluster_counts / cluster_counts.sum(axis=1,keepdims=True)
    EPS = -50
    entropy = -np.sum(proportion_genes * np.where(proportion_genes==0, EPS, np.log(proportion_genes)), axis=1)

    #  using binned counts in single_X, standard deviation of log-transformed normalized UMI counts per cluster
    cluster_X = np.array([ single_X[:, 0, adata.obs.leiden==c].sum(axis=1) for c in sorted_cluster_ids ]) # shape: (n_clusters, n_bins)
    log_normalized_cluster_X = np.log1p(cluster_X / cluster_X.sum(axis=0,keepdims=True))
    std_cluster_X = np.std(log_normalized_cluster_X, axis=1)

    # plot leiden cluster and summary statistics
    fig = special_plotting_spatial(coords, adata.obs.leiden, 'Set2', 'Leiden cluster')
    fig.savefig(f'{output_prefix}_leiden_cluster.png')
    plt.close(fig)

    fig = special_plotting_spatial(coords, adata.obs.leiden.map(dict(zip(sorted_cluster_ids, log_umis))).astype(float), 'coolwarm', 'Average log UMI')
    fig.savefig(f'{output_prefix}_log_umis.png')
    plt.close(fig)

    fig = special_plotting_spatial(coords, adata.obs.leiden.map(dict(zip(sorted_cluster_ids, log_num_expressedgenes))).astype(float), 'coolwarm', 'Average log number of expressed genes')
    fig.savefig(f'{output_prefix}_log_num_expressedgenes.png')
    plt.close(fig)

    fig = special_plotting_spatial(coords, adata.obs.leiden.map(dict(zip(sorted_cluster_ids, entropy))).astype(float), 'coolwarm', 'Entropy of gene expression')
    fig.savefig(f'{output_prefix}_entropy.png')
    plt.close(fig)

    fig = special_plotting_spatial(coords, adata.obs.leiden.map(dict(zip(sorted_cluster_ids, std_cluster_X))).astype(float), 'coolwarm', 'Standard deviation of log-normalized UMI counts per bin')
    fig.savefig(f'{output_prefix}_std_cluster_X.png')
    plt.close(fig)

    # Select the cluster with the max/min of the above summary statistics
    # - log_umis: min (assuming tumor spots have higher UMI counts)
    # - log_num_expressedgenes: min (assuming tumor spots express more genes)
    # - entropy: min (assuming tumor spots express more diverse genes)
    # - std_cluster_X: min (assuming tumor spots have more heterogeneous UMIs)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # selected normal spots based on log_umis
    np.savetxt(f'{output_prefix}_normalidx_from_log_umis.tsv', adata.obs[adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(log_umis)]])].index.values, fmt='%s')
    seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], hue=adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(log_umis)]]), s=15, linewidth=0, ax=axes[0], palette=seaborn.color_palette(['lightgrey', 'red']))
    axes[0].set_title('Normal spots based on log UMI')

    # selected normal spots based on log_num_expressedgenes
    np.savetxt(f'{output_prefix}_normalidx_from_log_num_expressedgenes.tsv', adata.obs[adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(log_num_expressedgenes)]])].index.values, fmt='%s')
    seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], hue=adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(log_num_expressedgenes)]]), s=15, linewidth=0, ax=axes[1], palette=seaborn.color_palette(['lightgrey', 'red']))
    axes[1].set_title('Normal spots based on log number of expressed genes')

    # selected normal spots based on entropy
    np.savetxt(f'{output_prefix}_normalidx_from_entropy.tsv', adata.obs[adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(entropy)]])].index.values, fmt='%s')
    seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], hue=adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(entropy)]]), s=15, linewidth=0, ax=axes[2], palette=seaborn.color_palette(['lightgrey', 'red']))
    axes[2].set_title('Normal spots based on entropy')

    # selected normal spots based on std_cluster_X
    np.savetxt(f'{output_prefix}_normalidx_from_std_cluster_X.tsv', adata.obs[adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(std_cluster_X)]])].index.values, fmt='%s')
    seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], hue=adata.obs.leiden.isin([sorted_cluster_ids[np.argmin(std_cluster_X)]]), s=15, linewidth=0, ax=axes[3], palette=seaborn.color_palette(['lightgrey', 'red']))
    axes[3].set_title('Normal spots based on std log-normalized UMI')
    fig.tight_layout()
    fig.savefig('selected_normal_spots.png')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer normal spots using expression data')
    # input_filelist and spaceranger_dir are mutually exclusive, add them as mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_filelist', type=str, help='Input file list')
    group.add_argument('--spaceranger_dir', type=str, help='spaceranger output directory')

    parser.add_argument('--hgtable_file', type=str, help='Human genome table file')
    parser.add_argument('--min_genes', type=int, default=3, help='Minimum number of genes per spot')
    parser.add_argument('--min_snpumi_perspot', type=int, default=50, help='Minimum number of SNP UMI per spot')
    parser.add_argument('--min_percent_expressed_spots', type=float, default=0.005, help='Minimum percentage of expressed spots per gene')
    parser.add_argument('--secondary_min_umi', type=int, default=400, help='Secondary minimum number of UMI per spot')
    parser.add_argument('--output_prefix', type=str, help='Output prefix (including path).')
    args = parser.parse_args()

    main(args.input_filelist, args.spaceranger_dir, args.hgtable_file, args.output_prefix, args.min_genes, args.min_snpumi_perspot, args.min_percent_expressed_spots, args.secondary_min_umi)
