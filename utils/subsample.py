import sys
import numpy as np
import scipy
import pandas as pd
from pathlib import Path
import scanpy as sc
import argparse


def subsample_cells_multi(input_filelist, outputdir, n_cells, random_seed=0):
    """
    Sub-sample cells from a list of spaceranger output directories.

    Parameters
    ----------
    input_filelist : str
        Path to a file containing a list of spaceranger output directories. Columns: bam, sample_id, spaceranger_dir

    outputdir : str
        Directory to save the subsampled h5ad files.

    n_cells : int
        Number of cells to subsample.
    """
    df_meta = pd.read_csv(input_filelist, sep="\t", header=None)
    df_meta.rename(columns=dict(zip( df_meta.columns[:3], ["bam", "sample_id", "spaceranger_dir"] )), inplace=True)
    
    # get the full list of barcodes by reading tissue_positions.csv or tissue_positions_list.csv file from each spaceranger directory
    df_pos = []
    for i, spaceranger in enumerate(df_meta.spaceranger_dir.values):
        pos_file = Path(spaceranger) / "spatial" / "tissue_positions.csv"
        if not pos_file.exists():
            pos_file = Path(spaceranger) / "spatial" / "tissue_positions_list.csv"
            this_pos = pd.read_csv(pos_file, sep=",", header=None, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
        else:
            this_pos = pd.read_csv(pos_file, sep=",", header=0, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
        
        this_pos = this_pos[this_pos.in_tissue == True]
        this_pos['sampleid'] = df_meta.sample_id.values[i]

        this_pos.barcode = this_pos.barcode + '_' + this_pos['sampleid']
        df_pos.append(this_pos)

    df_pos = pd.concat(df_pos, ignore_index=True)

    # compute the number of cells to subsample from each sample based on the number of cells in each sample
    n_cells_per_sample = np.round(n_cells * df_pos.groupby('sampleid').size() / df_pos.shape[0]).astype(int)

    # subsample cells randomly uniformly from each sample
    np.random.seed(random_seed)
    df_sampled = []
    for i, sampleid in enumerate(df_meta.sample_id.values):
        this_sample = df_pos[df_pos.sampleid == sampleid].sample(n=n_cells_per_sample[i], replace=False)
        this_sample.sort_index(inplace=True)
        df_sampled.append(this_sample)
    df_sampled = pd.concat(df_sampled, ignore_index=True)

    # write a new h5ad file with the subsampled cells and the corresponding tissue_positions.csv file
    for i, spaceranger in enumerate(df_meta.spaceranger_dir.values):
        if Path(f'{spaceranger}/filtered_feature_bc_matrix.h5').exists():
            adata = sc.read_10x_h5(f'{spaceranger}/filtered_feature_bc_matrix.h5', gex_only=False)
        else:
            adata = sc.read_h5ad(f'{spaceranger}/filtered_feature_bc_matrix.h5ad')
        this_pos = df_sampled[df_sampled.sampleid == df_meta.sample_id.values[i]].drop(columns='sampleid')
        this_pos.barcode = this_pos.barcode.str.split('_').str[0]
        adata = adata[this_pos.barcode.values, :]
        # write the subsampled h5ad file
        # mkdir
        sample_outdir = Path(outputdir) / df_meta.sample_id.values[i]
        sample_outdir.mkdir(parents=True, exist_ok=True)
        adata.write(sample_outdir / "filtered_feature_bc_matrix.h5ad")
        # write the subsampled tissue_positions.csv file
        sample_outdir2 = sample_outdir / 'spatial'
        sample_outdir2.mkdir(parents=True, exist_ok=True)
        this_pos.to_csv(sample_outdir2 / "tissue_positions.csv", sep=",", index=False, header=True)

    return df_sampled


def subsample_cells_single(spaceranger_dir, outputdir, n_cells, random_seed=0):
    """
    Sub-sample cells from a single spaceranger output directory.

    Parameters
    ----------
    spaceranger_dir : str
        Path to the spaceranger output directory.
    outputdir : str
        Directory to save the subsampled h5ad files.
    n_cells : int
        Number of cells to subsample.
    """
    # get the list of spot barcodes by reading the tissue_positions.csv or tissue_positions_list.csv file
    pos_file = Path(spaceranger_dir) / "spatial" / "tissue_positions.csv"
    if not pos_file.exists():
        pos_file = Path(spaceranger_dir) / "spatial" / "tissue_positions_list.csv"
        df_pos = pd.read_csv(pos_file, sep=",", header=None, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    else:
        df_pos = pd.read_csv(pos_file, sep=",", header=0, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    df_pos = df_pos[df_pos.in_tissue == True]
        
    # subsample cells randomly uniformly from each sample
    np.random.seed(random_seed)
    df_sampled = df_pos.sample(n=n_cells, replace=False)
    df_sampled.sort_index(inplace=True)

    # write a new h5ad file with the subsampled cells and the corresponding tissue_positions.csv file
    if Path(f'{spaceranger_dir}/filtered_feature_bc_matrix.h5').exists():
        adata = sc.read_10x_h5(f'{spaceranger_dir}/filtered_feature_bc_matrix.h5', gex_only=False)
    else:
        adata = sc.read_h5ad(f'{spaceranger_dir}/filtered_feature_bc_matrix.h5ad')
    adata = adata[df_sampled.barcode.values, :]
    # write the subsampled h5ad file
    # mkdir
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    adata.write(outputdir / "filtered_feature_bc_matrix.h5ad")
    # write the subsampled tissue_positions.csv file
    outputdir2 = outputdir / 'spatial'
    outputdir2.mkdir(parents=True, exist_ok=True)
    df_sampled.to_csv(outputdir2 / "tissue_positions.csv", sep=",", index=False, header=True)

    return df_sampled


def subsample_snps(snp_dir, output_snp_dir, df_sampled, n_snps, random_seed=0):
    """
    Sub-sample SNPs from a list of spaceranger output directories.

    Parameters
    ----------
    snp_dir : str
        Path to the directory containing the SNP files.

    output_snp_dir : str
        Directory to save the subsampled SNP files.

    df_sampled : pd.DataFrame
        Dataframe containing the subsampled barcodes.
    """
    snp_dir = Path(snp_dir)
    cell_snp_Aallele = scipy.sparse.load_npz(snp_dir / "cell_snp_Aallele.npz")
    cell_snp_Ballele = scipy.sparse.load_npz(snp_dir / "cell_snp_Ballele.npz")
    barcodes = np.loadtxt(snp_dir / "barcodes.txt", dtype=str)
    unique_snp_ids = np.load(snp_dir / "unique_snp_ids.npy", allow_pickle=True)

    # first get the indices of the barcodes to keep. The order of barcodes in df_sampled should be retained
    map_barcodes_index = {x:i for i, x in enumerate(barcodes)}
    idx = np.array([map_barcodes_index[x] for x in df_sampled.barcode.values if x in map_barcodes_index])
    cell_snp_Aallele = cell_snp_Aallele[idx, :]
    cell_snp_Ballele = cell_snp_Ballele[idx, :]
    barcodes = barcodes[idx]

    # subsample SNPs
    np.random.seed(random_seed)
    idx = np.random.choice(cell_snp_Aallele.shape[1], n_snps, replace=False)
    idx = np.sort(idx)
    cell_snp_Aallele = cell_snp_Aallele[:, idx]
    cell_snp_Ballele = cell_snp_Ballele[:, idx]
    unique_snp_ids = unique_snp_ids[idx]

    # write the subsampled SNP files
    output_snp_dir = Path(output_snp_dir)
    output_snp_dir.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(output_snp_dir / "cell_snp_Aallele.npz", cell_snp_Aallele)
    scipy.sparse.save_npz(output_snp_dir / "cell_snp_Ballele.npz", cell_snp_Ballele)
    np.savetxt(output_snp_dir / "barcodes.txt", barcodes, fmt='%s')
    np.save(output_snp_dir / "unique_snp_ids.npy", unique_snp_ids)

    return


def main(args):
    # subsample cells: call either subsample_cells_multi or subsample_cells_single based on whether the input_filelist or spaceranger_dir is provided
    if args.input_filelist is not None:
        df_sampled = subsample_cells_multi(args.input_filelist, args.output_gex_dir, args.n_cells)
    else:
        df_sampled = subsample_cells_single(args.spaceranger_dir, args.output_gex_dir, args.n_cells)
    
    # subsample SNPs
    subsample_snps(args.snp_dir, args.output_snp_dir, df_sampled, args.n_snps)

    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subsample cells and SNPs from a list of spaceranger output directories.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_filelist', type=str, help='Path to a file containing a list of spaceranger output directories. Columns: bam, sample_id, spaceranger_dir')
    group.add_argument('--spaceranger_dir', type=str, help='Path to the spaceranger output directory.')

    parser.add_argument('--snp_dir', type=str, help='Path to the directory containing the SNP files.')
    parser.add_argument('--output_gex_dir', type=str, help='Directory to save the subsampled h5ad files.')
    parser.add_argument('--output_snp_dir', type=str, help='Directory to save the subsampled SNP files.')
    parser.add_argument('--n_cells', type=int, help='Number of cells to subsample.')
    parser.add_argument('--n_snps', type=int, help='Number of SNPs to subsample.')

    args = parser.parse_args()

    main(args)
