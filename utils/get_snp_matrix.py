#!/bin/python

import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import scipy.io
from pathlib import Path
import json
import gzip
import pickle
from tqdm import trange
import copy
import argparse


def process_snp_phasing(cellsnp_folder, eagle_folder, outputfile):
    # create a (snp_id, GT) map from eagle2 output
    snp_gt_map = {}
    for c in range(1, 23):
        fname = [str(x) for x in Path(eagle_folder).glob("*chr{}.phased.vcf.gz".format(c))]
        assert len(fname) > 0
        fname = fname[0]
        tmpdf = pd.read_table(fname, compression = 'gzip', comment = '#', sep="\t", names=["CHR","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","PHASE"])
        this_snp_ids = [ "{}_{}_{}_{}".format(c, row.POS, row.REF, row.ALT) for i,row in tmpdf.iterrows() ]
        this_gt = list(tmpdf.iloc[:,-1])
        assert len(this_snp_ids) == len(this_gt)
        snp_gt_map.update( {this_snp_ids[i]:this_gt[i] for i in range(len(this_gt))} )
    # cellsnp DP (read depth) and AD (alternative allele depth)
    # first get a list of snp_id and spot barcodes
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.base.vcf.gz", header=1, sep="\t")
    snp_list = np.array([ "{}_{}_{}_{}".format(row["#CHROM"], row.POS, row.REF, row.ALT) for i,row in tmpdf.iterrows() ])
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.samples.tsv", header=None)
    sample_list = np.array(list(tmpdf.iloc[:,0]))
    # then get the DP and AD matrix
    DP = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.DP.mtx").tocsr()
    AD = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.AD.mtx").tocsr()
    # remove SNPs that are not phased
    is_phased = np.array([ (x in snp_gt_map) for x in snp_list ])
    DP = DP[is_phased,:]
    AD = AD[is_phased,:]
    snp_list = snp_list[is_phased]
    # generate a new dataframe with columns (cell, snp_id, DP, AD, CHROM, POS, GT)
    rows, cols = DP.nonzero()
    cell = sample_list[cols]
    snp_id = snp_list[rows]
    DP_df = DP[DP.nonzero()].A.flatten()
    AD_df = AD[DP.nonzero()].A.flatten()
    GT = [snp_gt_map[x] for x in snp_id]
    df = pd.DataFrame({"cell":cell, "snp_id":snp_id, "DP":DP_df, "AD":AD_df, \
                       "CHROM":[int(x.split("_")[0]) for x in snp_id], "POS":[int(x.split("_")[1]) for x in snp_id], "GT":GT})
    df.to_csv(outputfile, sep="\t", index=False, header=True, compression={'method': 'gzip'})
    return df


def read_cell_by_snp(allele_counts_file):
    df = pd.read_csv(allele_counts_file, sep="\t", header=0)
    index = np.array([i for i,x in enumerate(df.GT) if x=="0|1" or x=="1|0"])
    df = df.iloc[index, :]
    df.CHROM = df.CHROM.astype(int)
    return df


def cell_by_gene_lefthap_counts(cellsnp_folder, eagle_folder, barcode_list):
    # create a (snp_id, GT) map from eagle2 output
    snp_gt_map = {}
    for c in range(1, 23):
        fname = [str(x) for x in Path(eagle_folder).glob("*chr{}.phased.vcf.gz".format(c))]
        assert len(fname) > 0
        fname = fname[0]
        tmpdf = pd.read_table(fname, compression = 'gzip', comment = '#', sep="\t", names=["CHR","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","PHASE"])
        # only keep heterozygous SNPs
        tmpdf = tmpdf[ (tmpdf.PHASE=="0|1") | (tmpdf.PHASE=="1|0") ]
        this_snp_ids = (str(c) + "_" + tmpdf.POS.astype(str) +"_"+  tmpdf.REF +"_"+ tmpdf.ALT).values
        this_gt = tmpdf.PHASE.values
        assert len(this_snp_ids) == len(this_gt)
        snp_gt_map.update( {this_snp_ids[i]:this_gt[i] for i in range(len(this_gt))} )

    # cellsnp-lite output
    cellsnp_base = [str(x) for x in Path(cellsnp_folder).glob("cellSNP.base*")][0]
    df_snp = pd.read_csv(cellsnp_base, comment="#", sep="\t", names=["tmpCHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
    df_snp['snp_id'] = df_snp.tmpCHR.astype(str) + "_" + df_snp.POS.astype(str) + "_" + df_snp.REF + "_" + df_snp.ALT
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.samples.tsv", header=None)
    sample_list = np.array(list(tmpdf.iloc[:,0]))
    barcode_mapper = {x:i for i,x in enumerate(sample_list)}
    # DP and AD
    DP = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.DP.mtx").tocsr()
    AD = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.AD.mtx").tocsr()
    # retain only SNPs that are phased
    is_phased = (df_snp.snp_id.isin(snp_gt_map)).values
    df_snp = df_snp[is_phased]
    df_snp['GT'] = [snp_gt_map[x] for x in df_snp.snp_id]
    DP = DP[is_phased,:]
    AD = AD[is_phased,:]

    # phasing
    phased_AD = np.where( (df_snp.GT.values == "0|1").reshape(-1,1), AD.A, (DP-AD).A )
    phased_AD = scipy.sparse.csr_matrix(phased_AD)

    # re-order based on barcode_list
    index = np.array([barcode_mapper[x] for x in barcode_list if x in barcode_mapper])
    DP = DP[:, index]
    phased_AD = phased_AD[:, index]    
    
    # returned matrix has shape (N_cells, N_snps), which is the transpose of the original matrix
    return (DP-phased_AD).T, phased_AD.T, df_snp.snp_id.values


def cell_by_gene_lefthap_counts_v2(df_cell_snp, hg_table_file, gene_list, barcode_list):
    # index of genes and barcodes in the current gene expression matrix
    barcode_mapper = {x:i for i,x in enumerate(barcode_list)}
    gene_mapper = {x:i for i,x in enumerate(gene_list)}
    # make an numpy array for CHROM and POS in df_cell_snp
    cell_snp_CHROM = np.array(df_cell_snp.CHROM)
    cell_snp_POS = np.array(df_cell_snp.POS)
    # read gene ranges in genome
    # NOTE THAT THE FOLLOWING CODE REQUIRES hg_table_file IS SORTED BY GENOMIC POSITION!
    df_genes = pd.read_csv(hg_table_file, header=0, index_col=0, sep="\t")
    index = np.array([ i for i in range(df_genes.shape[0]) if (not "_" in df_genes.chrom.iloc[i]) and \
                      (df_genes.chrom.iloc[i] != "chrX") and (df_genes.chrom.iloc[i] != "chrY") and (df_genes.chrom.iloc[i] != "chrM") and \
                      (not "GL" in df_genes.chrom.iloc[i]) and (not "KI" in df_genes.chrom.iloc[i]) ])
    df_genes = df_genes.iloc[index, :]
    tmp_gene_ranges = {df_genes.name2.iloc[i]:(int(df_genes.chrom.iloc[i][3:]), df_genes.cdsStart.iloc[i], df_genes.cdsEnd.iloc[i]) for i in np.arange(df_genes.shape[0]) }
    gene_ranges = [(gname, tmp_gene_ranges[gname]) for gname in gene_list if gname in tmp_gene_ranges]
    del tmp_gene_ranges
    # aggregate snp counts to genes
    N = np.unique(df_cell_snp.cell).shape[0]
    G = len(gene_ranges)
    i = 0
    j = 0
    cell_gene_snp_counts = []
    snp_ids = np.array(df_cell_snp.snp_id)
    unique_snp_ids = df_cell_snp.snp_id.unique()
    snp_id_mapper = {unique_snp_ids[i]:i for i in range(len(unique_snp_ids))}
    N_snps = len(unique_snp_ids)
    cell_snp_Aallele = np.zeros((len(barcode_list), N_snps))
    cell_snp_Ballele = np.zeros((len(barcode_list), N_snps))
    snp_gene_list = [""] * N_snps
    for i in trange(df_cell_snp.shape[0]):
        if df_cell_snp.GT.iloc[i] == "1|1" or df_cell_snp.GT.iloc[i] == "0|0":
            continue
        # check cell barcode
        if not df_cell_snp.cell.iloc[i] in barcode_mapper:
            continue
        cell_idx = barcode_mapper[df_cell_snp.cell.iloc[i]]
        # if the SNP is not within any genes
        if j < len(gene_ranges) and (cell_snp_CHROM[i] < gene_ranges[j][1][0] or \
                                     (cell_snp_CHROM[i] == gene_ranges[j][1][0] and cell_snp_POS[i] < gene_ranges[j][1][1])):
            continue
        # if the SNP position passes gene j
        while j < len(gene_ranges) and (cell_snp_CHROM[i] > gene_ranges[j][1][0] or \
                                        (cell_snp_CHROM[i] == gene_ranges[j][1][0] and cell_snp_POS[i] > gene_ranges[j][1][2])):
            j += 1
        # if the SNP is within gene j, add the corresponding gene ID
        if j < len(gene_ranges) and cell_snp_CHROM[i] == gene_ranges[j][1][0] and \
        cell_snp_POS[i] >= gene_ranges[j][1][1] and cell_snp_POS[i] <= gene_ranges[j][1][2]:
            snp_gene_list[ snp_id_mapper[snp_ids[i]] ] = gene_ranges[j][0]
        # add the SNP UMI count to the corresponding cell and loci
        if df_cell_snp.GT.iloc[i] == "0|1":
            cell_snp_Aallele[cell_idx, snp_id_mapper[snp_ids[i]]] = df_cell_snp.DP.iloc[i] - df_cell_snp.AD.iloc[i]
            cell_snp_Ballele[cell_idx, snp_id_mapper[snp_ids[i]]] = df_cell_snp.AD.iloc[i]
        elif df_cell_snp.GT.iloc[i] == "1|0":
            cell_snp_Aallele[cell_idx, snp_id_mapper[snp_ids[i]]] = df_cell_snp.AD.iloc[i]
            cell_snp_Ballele[cell_idx, snp_id_mapper[snp_ids[i]]] = df_cell_snp.DP.iloc[i] - df_cell_snp.AD.iloc[i]
            
    index = np.where(np.logical_and( np.sum(cell_snp_Aallele + cell_snp_Ballele, axis=0) > 0))[0]
    cell_snp_Aallele = cell_snp_Aallele[:, index].astype(int)
    cell_snp_Ballele = cell_snp_Ballele[:, index].astype(int)
    snp_gene_list = np.array(snp_gene_list)[index]
    unique_snp_ids = unique_snp_ids[index]
    return cell_snp_Aallele, cell_snp_Ballele, snp_gene_list, unique_snp_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cellsnplite_result_dir", help="cellsnplite result directory", type=str)
    parser.add_argument("-e", "--eagle_out_dir", help="eagle output directory", type=str)
    parser.add_argument("-b", "--barcodefile", help="barcode file", type=str)
    parser.add_argument("-o", "--outputdir", help="output directory", type=str)
    args = parser.parse_args()

    barcode_list = list(pd.read_csv(args.barcodefile, header=None).iloc[:,0])
    cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids = cell_by_gene_lefthap_counts(args.cellsnplite_result_dir, args.eagle_out_dir, barcode_list)

    scipy.sparse.save_npz(f"{args.outputdir}/cell_snp_Aallele.npz", cell_snp_Aallele)
    scipy.sparse.save_npz(f"{args.outputdir}/cell_snp_Ballele.npz", cell_snp_Ballele)
    np.save(f"{args.outputdir}/unique_snp_ids.npy", unique_snp_ids)
