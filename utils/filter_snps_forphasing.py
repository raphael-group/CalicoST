#!/bin/python

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def main(cellsnplite_result_dir, eagle_out_dir, vaf_threshold=0.1):
    cellsnp_base = [str(x) for x in Path(cellsnplite_result_dir).glob("cellSNP.base*")][0]
    df_snp = pd.read_csv(cellsnp_base, comment="#", sep="\t", names=["tmpCHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
    df_snp["CHROM"] = [f"chr{x}" for x in df_snp.tmpCHR]
    df_snp["AD"] = [int(x.split(";")[0].split("=")[-1]) for x in df_snp.INFO]
    df_snp["DP"] = [int(x.split(";")[1].split("=")[-1]) for x in df_snp.INFO]
    df_snp["OTH"] = [int(x.split(";")[2].split("=")[-1]) for x in df_snp.INFO]
    # remove records with DP == 0
    df_snp = df_snp[df_snp.DP > 0]
    # keep het SNP (0.1 <= AD/DP <= 0.9) and hom ALT SNP (AD == DP >= 10)
    # df_snp = df_snp[((df_snp.AD / df_snp.DP >= 0.1) & (df_snp.AD / df_snp.DP <= 0.9)) | ((df_snp.AD == df_snp.DP) & (df_snp.DP >= 10))]
    df_snp = df_snp[((df_snp.AD >= 2) & (df_snp.DP - df_snp.AD >= 2) & (df_snp.AD / df_snp.DP >= vaf_threshold) & (df_snp.AD / df_snp.DP <= 1-vaf_threshold)) | ((df_snp.AD == df_snp.DP) & (df_snp.DP >= 10)) | ((df_snp.AD == 0) & (df_snp.DP >= 10))]
    # add addition columns
    df_snp["FORMAT"] = "GT"
    # df_snp[f"{sample_id}"] = ["0/1" if row.AD < row.DP else "1/1" for i,row in df_snp.iterrows()]
    gt_column = np.array(["0/0"] * df_snp.shape[0])
    gt_column[ (df_snp.AD == df_snp.DP) ] = "1/1"
    gt_column[ (df_snp.AD > 0) & (df_snp.DP - df_snp.AD > 0) ] = "0/1"
    df_snp["SAMPLE_ID"] = gt_column
    # output chromosome to folder
    for c in range(1, 23):
        df = df_snp[ (df_snp.tmpCHR == c) | (df_snp.tmpCHR == str(c)) ]
        # remove records that have duplicated snp_id
        snp_id = [f"{row.tmpCHR}_{row.POS}_{row.REF}_{row.ALT}" for i,row in df.iterrows()]
        df["snp_id"] = snp_id
        df = df.groupby("snp_id").agg({"CHROM":"first", "POS":"first", "ID":"first", "REF":"first", "ALT":"first", "QUAL":"first", "FILTER":"first", \
                                       "INFO":"first", "FORMAT":"first", "SAMPLE_ID":"first", "AD":"sum", "DP":"sum", "OTH":"sum"})
        info = [f"AD={row.AD};DP={row.DP};OTH={row.OTH}" for i,row in df.iterrows()]
        df["INFO"] = info
        df = df[["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE_ID"]]
        df.sort_values(by="POS", inplace=True)
        fp = open(f"{eagle_out_dir}/chr{c}.vcf", 'w')
        fp.write("##fileformat=VCFv4.2\n")
        fp.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Consensus Genotype across all datasets with called genotype\">\n")
        fp.write("#" + "\t".join(df.columns) + "\n")
        df.to_csv(fp, sep="\t", index=False, header=False)
        fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cellsnplite_result_dir", help="cellsnplite result directory", type=str)
    parser.add_argument("-o", "--eagle_out_dir", help="eagle output directory", type=str)
    parser.add_argument("-v", "--vaf_threshold", help="vaf threshold", default=0.1, type=float)
    args = parser.parse_args()
    main(args.cellsnplite_result_dir, args.eagle_out_dir, args.vaf_threshold)
