#!/bin/python

import sys
import pysam
import pandas as pd
import subprocess


def write_merged_bam(input_bamfile_list, suffix_list, output_bam):
    fpin = pysam.AlignmentFile(input_bamfile_list[0], "rb")
    fpout = pysam.AlignmentFile(output_bam, "wb", template=fpin)
    fpin.close()
    for i, fname in enumerate(input_bamfile_list):
        fpin = pysam.AlignmentFile(fname, "rb")
        suffix = suffix_list[i]
        for read in fpin:
            if read.has_tag("CB"):
                b = read.get_tag("CB")
                read.set_tag("CB", f"{b}_{suffix}")
            fpout.write(read)
        fpin.close()
    fpout.close()


def write_merged_deconvolution(input_deconvfile_list, suffix_list, output_deconv):
    df_combined = []
    for i, fname in enumerate(input_deconvfile_list):
        suffix = suffix_list[i]
        tmpdf = pd.read_csv(fname, header=0, index_col=0, sep="\t")
        tmpdf.index = [f"{x}_{suffix}" for x in tmpdf.index]
        df_combined.append(tmpdf)
    df_combined = pd.concat(df_combined, ignore_index=False)
    df_combined.to_csv(output_deconv, header=True, index=True, sep="\t")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python merge_bamfile.py <mode> <filename_table> <output_dir>")
        print("\tmode: either 'BAM' or 'deconv' for merging BAM files or deconvolution files respectively")
    else:
        mode = sys.argv[1]
        output_dir = sys.argv[3]
        df = pd.read_csv(sys.argv[2], sep="\t", header=None, index_col=None)
        if df.shape[1] == 3:
            df.columns=["bamfilename", "suffix", "cellrangerdir"]
        else:
            df.columns=["bamfilename", "suffix", "cellrangerdir", "deconv_filename"]

        if mode == "BAM":
            input_bamfile_list = df.bamfilename.values
            suffix_list = df.suffix.values
            write_merged_bam(input_bamfile_list, suffix_list, f"{output_dir}/unsorted_possorted_genome_bam.bam")

            # samtools sort BAM file by genomic coordinate
            p = subprocess.Popen(f"samtools sort -o {output_dir}/possorted_genome_bam.bam {output_dir}/unsorted_possorted_genome_bam.bam", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out,err = p.communicate()

            # index sorted BAM file
            p = subprocess.Popen(f"samtools index {output_dir}/possorted_genome_bam.bam", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out,err = p.communicate()

        elif mode == "deconv":
            # merge deconvolution file
            assert "deconv_filename" in df.columns
            input_deconvfile_list = df.deconv_filename.values
            suffix_list = df.suffix.values
            write_merged_deconvolution(input_deconvfile_list, suffix_list, f"{output_dir}/merged_deconvolution.tsv")