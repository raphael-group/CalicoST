import sys
sys.path.append( "/".join(sys.argv[0].split("/")[:-1]) + "/../src/" )
from utils_plotting import *
from calicost_supervised import *
import numpy as np
import pandas as pd
from pathlib import Path


def get_best_r_hmrf(configuration_file):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
            
    # find the best HMRF initialization random seed
    df_3_clone = []
    for random_state in range(10):
        outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{random_state}_w{config['spatial_weight']:.1f}"
        if Path(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz").exists():
            res_combine = dict(np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz"), allow_pickle=True)
            df_3_clone.append( pd.DataFrame({"random seed":random_state, "log-likelihood":res_combine["total_llf"]}, index=[0]) )
    if len(df_3_clone) > 0:
        df_3_clone = pd.concat(df_3_clone, ignore_index=True)
        idx = np.argmax(df_3_clone["log-likelihood"])
        r_hmrf_initialization = df_3_clone["random seed"].iloc[idx]
        return r_hmrf_initialization
    else:
        return None
    

def convert_copy_to_states(A_copy, B_copy, counts=None):
    if counts is None:
        tmp = A_copy + B_copy
        tmp = tmp[~np.isnan(tmp)]
    else:
        tmp = np.concatenate([ np.ones(counts[i]) * (A_copy[i]+B_copy[i]) for i in range(len(counts)) if ~np.isnan(A_copy[i]+B_copy[i]) ])
    base_ploidy = np.median(tmp)
    coarse_states = np.array(["neutral"] * A_copy.shape[0])
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy != B_copy) ] = "del"
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy == B_copy) ] = "bdel"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy != B_copy) ] = "amp"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy == B_copy) ] = "bamp"
    coarse_states[ (A_copy + B_copy == base_ploidy) & (A_copy != B_copy) ] = "loh"
    coarse_states[coarse_states == "neutral"] = "neu"
    return coarse_states


def convert_copy_to_totalcopy_states(A_copy, B_copy, counts=None):
    if counts is None:
        tmp = A_copy + B_copy
        tmp = tmp[~np.isnan(tmp)]
    else:
        tmp = np.concatenate([ np.ones(counts[i]) * (A_copy[i]+B_copy[i]) for i in range(len(counts)) if ~np.isnan(A_copy[i]+B_copy[i]) ])
    base_ploidy = np.median(tmp)
    coarse_states = np.array(["neutral"] * A_copy.shape[0])
    coarse_states[ (A_copy + B_copy < base_ploidy) ] = "del"
    coarse_states[ (A_copy + B_copy > base_ploidy) ] = "amp"
    coarse_states[coarse_states == "neutral"] = "neu"
    return coarse_states


def read_hatchet(hatchet_wes_file, ordered_chr=[str(c) for c in range(1,23)], purity_threshold=0.3):
    ##### HATCHet2 WES #####
    df_hatchet = pd.read_csv(hatchet_wes_file, sep='\t', index_col=None, header=0)
    # rename the "#CHR" column to "CHR"
    df_hatchet = df_hatchet.rename(columns={'#CHR': 'CHR'})

    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_hatchet.CHR.isin(ordered_chr) ):
        df_hatchet["CHR"] = df_hatchet.CHR.map(lambda x: x.replace("chr", ""))
    df_hatchet = df_hatchet[df_hatchet.CHR.isin(ordered_chr)]
    df_hatchet["int_chrom"] = df_hatchet.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_hatchet = df_hatchet.sort_values(by=["int_chrom", "START"])

    # find samples in hatchet results for which tumor purity > purity_threshold
    samples = np.unique(df_hatchet["SAMPLE"].values)
    tumor_purity = np.array([1-df_hatchet[df_hatchet["SAMPLE"]==s].u_normal.values[0] for s in samples])
    df_hatchet = df_hatchet[df_hatchet["SAMPLE"].isin(samples[tumor_purity > purity_threshold])]

    # make a copy of df_hatchet for the sample with highest tumor purity
    df_wes = df_hatchet[df_hatchet["SAMPLE"]==samples[np.argmax(tumor_purity)]].copy()

    # find hatchet clone with the highest proportion within each sample (with purity > purity_threshold)
    indexes = []
    for s in samples[tumor_purity > purity_threshold]:
        hatchet_clones = [x[7:] for x in df_hatchet.columns if x.startswith("u_clone")]
        prop = [ df_hatchet[df_hatchet["SAMPLE"]==s][f"u_clone{n}"].values[0] for n in hatchet_clones ]
        indexes.append( hatchet_clones[np.argmax(prop)] )
    indexes = np.unique(indexes)

    # parse A copy and B copy for each clone for each segment
    for i,idx in enumerate(indexes):
        df_wes[f"Acopy_{i}"] = [int(x.split("|")[0]) for x in df_wes[f"cn_clone{idx}"].values]
        df_wes[f"Bcopy_{i}"] = [int(x.split("|")[1]) for x in df_wes[f"cn_clone{idx}"].values]
    return df_wes


def map_hatchet_to_bins(df_wes, sorted_chr_pos):
    # map HATCHet to allele-specific STARCH bins
    snp_seg_index = []
    j = 0
    for i in range(len(sorted_chr_pos)):
        this_chr = sorted_chr_pos[i][0]
        this_pos = sorted_chr_pos[i][1]
        while j < df_wes.shape[0] and ( (df_wes.int_chrom.iloc[j] < this_chr) or (df_wes.int_chrom.iloc[j] == this_chr and df_wes.END.iloc[j] < this_pos) ):
            j += 1
        if j < df_wes.shape[0] and df_wes.int_chrom.iloc[j] == this_chr and df_wes.START.iloc[j] <= this_pos and df_wes.END.iloc[j] >= this_pos:
            snp_seg_index.append(j)
        else:
            snp_seg_index.append(-1) # maybe the SNP is in centromere, it's not inside any segment
    snp_seg_index = np.array(snp_seg_index)
    return snp_seg_index


def stateaccuracy_allele_starch(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # starch results
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    df_starch_cnv = pd.read_csv(f"{outdir}/cnv_{midfix}seglevel.tsv", sep="\t", header=0)
    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_starch_cnv.CHR.isin(ordered_chr) ):
        df_starch_cnv["CHR"] = df_starch_cnv.CHR.map(lambda x: x.replace("chr", ""))
    df_starch_cnv = df_starch_cnv[df_starch_cnv.CHR.isin(ordered_chr)]
    df_starch_cnv["int_chrom"] = df_starch_cnv.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_starch_cnv = df_starch_cnv.sort_values(by=["int_chrom", "START"])
    sorted_chr_pos = [[df_starch_cnv.int_chrom.iloc[i], df_starch_cnv.START.iloc[i]] for i in range(df_starch_cnv.shape[0])]
    df_starch_cnv.drop(columns=["int_chrom"], inplace=True)
    clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_starch_cnv.columns[3:] ])
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    percent_category = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    for s,sid in enumerate(retained_hatchet_clones):
        coarse_states_wes = fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index], counts=df_wes.END.values-df_wes.START.values)
        # accuracy
        for c,cid in enumerate(clone_ids):
            # coarse
            coarse_states_inferred = fun_hatchetconvert(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values, counts=df_starch_cnv.END.values-df_starch_cnv.START.values)
            percent_category[c, s] = 1.0 * np.sum(coarse_states_inferred == coarse_states_wes) / len(coarse_states_inferred)
    return percent_category, sorted_chr_pos


def exactaccuracy_allele_starch(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)]):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # starch results
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    df_starch_cnv = pd.read_csv(f"{outdir}/cnv_{midfix}seglevel.tsv", sep="\t", header=0)
    df_starch_cnv.CHR = df_starch_cnv.CHR.astype(str)
    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_starch_cnv.CHR.isin(ordered_chr) ):
        df_starch_cnv["CHR"] = df_starch_cnv.CHR.map(lambda x: x.replace("chr", ""))
    df_starch_cnv = df_starch_cnv[df_starch_cnv.CHR.isin(ordered_chr)]
    df_starch_cnv["int_chrom"] = df_starch_cnv.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_starch_cnv = df_starch_cnv.sort_values(by=["int_chrom", "START"])
    sorted_chr_pos = [[df_starch_cnv.int_chrom.iloc[i], df_starch_cnv.START.iloc[i]] for i in range(df_starch_cnv.shape[0])]
    df_starch_cnv.drop(columns=["int_chrom"], inplace=True)
    clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_starch_cnv.columns[3:] ])

    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    percent_exact = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    for s,sid in enumerate(retained_hatchet_clones):
        minor_copy_wes = np.minimum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        major_copy_wes = np.maximum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        for c, cid in enumerate(clone_ids):
            # exact
            minor_copy_inferred = np.minimum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            major_copy_inferred = np.maximum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            percent_exact[c, s] = 1.0 * np.sum((minor_copy_inferred==minor_copy_wes) & (major_copy_inferred==major_copy_wes)) / len(minor_copy_inferred)
    return percent_exact, sorted_chr_pos



def precision_recall_genelevel_allele_starch(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)]):
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # calicost results
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    df_calicost = pd.read_csv(f"{outdir}/cnv_{midfix}genelevel.tsv", sep="\t", header=0, index_col=0)
    calico_clones = [x.split(" ")[0][5:] for x in df_calicost.columns if x.endswith(" A")]
    for c in calico_clones:
        tmp = convert_copy_to_states(df_calicost[f"clone{c} A"].values, df_calicost[f"clone{c} B"].values)
        tmp[tmp == "bdel"] = "del"
        tmp[tmp == "bamp"] = "amp"
        df_calicost[f"srt_cnstate_clone{c}"] = tmp

    # read hatchet results and join tables
    df_hatchet = pd.read_csv(hatchet_wes_file, header=0, index_col=0, sep="\t")
    hatchet_clones = [x[13:] for x in df_hatchet.columns if x.startswith("cnstate_clone")]
    df_calicost = df_calicost.join( df_hatchet[[x for x in df_hatchet.columns if "cnstate_clone" in x]], how="inner" )

    # precision and recall of DEL, AMP, LOH for each pair of clones
    df_accuracy = []
    for event in ["del", "amp", "loh"]:
        for c in calico_clones:
            for s in hatchet_clones:
                precision = np.sum( (df_calicost[f"srt_cnstate_clone{c}"].values == event) & (df_calicost[f"cnstate_clone{s}"].values == event) ) / np.sum(df_calicost[f"srt_cnstate_clone{c}"].values == event)
                recall = np.sum( (df_calicost[f"srt_cnstate_clone{c}"].values == event) & (df_calicost[f"cnstate_clone{s}"].values == event) ) / np.sum(df_calicost[f"cnstate_clone{s}"].values == event)
                df_accuracy.append( pd.DataFrame({"clone_pair":f"{c},{s}", "event":event, "acc":[precision, recall], "acc_name":["precision", "recall"]}) )
            
    df_accuracy = pd.concat(df_accuracy, ignore_index=True)
    return df_accuracy


def plot_hatchet_acn(hatchet_dir, hatchet_cnfile, out_file, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = pd.read_csv(f"{hatchet_dir}/results/{hatchet_cnfile}", sep='\t', index_col=None, header=0)
    # rename the "#CHR" column to "CHR"
    df_hatchet = df_hatchet.rename(columns={'#CHR': 'CHR'})

    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_hatchet.CHR.isin(ordered_chr) ):
        df_hatchet["CHR"] = df_hatchet.CHR.map(lambda x: x.replace("chr", ""))
    df_hatchet = df_hatchet[df_hatchet.CHR.isin(ordered_chr)]
    df_hatchet["int_chrom"] = df_hatchet.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_hatchet = df_hatchet.sort_values(by=["int_chrom", "START"])

    # create another dataframe df_cnv, and apply to plot_acn_from_df function for plotting
    df_cnv = df_hatchet[["CHR", "START", "END"]]
    # for each clone in df_hatchet, split the A allele copy and B allele copy to separate columns
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]
    clone_names = []
    for n in hatchet_clones:
        A_copy = [int(x.split("|")[0]) for x in df_hatchet[f"cn_clone{n}"]]
        B_copy = [int(x.split("|")[1]) for x in df_hatchet[f"cn_clone{n}"]]
        df_cnv[f"clone{n} A"] = A_copy
        df_cnv[f"clone{n} B"] = B_copy
        prop = df_hatchet[f"u_clone{n}"].values[0]
        clone_names.append(f"clone{n} ({prop:.2f})")

    # plot
    fig, axes = plt.subplots(1, 1, figsize=(20, 0.6*len(hatchet_clones)+0.4), dpi=200, facecolor="white")
    plot_acn_from_df(df_cnv, axes, clone_ids=hatchet_clones, clone_names=clone_names, add_chrbar=True, chrbar_thickness=0.4/(0.6*len(hatchet_clones) + 0.4), add_legend=True, remove_xticks=True)
    fig.tight_layout()
    fig.savefig(out_file, transparent=True, bbox_inches="tight")


def add_cn_state(df_hatchet):
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]
    assert ("START" in df_hatchet.columns) and ("END" in df_hatchet.columns)
    for n in hatchet_clones:
        # compute total copy number per bin
        total_cn = np.array([ int(x.split("|")[0]) + int(x.split("|")[1]) for x in df_hatchet[f"cn_clone{n}"] ])
        # check whether each bin is homozygous, i.e., either first cn is 0 or second cn is 0
        is_homozygous = np.array([ (int(x.split("|")[0])==0) or (int(x.split("|")[1])==0) for x in df_hatchet[f"cn_clone{n}"] ])
        # compute median total copy number weighted by END - START
        median_cn = np.median(np.concatenate([ np.ones(df_hatchet.END.values[i]-df_hatchet.START.values[i]) * x for i,x in enumerate(total_cn) ]))
        # get cn_state vector such that total_cn > median_cn is "amp", total_cn < median_cn is "del", and total_cn == median_cn is "neu"
        cn_state = np.array(["neu"] * len(total_cn))
        cn_state[total_cn > median_cn] = "amp"
        cn_state[total_cn < median_cn] = "del"
        cn_state[ (total_cn == median_cn) & (is_homozygous) ] = "loh"
        # add cn_state to df_hatchet
        df_hatchet[f"cnstate_clone{n}"] = cn_state
    return df_hatchet


def map_hatchet_to_genes(hatchet_resultfile, hgtable_file, out_file, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = read_hatchet(hatchet_resultfile)
    if df_hatchet.shape[0] == 0:
        return
    # get CN state from integer copy numbers
    df_hatchet = add_cn_state(df_hatchet)
    # hatchet clones
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]

    # read hgtable file
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    df_genes = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
    if ~np.any( df_genes["chrom"].map(str).isin(ordered_chr) ):
        df_genes["chrom"] = df_genes["chrom"].map(lambda x: x.replace("chr", ""))
    df_genes = df_genes[df_genes.chrom.isin(ordered_chr)]
    df_genes["int_chrom"] = df_genes.chrom.map(ordered_chr_map)
    df_genes.sort_values(by=["int_chrom", "cdsStart"], inplace=True)

    # find the row corresponding to each gene in df_genes
    gene_names = []
    gene_cns = []
    gene_states = []
    b = 0
    for g in range(df_genes.shape[0]):
        this_chrom = df_genes.int_chrom.values[g]
        this_start = df_genes.cdsStart.values[g]
        this_end = df_genes.cdsEnd.values[g]
        while b < df_hatchet.shape[0] and (df_hatchet.int_chrom.values[b] < this_chrom or (df_hatchet.int_chrom.values[b] == this_chrom and df_hatchet.END.values[b] < this_start)):
            b += 1
        if b < df_hatchet.shape[0] and df_hatchet.int_chrom.values[b] == this_chrom and df_hatchet.START.values[b] <= this_start and df_hatchet.END.values[b] >= this_end:
            gene_names.append(df_genes.name2.values[g])
            gene_cns.append( np.array([ df_hatchet[f"cn_clone{c}"].values[b] for c in hatchet_clones ]) )
            gene_states.append( np.array([ df_hatchet[f"cnstate_clone{c}"].values[b] for c in hatchet_clones ]) )
    df_hatchet_genes = pd.DataFrame( np.hstack([ np.array(gene_cns), np.array(gene_states) ]), index=gene_names, columns=[f"cn_clone{x}" for x in hatchet_clones] + [f"cnstate_clone{x}" for x in hatchet_clones])
    df_hatchet_genes.to_csv(out_file, sep="\t", header=True, index=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python plot_hatchet.py <hatchet_dir> <hatchet_cnfile> <out_file>")
        sys.exit(1)
    
    hatchet_dir = sys.argv[1]
    hatchet_cnfile = sys.argv[2]
    out_plot = sys.argv[3]
    out_file = sys.argv[4]
    plot_hatchet_acn(hatchet_dir, hatchet_cnfile, out_plot)
    # output gene-level file
    hgtable_file = "/home/congma/congma/codes/locality-clustering-cnv/data/hgTables_hg38_gencode.txt"
    map_hatchet_to_genes(f"{hatchet_dir}/results/{hatchet_cnfile}", hgtable_file, out_file)