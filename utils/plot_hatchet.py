import sys
sys.path.append( "/".join(sys.argv[0].split("/")[:-1]) + "/../src/" )
from calicost.utils_plotting import *
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
    

def strict_convert_copy_to_states(A_copy, B_copy, counts=None):
    if counts is None:
        tmp = A_copy + B_copy
        tmp = tmp[~np.isnan(tmp)]
    else:
        tmp = np.concatenate([ np.ones(counts[i]) * (A_copy[i]+B_copy[i]) for i in range(len(counts)) if ~np.isnan(A_copy[i]+B_copy[i]) ])
    base_ploidy = np.median(tmp)
    is_homozygous = (A_copy == 0) | (B_copy == 0)
    coarse_states = np.array(["neutral"] * A_copy.shape[0])
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy != B_copy) ] = "del"
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy == B_copy) ] = "bdel"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy != B_copy) ] = "amp"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy == B_copy) ] = "bamp"
    coarse_states[ (A_copy + B_copy == base_ploidy) & (is_homozygous) ] = "loh"
    coarse_states[coarse_states == "neutral"] = "neu"
    return coarse_states


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


def fixed_convert_copy_to_states(A_copy, B_copy, counts=None):
    base_ploidy = 2
    is_homozygous = (A_copy == 0) | (B_copy == 0)
    coarse_states = np.array(["neutral"] * A_copy.shape[0])
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy != B_copy) ] = "del"
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy == B_copy) ] = "bdel"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy != B_copy) ] = "amp"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy == B_copy) ] = "bamp"
    coarse_states[ (A_copy + B_copy == base_ploidy) & (is_homozygous) ] = "loh"
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


def fixed_convert_copy_to_totalcopy_states(A_copy, B_copy, counts=None):
    base_ploidy = 2
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

    # find tumor clones that are > purity_threshold in any samples
    samples = np.unique(df_hatchet["SAMPLE"].values)
    passed_hatchet_clones = []
    for x in df_hatchet.columns:
        if not x.startswith("u_clone"):
            continue
        pu = np.array([ df_hatchet[df_hatchet["SAMPLE"]==s][f"u_clone{x[7:]}"].values[0] for s in samples ])
        if np.max(pu) > purity_threshold:
            passed_hatchet_clones.append(x[7:])

    # get the sample with largest tumor purity
    sample_tumor_purity = np.array([1-df_hatchet[df_hatchet["SAMPLE"]==s].u_normal.values[0] for s in samples])
    df_wes = df_hatchet[df_hatchet["SAMPLE"]==samples[np.argmax(sample_tumor_purity)]].copy()

    # drop the clones in df_wes that are not in passed_hatchet_clones
    columns_to_drop = [x for x in df_wes.columns if x.startswith("u_clone") and x[7:] not in passed_hatchet_clones] + \
                      [x for x in df_wes.columns if x.startswith("cn_clone") and x[8:] not in passed_hatchet_clones]
    df_wes = df_wes.drop(columns=columns_to_drop)
    hatchet_clones = [x[7:] for x in df_wes.columns if x.startswith("u_clone")]

    # parse A copy and B copy for each clone for each segment
    for c in hatchet_clones:
        df_wes[f"Acopy_{c}"] = [int(x.split("|")[0]) for x in df_wes[f"cn_clone{c}"].values]
        df_wes[f"Bcopy_{c}"] = [int(x.split("|")[1]) for x in df_wes[f"cn_clone{c}"].values]
    
    if len(passed_hatchet_clones) > 0:
        return df_wes
    else:
        return df_wes.iloc[0:0,:]


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


def stateaccuracy_allele_starch(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5, purity_threshold=0.3):
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
    coarse_states_inferred = np.array([fun_hatchetconvert(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values, counts=((df_starch_cnv.END.values-df_starch_cnv.START.values) / binsize).astype(int)) for cid in clone_ids])
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file, purity_threshold=purity_threshold)
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    coarse_states_wes = np.array([fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values, df_wes[f"Bcopy_{sid}"].values, counts=((df_wes.END.values-df_wes.START.values) / binsize).astype(int)) for sid in retained_hatchet_clones])

    percent_category = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    for s,sid in enumerate(retained_hatchet_clones):
        # accuracy
        for c,cid in enumerate(clone_ids):
            # coarse
            percent_category[c, s] = 1.0 * np.sum(coarse_states_inferred[c] == coarse_states_wes[s][snp_seg_index]) / len(snp_seg_index)
    return percent_category, sorted_chr_pos


def stateaccuracy_calicost_fromdf(configuration_file, r_hmrf_initialization, df_compare, midfix="", ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5, purity_threshold=0.3):
    """
    df_compare : DataFrame
        Contains the following columns: CHR, START, END, Acopy_clone1, Bcopy_clone1, Acopy_clone2, Bcopy_clone2, ...
    """
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
    coarse_states_inferred = np.array([fun_hatchetconvert(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values, counts=((df_starch_cnv.END.values-df_starch_cnv.START.values) / binsize).astype(int)) for cid in clone_ids])
    
    # format change on df_compare
    df_compare["CHR"] = df_compare["CHR"].astype(str)
    if ~np.any( df_compare.CHR.isin(ordered_chr) ):
        df_compare["CHR"] = df_compare.CHR.map(lambda x: x.replace("chr", ""))
    df_compare = df_compare[df_compare.CHR.isin(ordered_chr)]
    df_compare["int_chrom"] = df_compare.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_compare = df_compare.sort_values(by=["int_chrom", "START"])
    
    snp_seg_index = map_hatchet_to_bins(df_compare, sorted_chr_pos)
    ref_clones = [x[6:] for x in df_compare.columns if x.startswith("Acopy_")]
    coarse_states_compare = np.array([fun_hatchetconvert(df_compare[f"Acopy_{sid}"].values, df_compare[f"Bcopy_{sid}"].values, counts=((df_compare.END.values-df_compare.START.values) / binsize).astype(int)) for sid in ref_clones])

    percent_category = np.zeros( (len(clone_ids), len(ref_clones)) )
    for s,sid in enumerate(ref_clones):
        # accuracy
        for c,cid in enumerate(clone_ids):
            # coarse
            percent_category[c, s] = 1.0 * np.sum(coarse_states_inferred[c] == coarse_states_compare[s][snp_seg_index]) / len(snp_seg_index)
    return percent_category, sorted_chr_pos


def exactaccuracy_allele_starch(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)], purity_threshold=0.3):
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
    df_wes = read_hatchet(hatchet_wes_file, purity_threshold=purity_threshold)
    df_mapped_wes = pd.DataFrame({"CHR":df_starch_cnv.CHR, "START":df_starch_cnv.START, "END":df_starch_cnv.END})
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    percent_exact = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    for s,sid in enumerate(retained_hatchet_clones):
        minor_copy_wes = np.minimum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        major_copy_wes = np.maximum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        df_mapped_wes[[f'clone{s} A', f'clone{s} B']] = np.vstack([minor_copy_wes, major_copy_wes]).T
        for c, cid in enumerate(clone_ids):
            # exact
            minor_copy_inferred = np.minimum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            major_copy_inferred = np.maximum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            percent_exact[c, s] = 1.0 * np.sum((minor_copy_inferred==minor_copy_wes) & (major_copy_inferred==major_copy_wes)) / len(minor_copy_inferred)
    return percent_exact, sorted_chr_pos, df_mapped_wes


def stateaccuracy_numbat(numbat_dirs, hatchet_wes_file, sorted_chr_pos, ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5):
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    coarse_states_wes = np.array([fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values, df_wes[f"Bcopy_{sid}"].values, counts=((df_wes.END.values-df_wes.START.values) / binsize).astype(int)) for sid in retained_hatchet_clones])

    percent_category = []
    states_numbat = []
    for dirname in numbat_dirs:
        tmpdf_numbat = pd.read_csv(f"{dirname}/bulk_clones_final.tsv.gz", header=0, sep="\t")
        n_numbat_samples = len( np.unique(tmpdf_numbat["sample"]) )
        # check chromosome name
        tmpdf_numbat.CHROM = tmpdf_numbat.CHROM.astype(str)
        if ~np.any( tmpdf_numbat.CHROM.isin(ordered_chr) ):
            tmpdf_numbat["CHROM"] = tmpdf_numbat.CHROM.map(lambda x: x.replace("chr", ""))
        tmpdf_numbat = tmpdf_numbat[tmpdf_numbat.CHROM.isin(ordered_chr)]
        tmpdf_numbat["int_chrom"] = tmpdf_numbat.CHROM.map(ordered_chr_map)
        tmpdf_numbat.sort_values(by=['int_chrom', 'POS'], inplace=True)

        this_percent_category = np.zeros((n_numbat_samples, len(retained_hatchet_clones)))
        for sidx,s in enumerate(np.unique(tmpdf_numbat["sample"])):
            tmpdf_sample = tmpdf_numbat[tmpdf_numbat["sample"] == s][["int_chrom", "POS", "cnv_state"]]
            index = np.ones(len(sorted_chr_pos), dtype=int) * -1
            j = 0
            for i in range(len(sorted_chr_pos)):
                this_chr = sorted_chr_pos[i][0]
                this_pos = sorted_chr_pos[i][1]
                while (j < tmpdf_sample.shape[0]) and ((tmpdf_sample.int_chrom.values[j] < this_chr) or (tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.POS.values[j] < this_pos)):
                    j += 1
                if j < tmpdf_sample.shape[0] and tmpdf_sample.int_chrom.values[j] == this_chr:
                    index[i] = j
                else:
                    index[i] = j -1
            for c in range(len(retained_hatchet_clones)):
                this_percent_category[sidx, c] = 1.0 * np.sum(tmpdf_sample["cnv_state"].values[index] == coarse_states_wes[c][snp_seg_index]) / len(snp_seg_index)

            states_numbat.append( tmpdf_sample["cnv_state"].values[index] )
        percent_category.append(this_percent_category)

    percent_category = np.vstack(percent_category)
    states_numbat = np.array(states_numbat)
    return percent_category, coarse_states_wes[:,snp_seg_index], states_numbat


def stateaccuracy_numbat_fromdf(numbat_dirs, df_compare, sorted_chr_pos, ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5):
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    
    # format change on df_compare
    df_compare["CHR"] = df_compare["CHR"].astype(str)
    if ~np.any( df_compare.CHR.isin(ordered_chr) ):
        df_compare["CHR"] = df_compare.CHR.map(lambda x: x.replace("chr", ""))
    df_compare = df_compare[df_compare.CHR.isin(ordered_chr)]
    df_compare["int_chrom"] = df_compare.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_compare = df_compare.sort_values(by=["int_chrom", "START"])
    
    snp_seg_index = map_hatchet_to_bins(df_compare, sorted_chr_pos)
    ref_clones = [x[6:] for x in df_compare.columns if x.startswith("Acopy_")]
    coarse_states_compare = np.array([fun_hatchetconvert(df_compare[f"Acopy_{sid}"].values, df_compare[f"Bcopy_{sid}"].values, counts=((df_compare.END.values-df_compare.START.values) / binsize).astype(int)) for sid in ref_clones])

    percent_category = []
    states_numbat = []
    for dirname in numbat_dirs:
        tmpdf_numbat = pd.read_csv(f"{dirname}/bulk_clones_final.tsv.gz", header=0, sep="\t")
        n_numbat_samples = len( np.unique(tmpdf_numbat["sample"]) )
        # check chromosome name
        tmpdf_numbat.CHROM = tmpdf_numbat.CHROM.astype(str)
        if ~np.any( tmpdf_numbat.CHROM.isin(ordered_chr) ):
            tmpdf_numbat["CHROM"] = tmpdf_numbat.CHROM.map(lambda x: x.replace("chr", ""))
        tmpdf_numbat = tmpdf_numbat[tmpdf_numbat.CHROM.isin(ordered_chr)]
        tmpdf_numbat["int_chrom"] = tmpdf_numbat.CHROM.map(ordered_chr_map)
        tmpdf_numbat.sort_values(by=['int_chrom', 'POS'], inplace=True)

        this_percent_category = np.zeros((n_numbat_samples, len(ref_clones)))
        for sidx,s in enumerate(np.unique(tmpdf_numbat["sample"])):
            tmpdf_sample = tmpdf_numbat[tmpdf_numbat["sample"] == s][["int_chrom", "POS", "cnv_state"]]
            index = np.ones(len(sorted_chr_pos), dtype=int) * -1
            j = 0
            for i in range(len(sorted_chr_pos)):
                this_chr = sorted_chr_pos[i][0]
                this_pos = sorted_chr_pos[i][1]
                while (j < tmpdf_sample.shape[0]) and ((tmpdf_sample.int_chrom.values[j] < this_chr) or (tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.POS.values[j] < this_pos)):
                    j += 1
                if j < tmpdf_sample.shape[0] and tmpdf_sample.int_chrom.values[j] == this_chr:
                    index[i] = j
                else:
                    index[i] = j -1
            for c in range(len(ref_clones)):
                this_percent_category[sidx, c] = 1.0 * np.sum(tmpdf_sample["cnv_state"].values[index] == coarse_states_compare[c][snp_seg_index]) / len(snp_seg_index)

            states_numbat.append( tmpdf_sample["cnv_state"].values[index] )
        percent_category.append(this_percent_category)

    percent_category = np.vstack(percent_category)
    states_numbat = np.array(states_numbat)
    return percent_category, coarse_states_compare[:,snp_seg_index], states_numbat


def cna_precisionrecall_numbat(numbat_dirs, hatchet_wes_file, sorted_chr_pos, ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5):
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    coarse_states_wes = np.array([fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values, df_wes[f"Bcopy_{sid}"].values, counts=((df_wes.END.values-df_wes.START.values) / binsize).astype(int)) for sid in retained_hatchet_clones])

    precision = []
    recall = []
    states_numbat = []
    for dirname in numbat_dirs:
        tmpdf_numbat = pd.read_csv(f"{dirname}/bulk_clones_final.tsv.gz", header=0, sep="\t")
        n_numbat_samples = len( np.unique(tmpdf_numbat["sample"]) )
        # check chromosome name
        tmpdf_numbat.CHROM = tmpdf_numbat.CHROM.astype(str)
        if ~np.any( tmpdf_numbat.CHROM.isin(ordered_chr) ):
            tmpdf_numbat["CHROM"] = tmpdf_numbat.CHROM.map(lambda x: x.replace("chr", ""))
        tmpdf_numbat = tmpdf_numbat[tmpdf_numbat.CHROM.isin(ordered_chr)]
        tmpdf_numbat["int_chrom"] = tmpdf_numbat.CHROM.map(ordered_chr_map)
        tmpdf_numbat.sort_values(by=['int_chrom', 'POS'], inplace=True)

        this_precision = np.zeros((n_numbat_samples, len(retained_hatchet_clones)))
        this_recall = np.zeros((n_numbat_samples, len(retained_hatchet_clones)))
        for sidx,s in enumerate(np.unique(tmpdf_numbat["sample"])):
            tmpdf_sample = tmpdf_numbat[tmpdf_numbat["sample"] == s][["int_chrom", "POS", "cnv_state"]]
            index = np.ones(len(sorted_chr_pos), dtype=int) * -1
            j = 0
            for i in range(len(sorted_chr_pos)):
                this_chr = sorted_chr_pos[i][0]
                this_pos = sorted_chr_pos[i][1]
                while (j < tmpdf_sample.shape[0]) and ((tmpdf_sample.int_chrom.values[j] < this_chr) or (tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.POS.values[j] < this_pos)):
                    j += 1
                if j < tmpdf_sample.shape[0] and tmpdf_sample.int_chrom.values[j] == this_chr:
                    index[i] = j
                else:
                    index[i] = j -1
            for c in range(len(retained_hatchet_clones)):
                index_truth = set(list( np.where(coarse_states_wes[c][snp_seg_index] != 'neu')[0] ))
                index_pred = set(list( np.where(tmpdf_sample["cnv_state"].values[index] != 'neu')[0] ))
                this_precision[sidx, c] = 0 if len(index_pred)==0 else len(index_truth & index_pred) / len(index_pred)
                this_recall[sidx, c] = 0 if len(index_truth)==0 else len(index_truth & index_pred) / len(index_truth)

            states_numbat.append( tmpdf_sample["cnv_state"].values[index] )
        precision.append(this_precision)
        recall.append(this_recall)

    precision = np.vstack(precision)
    recall = np.vstack(recall)
    f1 = 2 * precision * recall / (precision + recall)
    states_numbat = np.array(states_numbat)
    return precision, recall, f1, coarse_states_wes[:,snp_seg_index], states_numbat


def cna_precisionrecall_calicost(configuration_file, r_hmrf_initialization, hatchet_wes_file, midfix="", ordered_chr=[str(c) for c in range(1,23)], purity_threshold=0.3):
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
    df_wes = read_hatchet(hatchet_wes_file, purity_threshold=purity_threshold)
    df_mapped_wes = pd.DataFrame({"CHR":df_starch_cnv.CHR, "START":df_starch_cnv.START, "END":df_starch_cnv.END})
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    precision = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    recall = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    f1 = np.zeros( (len(clone_ids), len(retained_hatchet_clones)) )
    for s,sid in enumerate(retained_hatchet_clones):
        minor_copy_wes = np.minimum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        major_copy_wes = np.maximum(df_wes[f"Acopy_{sid}"].values[snp_seg_index], df_wes[f"Bcopy_{sid}"].values[snp_seg_index])
        df_mapped_wes[[f'clone{s} A', f'clone{s} B']] = np.vstack([minor_copy_wes, major_copy_wes]).T
        for c, cid in enumerate(clone_ids):
            # exact
            minor_copy_inferred = np.minimum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            major_copy_inferred = np.maximum(df_starch_cnv[f"clone{cid} A"].values, df_starch_cnv[f"clone{cid} B"].values)
            index_truth = set(list( np.where( ~((minor_copy_wes==1) & (major_copy_wes==1)) )[0] ))
            index_pred = set(list( np.where( ~((minor_copy_inferred==1) & (major_copy_inferred==1)) )[0] ))

            precision[c, s] = 0 if len(index_pred)==0 else len(index_truth & index_pred) / len(index_pred)
            recall[c, s] = 0 if len(index_truth)==0 else  len(index_truth & index_pred) / len(index_truth)
            f1[c,s] = 0 if precision[c, s] + recall[c, s]==0 else 2 * (precision[c, s] * recall[c, s]) / (precision[c, s] + recall[c, s])

    return precision, recall, f1, sorted_chr_pos, df_mapped_wes



def stateaccuracy_infercnv(infercnv_dirs, hatchet_wes_file, sorted_chr_pos, ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5):
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    coarse_states_wes = np.array([fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values, df_wes[f"Bcopy_{sid}"].values, counts=((df_wes.END.values-df_wes.START.values) / binsize).astype(int)) for sid in retained_hatchet_clones])

    map_states = {1:"del", 2:"del", 3:"neu", 4:"amp", 5:"amp", 6:"amp"}
    percent_category = []
    for dirname in infercnv_dirs:
        tmpdf_infercnv = pd.read_csv(f"{dirname}/HMM_CNV_predictions.HMMi6.leiden.hmm_mode-subclusters.Pnorm_0.5.pred_cnv_regions.dat", header=0, index_col=None, sep="\t")
        infercnv_samples = np.unique(tmpdf_infercnv["cell_group_name"])
        # check chromosome name
        tmpdf_infercnv.chr = tmpdf_infercnv.chr.astype(str)
        if ~np.any( tmpdf_infercnv.chr.isin(ordered_chr) ):
            tmpdf_infercnv["CHROM"] = tmpdf_infercnv.chr.map(lambda x: x.replace("chr", ""))
        tmpdf_infercnv = tmpdf_infercnv[tmpdf_infercnv.chr.isin(ordered_chr)]
        tmpdf_infercnv["int_chrom"] = tmpdf_infercnv.chr.map(ordered_chr_map)

        this_percent_category = np.zeros((len(infercnv_samples), len(retained_hatchet_clones)))
        for sidx,s in enumerate(infercnv_samples):
            tmpdf_sample = tmpdf_infercnv[tmpdf_infercnv["cell_group_name"] == s][["int_chrom", "start", "end", "state"]]
            tmpdf_sample = tmpdf_sample.sort_values(by=["int_chrom", "start"])
            # inferCNV states for sorted_chr_pos
            cnvstate_sample = 3 * np.ones(len(sorted_chr_pos), dtype=int)
            j = 0
            for i in range(len(sorted_chr_pos)):
                this_chr = sorted_chr_pos[i][0]
                this_pos = sorted_chr_pos[i][1]
                while (j < tmpdf_sample.shape[0]) and ((tmpdf_sample.int_chrom.values[j] < this_chr) or (tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.end.values[j] < this_pos)):
                    j += 1
                if j < tmpdf_sample.shape[0] and tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.start.values[j] <= this_pos:
                    cnvstate_sample[i] = tmpdf_sample.state.values[j]
            for c in range(len(retained_hatchet_clones)):
                this_percent_category[sidx, c] = 1.0 * np.sum( np.array([map_states[x] for x in cnvstate_sample]) == coarse_states_wes[c][snp_seg_index]) / len(snp_seg_index)
            
        percent_category.append(this_percent_category)
    percent_category = np.vstack(percent_category)
    return percent_category


def stateaccuracy_oldstarch(oldstarch_dirs, map_hgtable, hatchet_wes_file, sorted_chr_pos, ordered_chr=[str(c) for c in range(1,23)], fun_hatchetconvert=convert_copy_to_states, binsize=1e5):
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    
    # hatchet results
    df_wes = read_hatchet(hatchet_wes_file)
    if df_wes.shape[0] == 0:
        return None, None
    snp_seg_index = map_hatchet_to_bins(df_wes, sorted_chr_pos)
    retained_hatchet_clones = [x[6:] for x in df_wes.columns if x.startswith("Acopy_")]
    coarse_states_wes = np.array([fun_hatchetconvert(df_wes[f"Acopy_{sid}"].values, df_wes[f"Bcopy_{sid}"].values, counts=((df_wes.END.values-df_wes.START.values) / binsize).astype(int)) for sid in retained_hatchet_clones])

    map_states = {0:"del", 1:"neu", 2:"amp"}
    percent_category = []
    for dirname in oldstarch_dirs:
        tmpdf_oldstarch = pd.read_csv(f"{dirname}/states_STITCH_output.csv", header=0, index_col=0, sep=",")
        tmpdf_oldstarch["int_chrom"] = [map_hgtable[x][0] for x in tmpdf_oldstarch.index]
        tmpdf_oldstarch["POS"] = [map_hgtable[x][1] for x in tmpdf_oldstarch.index]

        this_percent_category = np.zeros((tmpdf_oldstarch.shape[1]-2, len(retained_hatchet_clones)))
        for sidx in range(tmpdf_oldstarch.shape[1]-2):
            tmpdf_sample = pd.DataFrame({'int_chrom':tmpdf_oldstarch.int_chrom.values, 'POS':tmpdf_oldstarch.POS.values, 'cnv_state':[map_states[x] for x in tmpdf_oldstarch.iloc[:,sidx]]})
            index = np.ones(len(sorted_chr_pos), dtype=int) * -1
            j = 0
            for i in range(len(sorted_chr_pos)):
                this_chr = sorted_chr_pos[i][0]
                this_pos = sorted_chr_pos[i][1]
                while (j < tmpdf_sample.shape[0]) and ((tmpdf_sample.int_chrom.values[j] < this_chr) or (tmpdf_sample.int_chrom.values[j] == this_chr and tmpdf_sample.POS.values[j] < this_pos)):
                    j += 1
                if j < tmpdf_sample.shape[0] and tmpdf_sample.int_chrom.values[j] == this_chr:
                    index[i] = j
                else:
                    index[i] = j -1
            for c in range(len(retained_hatchet_clones)):
                this_percent_category[sidx, c] = 1.0 * np.sum(tmpdf_sample["cnv_state"].values[index] == coarse_states_wes[c][snp_seg_index]) / len(snp_seg_index)
        
        percent_category.append(this_percent_category)
    percent_category = np.vstack(percent_category)
    return percent_category
        


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
        tmp = strict_convert_copy_to_states(df_calicost[f"clone{c} A"].values, df_calicost[f"clone{c} B"].values)
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


def plot_hatchet_acn(hatchet_dir, hatchet_cnfile, out_file, binsize=1e6, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = read_hatchet(f"{hatchet_dir}/results/{hatchet_cnfile}", purity_threshold=0.1)
    if df_hatchet.shape[0] == 0:
        return
    # get CN state from integer copy numbers
    df_hatchet = add_cn_state(df_hatchet)
    # hatchet clones
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]

    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_hatchet.CHR.isin(ordered_chr) ):
        df_hatchet["CHR"] = df_hatchet.CHR.map(lambda x: x.replace("chr", ""))
    df_hatchet = df_hatchet[df_hatchet.CHR.isin(ordered_chr)]
    df_hatchet["int_chrom"] = df_hatchet.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_hatchet = df_hatchet.sort_values(by=["int_chrom", "START"])

    # expand each row in df_hatchet to multiple rows such that the new row has END - START = binsize
    df_expand = []
    for i in range(df_hatchet.shape[0]):
        # repeat the row i for int(END - START / binsize) times and save to a new dataframe
        n_bins = max(1, int(1.0*(df_hatchet.iloc[i].END - df_hatchet.iloc[i].START) / binsize))
        tmp = pd.DataFrame(np.repeat(df_hatchet.iloc[i:(i+1),:].values, n_bins, axis=0), columns=df_hatchet.columns)
        for k in range(n_bins):
            tmp.END.iloc[k] = df_hatchet.START.iloc[i]+ k*binsize
        tmp.END.iloc[-1] = df_hatchet.END.iloc[i]
        df_expand.append(tmp)
    df_hatchet = pd.concat(df_expand, ignore_index=True)

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


def plot_hatchet_totalcn(hatchet_dir, hatchet_cnfile, out_file, binsize=1e6, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = read_hatchet(f"{hatchet_dir}/results/{hatchet_cnfile}", purity_threshold=0.1)
    if df_hatchet.shape[0] == 0:
        return
    # get CN state from integer copy numbers
    df_hatchet = add_cn_state(df_hatchet)
    # hatchet clones
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]

    # check agreement with ordered_chr
    ordered_chr_map = {ordered_chr[i]:i for i in range(len(ordered_chr))}
    if ~np.any( df_hatchet.CHR.isin(ordered_chr) ):
        df_hatchet["CHR"] = df_hatchet.CHR.map(lambda x: x.replace("chr", ""))
    df_hatchet = df_hatchet[df_hatchet.CHR.isin(ordered_chr)]
    df_hatchet["int_chrom"] = df_hatchet.CHR.map(ordered_chr_map)
    # sort by int_chrom and START
    df_hatchet = df_hatchet.sort_values(by=["int_chrom", "START"])

    # expand each row in df_hatchet to multiple rows such that the new row has END - START = binsize
    df_expand = []
    for i in range(df_hatchet.shape[0]):
        # repeat the row i for int(END - START / binsize) times and save to a new dataframe
        n_bins = max(1, int(1.0*(df_hatchet.iloc[i].END - df_hatchet.iloc[i].START) / binsize))
        tmp = pd.DataFrame(np.repeat(df_hatchet.iloc[i:(i+1),:].values, n_bins, axis=0), columns=df_hatchet.columns)
        for k in range(n_bins):
            tmp.END.iloc[k] = df_hatchet.START.iloc[i]+ k*binsize
        tmp.END.iloc[-1] = df_hatchet.END.iloc[i]
        df_expand.append(tmp)
    df_hatchet = pd.concat(df_expand, ignore_index=True)

    # create another dataframe df_cnv, and apply to plot_acn_from_df function for plotting
    df_cnv = df_hatchet[["CHR", "START", "END"]]
    # for each clone in df_hatchet, split the A allele copy and B allele copy to separate columns
    clone_names = []
    for n in hatchet_clones:
        df_cnv[f"clone {n}"] = df_hatchet[f"cnstate_clone{n}"]

    # plot
    fig, axes = plt.subplots(1, 1, figsize=(15,0.9*2+0.6), dpi=200, facecolor="white")
    plot_total_cn(df_cnv, axes, palette_mode=6, add_chrbar=True, chrbar_thickness=0.4/(0.6*2 + 0.4), add_legend=True, remove_xticks=True)
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
        counts = df_hatchet.END.values-df_hatchet.START.values
        counts = np.maximum(1, counts / 1e4).astype(int)
        tmp = np.concatenate([ np.ones(counts[i]) * (total_cn[i]) for i in range(len(counts)) if ~np.isnan(total_cn[i]) ])
        median_cn = np.median(tmp)
        # get cn_state vector such that total_cn > median_cn is "amp", total_cn < median_cn is "del", and total_cn == median_cn is "neu"
        cn_state = np.array(["neu"] * len(total_cn))
        cn_state[total_cn > median_cn] = "amp"
        cn_state[total_cn < median_cn] = "del"
        cn_state[ (total_cn == median_cn) & (is_homozygous) ] = "loh"
        # add cn_state to df_hatchet
        df_hatchet[f"cnstate_clone{n}"] = cn_state
    return df_hatchet


def add_log2cnratio(df_hatchet):
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]
    assert ("START" in df_hatchet.columns) and ("END" in df_hatchet.columns)
    for n in hatchet_clones:
        # compute total copy number per bin
        total_cn = np.array([ int(x.split("|")[0]) + int(x.split("|")[1]) for x in df_hatchet[f"cn_clone{n}"] ])
        # check whether each bin is homozygous, i.e., either first cn is 0 or second cn is 0
        is_homozygous = np.array([ (int(x.split("|")[0])==0) or (int(x.split("|")[1])==0) for x in df_hatchet[f"cn_clone{n}"] ])
        # compute median total copy number weighted by END - START
        counts = df_hatchet.END.values-df_hatchet.START.values
        counts = np.maximum(1, counts / 1e4).astype(int)
        tmp = np.concatenate([ np.ones(counts[i]) * (total_cn[i]) for i in range(len(counts)) if ~np.isnan(total_cn[i]) ])
        median_cn = np.median(tmp)
        # compute log2 CNV ratio
        df_hatchet[f"log2_cnratio_clone{n}"] = np.log2(total_cn / median_cn)
    return df_hatchet


def map_hatchet_to_states(hatchet_resultfile, out_file, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = read_hatchet(hatchet_resultfile, purity_threshold=0.1)
    if df_hatchet.shape[0] == 0:
        return
    # get CN state from integer copy numbers
    df_hatchet = add_cn_state(df_hatchet)
    # add log2 CNV ratio
    df_hatchet = add_log2cnratio(df_hatchet)
    # hatchet clones
    hatchet_clones = [x[8:] for x in df_hatchet.columns if x.startswith("cn_clone")]

    # re-order df_hatchet_columns
    ordered_columns = ["CHR", "START", "END"]
    for n in hatchet_clones:
        ordered_columns += [f"cn_clone{n}", f"cnstate_clone{n}", f"log2_cnratio_clone{n}"]

    df_hatchet[ordered_columns].to_csv(out_file, sep="\t", header=True, index=False)


def map_hatchet_to_genes(hatchet_resultfile, hgtable_file, out_file, ordered_chr=[str(c) for c in range(1,23)]):
    # read in hatchet integer copy number file
    df_hatchet = read_hatchet(hatchet_resultfile, purity_threshold=0.1)
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


# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         print("python plot_hatchet.py <hatchet_dir> <hatchet_cnfile> <out_file>")
#         sys.exit(1)
    
#     hatchet_dir = sys.argv[1]
#     hatchet_cnfile = sys.argv[2]
#     out_plot_acn = sys.argv[3]
#     out_plot_tcn = sys.argv[4]
#     out_file_seg = sys.argv[5]
#     out_file_gene = sys.argv[6]
#     plot_hatchet_acn(hatchet_dir, hatchet_cnfile, out_plot_acn)
#     # plot total copy number
#     plot_hatchet_totalcn(hatchet_dir, hatchet_cnfile, out_plot_tcn)
#     # output log2 CNV ratio and copy number state of hatchet
#     map_hatchet_to_states(f"{hatchet_dir}/results/{hatchet_cnfile}", out_file_seg)
#     # output gene-level file
#     hgtable_file = "/home/congma/congma/codes/locality-clustering-cnv/data/hgTables_hg38_gencode.txt"
#     map_hatchet_to_genes(f"{hatchet_dir}/results/{hatchet_cnfile}", hgtable_file, out_file_gene)