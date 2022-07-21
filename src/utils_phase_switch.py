import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange


def get_position_cM_table(chr_pos_vector):
    """
    Attributes
    ----------
    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs
    """
    df = pd.read_csv("/u/congma/ragr-data/datasets/genetic_map/recomb-hg38/genetic_map_GRCh38_merged.tab", header=0, sep="\t")
    # remove chrX
    df = df[df.chrom != "chrX"]
    # check the chromosome names
    if not ("chr" in str(chr_pos_vector[0][0])):
        df["chrom"] = [int(x[3:]) for x in df.chrom]
    df = df.sort_values(by=["chrom", "pos"])
    ref_chrom = np.array(df.chrom)
    ref_pos = np.array(df.pos)
    ref_cm = np.array(df.pos_cm)
    # also sort the input argument
    chr_pos_vector.sort()
    # find the centimorgan values (interpolate between (k-1)-th and k-th rows in centimorgan tables)
    position_cM = np.ones(len(chr_pos_vector)) * np.nan
    k = 0
    for i,x in enumerate(chr_pos_vector):
        chrname = x[0]
        pos = x[1]
        while k < len(ref_chrom) and (ref_chrom[k] < chrname or (ref_chrom[k] == chrname and ref_pos[k] < pos)):
            k += 1
        if ref_chrom[k] == chrname and ref_pos[k] >= pos:
            if k > 0 and ref_chrom[k-1] == chrname:
                position_cM[i] = ref_cm[k-1] + (pos - ref_pos[k-1]) / (ref_pos[k] - ref_pos[k-1]) * (ref_cm[k] - ref_cm[k-1])
            else:
                position_cM[i] = (pos - 0) / (ref_pos[k] - 0) * (ref_cm[k] - 0)
    return position_cM


def compute_phase_switch_probability_position(position_cM, chr_pos_vector, min_prob=1e-20):
    """
    Attributes
    ----------
    position_cM : array, (number SNP positions)
        Centimorgans of SNPs located at each entry of position_cM.

    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs. It is used to identify start of a new chr.
    """
    phase_switch_prob = np.ones(len(position_cM)) * 1e-20
    for i,cm in enumerate(position_cM[:-1]):
        cm_next = position_cM[i+1]
        if np.isnan(cm) or np.isnan(cm_next) or chr_pos_vector[i][0] != chr_pos_vector[i+1][0]:
            continue
        assert cm <= cm_next
        d = cm_next - cm
        phase_switch_prob[i] = (1 - np.exp(-2 * d)) / 2
    phase_switch_prob[phase_switch_prob < min_prob] = min_prob
    return phase_switch_prob


def duplicate_RD(chr_baf, pos_baf, chr_rd, start_rd, end_rd, tumor_rd, normal_rd):
    tumor_reads = np.ones(len(chr_baf)) * np.nan
    normal_reads = np.ones(len(chr_baf)) * np.nan
    idx = 0
    for i in range(len(chr_baf)):
        while idx < len(chr_rd) and (chr_rd[idx] < chr_baf[i] or (chr_rd[idx] == chr_baf[i] and end_rd[idx] < pos_baf[i])):
            idx += 1
        if idx < len(chr_rd) and chr_rd[idx] == chr_baf[i] and end_rd[idx] >= pos_baf[i] and start_rd[idx] <= pos_baf[i]:
            tumor_reads[i] = tumor_rd[idx]
            normal_reads[i] = normal_rd[idx]
    return tumor_reads, normal_reads


def generate_input_from_HATCHet(hatchetdir, output_picklefile, rdrfile="abin/bulk.bb", baffile="baf/bulk.1bed", phasefile="phase/phased.vcf.gz", with_chr_prefix=True):
    if with_chr_prefix:
        unique_chrs = [f"chr{i}" for i in range(1, 23)]
    else:
        unique_chrs = np.arange(1, 23)
    
    ### load hatchet outputs ###
    if Path(output_picklefile).exists():
        # RDR file
        df_all = pd.read_csv(f"{hatchetdir}/{rdrfile}", header=0, sep="\t")
        df_all.iloc[:,0] = pd.Categorical(df_all.iloc[:,0], categories=unique_chrs, ordered=True)
        df_all.sort_values(by=["#CHR", "START"], inplace=True)
        # samples
        unique_samples = np.unique(df_all["SAMPLE"])
        # allele counts
        df_baf = pd.read_pickle(output_picklefile)
    else:
        # RDR file
        df_all = pd.read_csv(f"{hatchetdir}/{rdrfile}", header=0, sep="\t")
        df_all.iloc[:,0] = pd.Categorical(df_all.iloc[:,0], categories=unique_chrs, ordered=True)
        df_all.sort_values(by=["#CHR", "START"], inplace=True)
        # samples
        unique_samples = np.unique(df_all["SAMPLE"])
        # allele counts for individual SNPs
        def load_shared_BAF(hatchetdir, baffile, unique_chrs, unique_samples):
            tmpdf = pd.read_csv(f"{hatchetdir}/{baffile}", header=None, sep="\t", names=["CHR", "POS", "SAMPLE", "REF", "ALT"])
            df_baf = []
            for chrname in unique_chrs:
                tmp = tmpdf[tmpdf.CHR == chrname]
                list_pos = [set(list(tmp[tmp["SAMPLE"] == s].POS)) for s in unique_samples] # SNP set of each individual sample
                shared_pos = set.intersection(*list_pos) # SNPs that are shared across samples
                index = np.array([i for i in range(tmp.shape[0]) if tmp.iloc[i,1] in shared_pos])
                tmp = tmp.iloc[index,:]
                tmp.sort_values(by=["POS", "SAMPLE"], inplace=True)
                df_baf.append( tmp )
            df_baf = pd.concat(df_baf, ignore_index=True)
            return df_baf
        df_baf = load_shared_BAF(hatchetdir, baffile, unique_chrs, unique_samples)
        # reference-based phasing results
        df_phase = pd.read_csv(f"{hatchetdir}/{phasefile}", comment="#", sep="\t", \
                        names=["CHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLENAME"])
        df_phase = df_phase[(df_phase.SAMPLENAME=="0|1") | (df_phase.SAMPLENAME=="1|0")]
        print("HATCHet dataframes loaded.")

        ### gather phased BAF info ###
        df_combined_baf = []
        for chrname in unique_chrs:
            tmpdf_baf = df_baf[df_baf.CHR == chrname]
            tmpdf_phase = df_phase[df_phase.CHR == chrname][["POS", "SAMPLENAME"]]
            tmpdf_baf = tmpdf_baf.join( tmpdf_phase.set_index("POS"), on="POS")
            tmpdf_baf = tmpdf_baf[~tmpdf_baf.SAMPLENAME.isnull()]
            tmpdf_baf["B_count"] = np.where(tmpdf_baf.SAMPLENAME=="0|1", tmpdf_baf.REF, tmpdf_baf.ALT)
            tmpdf_baf["DP"] = tmpdf_baf.REF + tmpdf_baf.ALT
            df_combined_baf.append( tmpdf_baf )
        df_combined_baf = pd.concat(df_combined_baf, ignore_index=True)
        df_combined_baf.iloc[:,0] = pd.Categorical(df_combined_baf.CHR, categories=unique_chrs, ordered=True)
        df_combined_baf.sort_values(by=["CHR", "POS"], inplace=True)
        df_baf = df_combined_baf

        ### duplicate RDR info for each SNP ###
        df_baf["TOTAL_READS"] = np.nan
        df_baf["NORMAL_READS"] = np.nan
        for s in unique_samples:
            index = np.where(df_baf["SAMPLE"] == s)[0]
            index_rd = np.where(df_all["SAMPLE"] == s)[0]
            tumor_reads, normal_reads = duplicate_RD(np.array(df_baf.iloc[index,:].CHR.cat.codes), np.array(df_baf.iloc[index,:].POS), \
                                                    np.array(df_all.iloc[index_rd,0].cat.codes), np.array(df_all.iloc[index_rd,:].START), np.array(df_all.iloc[index_rd,:].END), \
                                                    np.array(df_all.iloc[index_rd,:].TOTAL_READS), np.array(df_all.iloc[index_rd,:].NORMAL_READS))
            df_baf.iloc[index, -2] = tumor_reads
            df_baf.iloc[index, -1] = normal_reads
        # remove SNP positions with TOTAL_READS=NAN (if NAN occurs in one sample, remove the corresponding SNPs for the other samples too)
        def remove_nan_RD(df_baf):
            idx_nan = np.where(np.logical_or( df_baf.TOTAL_READS.isnull(), df_baf.NORMAL_READS.isnull() ))[0]
            chr = np.array(df_baf.CHR)
            pos = np.array(df_baf.POS)
            chr_pos = np.array([f"{chr[i]}_{pos[i]}" for i in range(len(chr))])
            nan_chr_pos = set(list(chr_pos[idx_nan]))
            idx_remain = np.array([i for i,snpid in enumerate(chr_pos) if not (snpid in nan_chr_pos)])
            df_baf = df_baf.iloc[idx_remain, :]
            return df_baf
        df_baf = remove_nan_RD(df_baf)
        df_baf.to_pickle(output_picklefile)
        print("SNP-level BAF and bin-level RDR paired up.")

    ### from BAF, RDR table, generate HMM input ###
    lengths = np.array([ np.sum(np.logical_and(df_baf["CHR"]==chrname, df_baf["SAMPLE"]==unique_samples[0])) for chrname in unique_chrs ])

    X = np.zeros(( np.sum(lengths), 2, len(unique_samples) ))
    base_nb_mean = np.zeros((np.sum(lengths), len(unique_samples) ))
    total_bb_RD = np.zeros((np.sum(lengths), len(unique_samples) ))

    for k,s in enumerate(unique_samples):
        df = df_baf[df_baf["SAMPLE"] == s]
        X[:,0,k] = df.TOTAL_READS
        X[:,1,k] = df.B_count

        total_bb_RD[:,k] = np.array(df.DP)
        df2 = df_all[df_all["SAMPLE"] == s]
        base_nb_mean[:,k] = np.array(df.NORMAL_READS / np.sum(df2.NORMAL_READS) * np.sum(df2.TOTAL_READS))

    # site-wise transition matrix
    chr_pos_vector = [(df_baf.CHR.iloc[i], df_baf.POS.iloc[i]) for i in np.where(df_baf["SAMPLE"]==unique_samples[0])[0]]
    position_cM = get_position_cM_table(chr_pos_vector)
    phase_switch_prob = compute_phase_switch_probability_position(position_cM, chr_pos_vector)
    log_sitewise_transmat = np.log(phase_switch_prob)

    return X, lengths, base_nb_mean, total_bb_RD, log_sitewise_transmat
