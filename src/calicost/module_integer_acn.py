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
from calicost.arg_parse import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.utils_distribution_fitting import *
from calicost.utils_hmrf import *
from calicost.hmrf import *
from calicost.utils_IO import *
from calicost.module_parse_input import *
from calicost.find_integer_copynumber import *


def run_infer_integer_acn(config, foldername, res_combine, single_X, single_base_nb_mean, single_total_bb_RD, barcodes, single_tumor_prop, df_gene_snp, df_bininfo):
    # outdir
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"

    # create directory
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()

    n_obs = single_X.shape[0]

    # combined HMRF-HMM results
    final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
    # add clone 0 as normal clone if it doesn't appear in final_clone_ids
    if not (0 in final_clone_ids):
        final_clone_ids = np.append(0, final_clone_ids)
    # chr position
    medfix = ["", "_diploid", "_triploid", "_tetraploid"]
    for o,max_medploidy in enumerate([None, 2, 3, 4]):
        # A/B copy number per bin
        allele_specific_copy = []
        # A/B copy number per state
        state_cnv = []

        df_genelevel_cnv = None
        if config["tumorprop_file"] is None:
            X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids])
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, [np.where(res_combine["new_assignment"]==cid)[0] for cid in final_clone_ids], single_tumor_prop, threshold=config["tumorprop_threshold"])

        for s, cid in enumerate(final_clone_ids):
            if np.sum(base_nb_mean[:,s]) == 0:
                continue
            # adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1
            lambd = base_nb_mean[:,s] / np.sum(base_nb_mean[:,s])
            this_pred_cnv = res_combine["pred_cnv"][:,s]
            adjusted_log_mu = np.log( np.exp(res_combine["new_log_mu"][:,s]) / np.sum(np.exp(res_combine["new_log_mu"][this_pred_cnv,s]) * lambd) )
            if not max_medploidy is None:
                best_integer_copies, _ = hill_climbing_integer_copynumber_oneclone(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, max_medploidy=max_medploidy)
            else:
                try:
                    best_integer_copies, _ = hill_climbing_integer_copynumber_fixdiploid(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, nonbalance_bafdist=config["nonbalance_bafdist"], nondiploid_rdrdist=config["nondiploid_rdrdist"])
                except:
                    try:
                        best_integer_copies, _ = hill_climbing_integer_copynumber_fixdiploid(adjusted_log_mu, base_nb_mean[:,s], res_combine["new_p_binom"][:,s], this_pred_cnv, nonbalance_bafdist=config["nonbalance_bafdist"], nondiploid_rdrdist=config["nondiploid_rdrdist"], min_prop_threshold=0.02)
                    except:
                        finding_distate_failed = True
                        continue

            print(f"max med ploidy = {max_medploidy}, clone {s}, integer copy inference loss = {_}")
            #
            allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 0].reshape(1,-1), index=[f"clone{cid} A"], columns=np.arange(n_obs) ) )
            allele_specific_copy.append( pd.DataFrame( best_integer_copies[res_combine["pred_cnv"][:,s], 1].reshape(1,-1), index=[f"clone{cid} B"], columns=np.arange(n_obs) ) )
            #
            state_cnv.append( pd.DataFrame( res_combine["new_log_mu"][:,s].reshape(-1,1), columns=[f"clone{cid} logmu"], index=np.arange(config['n_states']) ) )
            state_cnv.append( pd.DataFrame( res_combine["new_p_binom"][:,s].reshape(-1,1), columns=[f"clone{cid} p"], index=np.arange(config['n_states']) ) )
            state_cnv.append( pd.DataFrame( best_integer_copies[:,0].reshape(-1,1), columns=[f"clone{cid} A"], index=np.arange(config['n_states']) ) )
            state_cnv.append( pd.DataFrame( best_integer_copies[:,1].reshape(-1,1), columns=[f"clone{cid} B"], index=np.arange(config['n_states']) ) )
            #
            # tmpdf = get_genelevel_cnv_oneclone(best_integer_copies[res_combine["pred_cnv"][:,s], 0], best_integer_copies[res_combine["pred_cnv"][:,s], 1], x_gene_list)
            # tmpdf.columns = [f"clone{s} A", f"clone{s} B"]
            bin_Acopy_mappers = {i:x for i,x in enumerate(best_integer_copies[res_combine["pred_cnv"][:,s], 0])}
            bin_Bcopy_mappers = {i:x for i,x in enumerate(best_integer_copies[res_combine["pred_cnv"][:,s], 1])}
            tmpdf = pd.DataFrame({"gene":df_gene_snp[df_gene_snp.is_interval].gene, f"clone{s} A":df_gene_snp[df_gene_snp.is_interval]['bin_id'].map(bin_Acopy_mappers), \
                f"clone{s} B":df_gene_snp[df_gene_snp.is_interval]['bin_id'].map(bin_Bcopy_mappers)}).set_index('gene')
            if df_genelevel_cnv is None:
                df_genelevel_cnv = copy.copy( tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int) )
            else:
                df_genelevel_cnv = df_genelevel_cnv.join( tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int) )
        if len(state_cnv) == 0:
            continue
        # output gene-level copy number
        df_genelevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_genelevel.tsv", header=True, index=True, sep="\t")
        # output segment-level copy number
        allele_specific_copy = pd.concat(allele_specific_copy)
        df_seglevel_cnv = pd.DataFrame({"CHR":df_bininfo.CHR.values, "START":df_bininfo.START.values, "END":df_bininfo.END.values })
        df_seglevel_cnv = df_seglevel_cnv.join( allele_specific_copy.T )
        df_seglevel_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_seglevel.tsv", header=True, index=False, sep="\t")
        # output per-state copy number
        state_cnv = functools.reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True, how='inner'), state_cnv)
        state_cnv.to_csv(f"{outdir}/cnv{medfix[o]}_perstate.tsv", header=True, index=False, sep="\t")
    
    ##### output clone label #####
    df_clone_label = pd.DataFrame({"clone_label":res_combine["new_assignment"]}, index=barcodes)
    if not config["tumorprop_file"] is None:
        df_clone_label["tumor_proportion"] = single_tumor_prop
    df_clone_label.to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    parser.add_argument("--foldername", help="folder name to save inferred integer copy number results", required=True, type=str)
    args = parser.parse_args()

    try:
        config = read_configuration_file(args.configfile)
    except:
        config = read_joint_configuration_file(args.configfile)

    print("Configurations:")
    for k in sorted(list(config.keys())):
        print(f"\t{k} : {config[k]}")
    
    # load initial parsed input
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)

    # load ASE-filtered data matrix
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    filter_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_FILTERASE}"
    df_bininfo = pd.read_csv(f"{filter_outdir}/table_bininfo.csv.gz", header=0, index_col=None, sep="\t")
    table_rdrbaf = pd.read_csv(f"{filter_outdir}/table_rdrbaf.csv.gz", header=0, index_col=None, sep="\t")
    # reconstruct single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, and lengths
    n_bins = df_bininfo.shape[0]
    n_spots = len(table_rdrbaf.BARCODES.unique())
    single_X = np.zeros((n_bins, 2, n_spots))
    single_X[:, 0, :] = table_rdrbaf["EXP"].values.reshape((n_bins, n_spots), order="F")
    single_X[:, 1, :] = table_rdrbaf["B"].values.reshape((n_bins, n_spots), order="F")
    single_base_nb_mean = df_bininfo["NORMAL_COUNT"].values.reshape(-1,1) / np.sum(df_bininfo["NORMAL_COUNT"].values) @ np.sum(single_X[:,0,:], axis=0).reshape(1,-1)
    single_total_bb_RD = table_rdrbaf["TOT"].values.reshape((n_bins, n_spots), order="F")
    log_sitewise_transmat = df_bininfo["LOG_PHASE_TRANSITION"].values
    lengths = np.array([ np.sum(df_bininfo.CHR == c) for c in df_bininfo.CHR.unique() ])

    # load combined HMRF-HMM results
    rdrbaf_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_RDRBAFCLONES}"
    res_combine = dict(np.load(f"{rdrbaf_outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))

    # infer integer allele-specific copy numbers
    run_infer_integer_acn(config, args.foldername, res_combine, single_X, single_base_nb_mean, single_total_bb_RD, barcodes, single_tumor_prop, df_gene_snp, df_bininfo)