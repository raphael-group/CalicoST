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
from calicost.utils_hmrf import *
from calicost.hmrf import *
from calicost.utils_IO import *
from calicost.module_parse_input import *


def run_filter_ase(config, foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, df_gene_snp, barcodes, single_tumor_prop, sample_list, sample_ids, smooth_mat, exp_counts, merged_baf_profiles):
    # output directory
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"

    # create directory
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()

    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    # find confident normal spots
    if (config["tumorprop_file"] is None):
        # Assuming spots have pure clones
        # Identify the clone with smallest BAF deviation from 0.5
        # Then identify spots with smallest variation of expression across bins within that BAF clone
        EPS_BAF = 0.05
        PERCENT_NORMAL = 40
        vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
        id_nearnormal_clone = np.argmin(np.sum( np.maximum(np.abs(merged_baf_profiles - 0.5)-EPS_BAF, 0), axis=1))
        while True:
            stdthreshold = np.percentile(vec_stds[merged_res["new_assignment"] == id_nearnormal_clone], PERCENT_NORMAL)
            normal_candidate = (vec_stds < stdthreshold) & (merged_res["new_assignment"] == id_nearnormal_clone)
            if np.sum(copy_single_X_rdr[:, (normal_candidate==True)]) > single_X.shape[0] * 200 or PERCENT_NORMAL == 100:
                break
            PERCENT_NORMAL += 10
        pd.Series(barcodes[normal_candidate==True].index).to_csv(f"{outdir}/normal_candidate_barcodes.txt", header=False, index=False)
    else:
        # Assuming spots have a mixture of normal and tumor spots
        # Tumor proportion per spot is given
        # Find spots with smallest tumor proportion as confident normal spots.
        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            normal_candidate = (single_tumor_prop < prop_threshold)
            if np.sum(copy_single_X_rdr[:, (normal_candidate==True)]) > single_X.shape[0] * 200:
                break
    # filter bins with potential allele-specific expression (ASE) based on normal
    index_normal = np.where(normal_candidate)[0]
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_gene_snp = bin_selection_basedon_normal(df_gene_snp, \
        single_X, single_base_nb_mean, single_total_bb_RD, config['nu'], config['logphase_shift'], index_normal, config['geneticmap_file'])
    df_bininfo = genesnp_to_bininfo(df_gene_snp)
    df_bininfo['LOG_PHASE_TRANSITION'] = log_sitewise_transmat

    # filter out high-UMI DE genes, which may bias RDR estimates
    copy_single_X_rdr = copy.copy(single_X[:,0,:])
    copy_single_X_rdr, _ = filter_de_genes_tri(exp_counts, df_bininfo, normal_candidate, sample_list=sample_list, sample_ids=sample_ids)
    MIN_NORMAL_COUNT_PERBIN = 20
    bidx_inconfident = np.where( np.sum(copy_single_X_rdr[:, (normal_candidate==True)], axis=1) < MIN_NORMAL_COUNT_PERBIN )[0]
    rdr_normal = np.sum(copy_single_X_rdr[:, (normal_candidate==True)], axis=1)
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)
    copy_single_X_rdr[bidx_inconfident, :] = 0 # avoid ill-defined distributions if normal has 0 count in that bin.
    copy_single_base_nb_mean = rdr_normal.reshape(-1,1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1,-1)
        
    # adding back RDR signal
    single_X[:,0,:] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean
    n_obs = single_X.shape[0]
    df_bininfo['NORMAL_COUNT'] = copy_single_base_nb_mean[:,0]

    # # save parsed data
    # np.savez(f"{outdir}/binned_data.npz", lengths=lengths, single_X=single_X, single_base_nb_mean=single_base_nb_mean, single_total_bb_RD=single_total_bb_RD, log_sitewise_transmat=log_sitewise_transmat, single_tumor_prop=(None if config["tumorprop_file"] is None else single_tumor_prop))
    
    # save file
    df_bininfo.to_csv( f"{outdir}/table_bininfo.csv.gz", header=True, index=False, sep="\t" )
    
    table_rdrbaf = []
    for i in range(single_X.shape[2]):
        table_rdrbaf.append( pd.DataFrame({"BARCODES":barcodes[i], "EXP":single_X[:,0,i], "TOT":single_total_bb_RD[:,i], "B":single_X[:,1,i]}) )
    table_rdrbaf = pd.concat(table_rdrbaf, ignore_index=True)
    table_rdrbaf.to_csv( f"{outdir}/table_rdrbaf.csv.gz", header=True, index=False, sep="\t" )
    
    df_gene_snp.to_csv( f"{outdir}/gene_snp_info.csv.gz", header=True, index=False, sep="\t" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    parser.add_argument("--foldername", help="folder name to save inferred BAF clone results", required=True, type=str)
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

    # load BAF clone results
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    bafclone_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_BAFCLONES}"
    merged_res = dict(np.load(f"{bafclone_outdir}/mergedallspots_nstates{config['n_states']}_sp.npz", allow_pickle=True))
    n_baf_clones = len(np.unique(merged_res["new_assignment"]))
    pred = np.argmax(merged_res["log_gamma"], axis=0)
    pred = np.array([ pred[(c*n_obs):(c*n_obs+n_obs)] for c in range(n_baf_clones) ])
    merged_baf_profiles = np.array([ np.where(pred[c,:] < config["n_states"], merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0], 1-merged_res["new_p_binom"][pred[c,:]%config["n_states"], 0]) \
                                    for c in range(n_baf_clones) ])

    # filter allele-specific expression
    run_filter_ase(config, args.foldername, single_X, single_base_nb_mean, single_total_bb_RD, lengths, df_gene_snp, barcodes, single_tumor_prop, sample_list, sample_ids, smooth_mat, exp_counts, merged_baf_profiles)