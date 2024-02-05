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
from calicost.utils_plotting import *


def run_makeplots(config, foldername, configuration_file, res_combine, coords, single_tumor_prop, sample_ids, sample_list):
    # outdir
    r_hmrf_initialization = config["num_hmrf_initialization_start"]
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{foldername}"

    # create directory
    p = subprocess.Popen(f"mkdir -p {outdir}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()

    # combined HMRF-HMM results
    final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))
    # note that don't need to append additional 0 for normal clone here

    cn_file = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_INTEGER_CN}/cnv_diploid_seglevel.tsv"
    medfix = ["", "_diploid", "_triploid", "_tetraploid"]

    # plot RDR and BAF
    fig = plot_rdr_baf(configuration_file, r_hmrf_initialization, cn_file, clone_ids=None, remove_xticks=True, rdr_ylim=5, chrtext_shift=-0.3, base_height=3.2, pointsize=30, palette="tab10")
    fig.savefig(f"{outdir}/plots/rdr_baf_defaultcolor.pdf", transparent=True, bbox_inches="tight")
    
    # plot allele-specific copy number
    for o,max_medploidy in enumerate([None, 2, 3, 4]):
        cn_file = f"{outdir}/cnv{medfix[o]}_seglevel.tsv"
        if not Path(cn_file).exists():
            continue
        df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
        df_cnv = expand_df_cnv(df_cnv)
        df_cnv = df_cnv[~df_cnv.iloc[:,-1].isnull()]
        fig, axes = plt.subplots(1, 1, figsize=(15, 0.9*len(final_clone_ids) + 0.6), dpi=200, facecolor="white")
        axes = plot_acn_from_df_anotherscheme(df_cnv, axes, chrbar_pos='top', chrbar_thickness=0.3, add_legend=False, remove_xticks=True)
        fig.tight_layout()
        fig.savefig(f"{outdir}/plots/acn_genome{medfix[o]}.pdf", transparent=True, bbox_inches="tight")
        # additionally plot the allele-specific copy number per region
        if not config["supervision_clone_file"] is None:
            fig, axes = plt.subplots(1, 1, figsize=(15, 0.6*len(final_clone_ids) + 0.4), dpi=200, facecolor="white")
            merged_df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
            df_cnv = merged_df_cnv[["CHR", "START", "END"]]
            df_cnv = df_cnv.join( pd.DataFrame({f"clone{x} A":merged_df_cnv[f"clone{res_combine['new_assignment'][i]} A"] for i,x in enumerate(final_clone_ids)}) )
            df_cnv = df_cnv.join( pd.DataFrame({f"clone{x} B":merged_df_cnv[f"clone{res_combine['new_assignment'][i]} B"] for i,x in enumerate(final_clone_ids)}) )
            df_cnv = expand_df_cnv(df_cnv)
            clone_ids = np.concatenate([ final_clone_ids[res_combine["new_assignment"]==c].astype(str) for c in final_clone_ids ])
            axes = plot_acn_from_df(df_cnv, axes, clone_ids=clone_ids, clone_names=[f"region {x}" for x in clone_ids], add_chrbar=True, add_arrow=False, chrbar_thickness=0.4/(0.6*len(final_clone_ids) + 0.4), add_legend=True, remove_xticks=True)
            fig.tight_layout()
            fig.savefig(f"{outdir}/plots/acn_genome{medfix[o]}_per_region.pdf", transparent=True, bbox_inches="tight")
    
    # plot clones in space
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    fig = plot_individual_spots_in_space(coords, assignment, single_tumor_prop, sample_list=sample_list, sample_ids=sample_ids)
    fig.savefig(f"{outdir}/plots/clone_spatial.pdf", transparent=True, bbox_inches="tight")


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
    # note that don't need to load filtered allele-specific expression data matrices since they are not used in the function

    # load combined HMRF-HMM results
    rdrbaf_outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}/{FOLDER_RDRBAFCLONES}"
    res_combine = dict(np.load(f"{rdrbaf_outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True))

    # make plots
    run_makeplots(config, args.foldername, args.configfile, res_combine, coords, single_tumor_prop, sample_ids, sample_list)