import sys
import argparse

import numpy as np
import scipy
import pandas as pd
from itertools import cycle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from calicost.arg_parse import *
from calicost.parse_input import *
from calicost.utils_plotting import *


def plot_individual_spots_in_space_updated(coords, assignment, single_tumor_prop=None, sample_list=None, sample_ids=None, base_width=4, base_height=3, point_size=10, palette="Set2"):
    # combine coordinates across samples
    shifted_coords = copy.copy(coords)
    if not (sample_ids is None):
        x_offset = 0
        for s,sname in enumerate(sample_list):
            index = np.where(sample_ids == s)[0]
            shifted_coords[index,0] = shifted_coords[index,0] + x_offset
            x_offset += np.max(coords[index,0]) + 10

    # number of clones and samples
    final_clone_ids = np.unique(assignment[~assignment.isnull()].values)
    n_final_clones = len(final_clone_ids)
    n_samples = 1 if sample_list is None else len(sample_list)

    # remove nan of single_tumor_prop
    if not single_tumor_prop is None:
        copy_single_tumor_prop = copy.copy(single_tumor_prop)
        copy_single_tumor_prop[np.isnan(copy_single_tumor_prop)] = 0.5
    
    fig, axes = plt.subplots(1, 1, figsize=(base_width*n_samples, base_height), dpi=200, facecolor="white")
    if "clone 0" in final_clone_ids:
        colorlist = ['lightgrey'] + seaborn.color_palette(palette, n_final_clones-1).as_hex()
    else:
        colorlist = seaborn.color_palette(palette, n_final_clones).as_hex()

    for c,cid in enumerate(final_clone_ids):
        idx = np.where( (assignment.values==cid) )[0]
        if single_tumor_prop is None:
            seaborn.scatterplot(x=shifted_coords[idx,0], y=-shifted_coords[idx,1], s=point_size, color=colorlist[c], linewidth=0, legend=None, ax=axes)
        else:
            # cmap
            this_full_cmap = seaborn.color_palette(f"blend:lightgrey,{colorlist[c]}", as_cmap=True)
            quantile_colors = this_full_cmap(np.array([0, np.min(copy_single_tumor_prop[idx]), np.max(copy_single_tumor_prop[idx]), 1]))
            quantile_colors = [matplotlib.colors.rgb2hex(x) for x in quantile_colors[1:-1]]
            this_cmap = seaborn.color_palette(f"blend:{quantile_colors[0]},{quantile_colors[-1]}", as_cmap=True)
            seaborn.scatterplot(x=shifted_coords[idx,0], y=-shifted_coords[idx,1], s=point_size, hue=copy_single_tumor_prop[idx], palette=this_cmap, linewidth=0, legend=None, ax=axes)

    legend_elements = [Line2D([0], [0], marker='o', color="w", markerfacecolor=colorlist[c], label=cid, markersize=10) for c,cid in enumerate(final_clone_ids)]
    axes.legend(legend_elements, final_clone_ids, handlelength=0.1, loc="upper left", bbox_to_anchor=(1,1))
    axes.axis("off")

    fig.tight_layout()
    return fig


def main(configuration_file, output_file, r_hmrf_initialization=None, base_width=4, base_height=3, point_size=20, palette='Set2'):
    """
    Plot clones in space based on existing CalicoST output (output_dir/clone<n_clones>_rectangle<r_hmrf_initialization>_w<spatial_weight>/clone_labels.tsv)
    
    :param configuration_file: Path to the configuration file
    :param output_file: Path to the output file
    :param base_width: Width of the figure
    :param base_height: Height of the figure
    :param point_size: Size of the points
    :param palette: Color palette
    """
    # Load configuration
    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # Load data
    lengths, single_X, single_base_nb_mean, single_total_bb_RD, log_sitewise_transmat, df_bininfo, df_gene_snp, \
        barcodes, coords, single_tumor_prop, sample_list, sample_ids, adjacency_mat, smooth_mat, exp_counts = run_parse_n_load(config)

    # Load results and plot
    if r_hmrf_initialization is None:
        r_hmrf_initialization = config["num_hmrf_initialization_start"]
    
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    assert Path(outdir).is_dir(), f"Directory {outdir} does not exist. Did you run CalicoST with the same random seed 'num_hmrf_initialization_start'?"

    df_clone_label = pd.read_csv(f"{outdir}/clone_labels.tsv", header=0, index_col=0, sep="\t")
    assignment = pd.Series([f"clone {x}" for x in df_clone_label.clone_label.values])

    fig = plot_individual_spots_in_space_updated(coords, assignment, single_tumor_prop, sample_list=sample_list, sample_ids=sample_ids,
                                                    base_width=base_width, base_height=base_height, point_size=point_size, palette=palette)
    fig.savefig(output_file, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot clones in space based on existing CalicoST output")
    parser.add_argument("-c", "--configfile", help="configuration file of CalicoST", required=True, type=str)
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--r_hmrf_initialization", type=int, default=None, help="Random seed for HMRF initialization")
    parser.add_argument("--base_width", type=float, default=4, help="Width of the figure")
    parser.add_argument("--base_height", type=float, default=3, help="Height of the figure")
    parser.add_argument("--point_size", type=int, default=20, help="Size of the points")
    parser.add_argument("--palette", type=str, default="Set2", help="Color palette")
    args = parser.parse_args()

    main(args.configfile, args.output_file, args.r_hmrf_initialization, args.base_width, args.base_height, args.point_size, args.palette)