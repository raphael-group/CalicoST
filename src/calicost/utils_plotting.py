
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

from calicost.utils_IO import *
from calicost.utils_phase_switch import *
from calicost.hmrf import *
from calicost.arg_parse import *


def get_full_palette():
    palette = {}
    palette.update({(0, 0) : 'darkblue'})
    palette.update({(1, 0) : 'lightblue'})
    palette.update({(1, 1) : 'lightgray', (2, 0) : 'dimgray'})
    palette.update({(2, 1) : 'lightgoldenrodyellow', (3, 0) : 'gold'})
    # palette.update({(2, 1) : 'greenyellow', (3, 0) : 'darkseagreen'})
    palette.update({(2, 2) : 'navajowhite', (3, 1) : 'orange', (4, 0) : 'darkorange'})
    palette.update({(3, 2) : 'salmon', (4, 1) : 'red', (5, 0) : 'darkred'})
    palette.update({(3, 3) : 'plum', (4, 2) : 'orchid', (5, 1) : 'purple', (6, 0) : 'indigo'})
    ordered_acn = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), \
                   (2, 2), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), \
                   (3, 3), (4, 2), (5, 1), (6, 0)]
    return palette, ordered_acn


def plot_acn(cn_file, ax_handle, clone_ids=None, clone_names=None, add_chrbar=True, add_arrow=True, chrbar_thickness=0.1, add_legend=True, remove_xticks=True):
    # full color palette
    palette,_ = get_full_palette()

    # read CN profiles
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    print(final_clone_ids)
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])

    found = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        found += list(zip(major, minor))
    found = list(set(found))
    found.sort()

    # map CN to single digit number
    map_cn = {x:i for i,x in enumerate(found)}
    cnv_mapped = []
    ploidy = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        cnv_mapped.append( [map_cn[(major[i], minor[i])] for i in range(len(major))] )
        ploidy.append( np.mean(major + minor) )
    cnv_mapped = pd.DataFrame( np.array(cnv_mapped), index=[f"clone {cid}" for cid in final_clone_ids])
    ploidy = pd.DataFrame(np.around(np.array(ploidy), decimals=2).reshape(-1,1), index=[f"clone {cid}" for cid in final_clone_ids])
    chr_ids = df_cnv.CHR

    colors = [palette[c] for c in found]
    if clone_ids is None:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in final_clone_ids]
        rename_cnv_mapped = pd.DataFrame(cnv_mapped.values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(final_clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)
    else:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in clone_ids]
        if clone_names is None:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        else:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"{clone_names[c]}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)

    # indicate allele switches
    if add_arrow:
        if clone_ids is None:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(final_clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y2+0.1*y1, dx=0, dy=0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
                    # down-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y1+0.1*y2, dx=0, dy=-0.7*(y1-y2), head_width=0.3*(sub_int[1]-sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
        else:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y2+0.1*y1, dx=0, dy=0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
                    # down-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y1+0.1*y2, dx=0, dy=-0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")

    if add_chrbar:
        # add chr color
        chr_palette = cycle(['#525252', '#969696', '#cccccc'])
        lut = {c:next(chr_palette) for c in np.unique(chr_ids.values)}
        col_colors = chr_ids.map(lut)
        for i, color in enumerate(col_colors):
            ax_handle.add_patch(plt.Rectangle(xy=(i, 1.01), width=1, height=chrbar_thickness, color=color, lw=0, transform=ax_handle.get_xaxis_transform(), clip_on=False, rasterized=True))

        for c in np.unique(chr_ids.values):
            interval = np.where(chr_ids.values == c)[0]
            mid = np.percentile(interval, 45)
            ax_handle.text(mid-10, 1.04, str(c), transform=ax_handle.get_xaxis_transform())

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        ax_handle.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1))
    return ax_handle


def plot_acn_from_df(df_cnv, ax_handle, clone_ids=None, clone_names=None, add_chrbar=True, add_arrow=True, chrbar_thickness=0.1, add_legend=True, remove_xticks=True, rasterized=True):
    # full color palette
    palette,_ = get_full_palette()

    # read CN profiles
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    print(final_clone_ids)
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])

    found = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        found += list(zip(major, minor))
    found = list(set(found))
    found.sort()

    # map CN to single digit number
    map_cn = {x:i for i,x in enumerate(found)}
    cnv_mapped = []
    ploidy = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        cnv_mapped.append( [map_cn[(major[i], minor[i])] for i in range(len(major))] )
        ploidy.append( np.mean(major + minor) )
    cnv_mapped = pd.DataFrame( np.array(cnv_mapped), index=[f"clone {cid}" for cid in final_clone_ids])
    ploidy = pd.DataFrame(np.around(np.array(ploidy), decimals=2).reshape(-1,1), index=[f"clone {cid}" for cid in final_clone_ids])
    chr_ids = df_cnv.CHR

    colors = [palette[c] for c in found]
    if clone_ids is None:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in final_clone_ids]
        rename_cnv_mapped = pd.DataFrame(cnv_mapped.values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(final_clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=rasterized, ax=ax_handle)
    else:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in clone_ids]
        if clone_names is None:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        else:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"{clone_names[c]}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=rasterized, ax=ax_handle)

    # indicate allele switches
    if add_arrow:
        if clone_ids is None:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(final_clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y2+0.1*y1, dx=0, dy=0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
                    # down-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y1+0.1*y2, dx=0, dy=-0.7*(y1-y2), head_width=0.3*(sub_int[1]-sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
        else:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y2+0.1*y1, dx=0, dy=0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")
                    # down-arrow
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black")
                            ax_handle.arrow(x=intervals[i][0]+np.mean(sub_int), y=0.9*y1+0.1*y2, dx=0, dy=-0.7*(y1-y2), head_width=0.3*(sub_int[1] - sub_int[0]), head_length=0.1*np.abs(y1-y2), fc="black")

    if add_chrbar:
        # add chr color
        chr_palette = cycle(['#525252', '#969696', '#cccccc'])
        lut = {c:next(chr_palette) for c in np.unique(chr_ids.values)}
        col_colors = chr_ids.map(lut)
        for i, color in enumerate(col_colors):
            ax_handle.add_patch(plt.Rectangle(xy=(i, 1 + 0.02*chrbar_thickness), width=1, height=chrbar_thickness, color=color, lw=0, transform=ax_handle.get_xaxis_transform(), clip_on=False, rasterized=rasterized))

        for c in np.unique(chr_ids.values):
            interval = np.where(chr_ids.values == c)[0]
            mid = np.percentile(interval, 45)
            ax_handle.text(mid-10, 1 + 0.2*chrbar_thickness, str(c), transform=ax_handle.get_xaxis_transform())

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        ax_handle.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1))
    return ax_handle


def plot_acn_from_df_anotherscheme(df_cnv, ax_handle, clone_ids=None, clone_names=None, clone_proportions=None, chrbar_pos=None, add_arrow=True, border_linewidth=1, chrbar_thickness=0.1, add_legend=True, remove_xticks=True, rasterized=True):
    # full color palette
    palette,_ = get_full_palette()

    # read CN profiles
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    print(final_clone_ids)
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])

    found = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        found += list(zip(major, minor))
    found = list(set(found))
    found.sort()

    # map CN to single digit number
    map_cn = {x:i for i,x in enumerate(found)}
    cnv_mapped = []
    ploidy = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        cnv_mapped.append( [map_cn[(major[i], minor[i])] for i in range(len(major))] )
        ploidy.append( np.mean(major + minor) )
    cnv_mapped = pd.DataFrame( np.array(cnv_mapped), index=[f"clone {cid}" for cid in final_clone_ids])
    ploidy = pd.DataFrame(np.around(np.array(ploidy), decimals=2).reshape(-1,1), index=[f"clone {cid}" for cid in final_clone_ids])
    chr_ids = df_cnv.CHR

    colors = [palette[c] for c in found]
    if clone_ids is None:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in final_clone_ids]
        rename_cnv_mapped = pd.DataFrame(cnv_mapped.values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(final_clone_ids)])
        if len(np.unique(rename_cnv_mapped.values)) == 1:
            colors = colors + colors
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=rasterized, ax=ax_handle)
    else:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in clone_ids]
        if clone_names is None:
            index_str = [f"clone {cid}\nploidy {tmp_ploidy[c]}"  for c,cid in enumerate(clone_ids)]
        else:
            index_str = [f"{clone_names[c]}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)]
        if not clone_proportions is None:
            index_str = [f"{index_str[c]}\nu={clone_proportions[c]:.2f}" for c in range(len(clone_ids))]
        rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=index_str)
        if len(np.unique(rename_cnv_mapped.values)) == 1:
            colors = colors + colors
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=rasterized, ax=ax_handle)

    # indicate allele switches
    if add_arrow:
        if clone_ids is None:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in final_clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(final_clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    y_diverge1 = 0.8*y2+0.2*y1
                    y_diverge2 = 0.6*y2+0.4*y1
                    y_merge = 0.7*y2+0.3*y1
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            # bounding box
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black", linewidth=border_linewidth)
                            # arrow
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], [y_diverge1,y_merge], [y_diverge2,y_merge], color="black", edgecolor="black")
                    # down-arrow
                    y_diverge1 = 0.2*y2+0.8*y1
                    y_diverge2 = 0.4*y2+0.6*y1
                    y_merge = 0.3*y2+0.7*y1
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            # bounding box
                            ax_handle.fill_between( np.arange(intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]), y1, y2, color="none", edgecolor="black", linewidth=border_linewidth)
                            # arrow
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], [y_merge,y_diverge1], [y_merge,y_diverge2], color="black", edgecolor="black")
        else:
            # find regions where there exist both clones with A > B and clones with A < B
            has_up = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values > df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            has_down = np.any(np.vstack([ df_cnv[f"clone{cid} A"].values < df_cnv[f"clone{cid} B"].values for cid in clone_ids]), axis=0)
            intervals, labs = get_intervals( (has_up & has_down) )
            # for each intervals, find the corresponding clones with A > B to plot up-arrow, and corresponding clones with A < B to plot down-arrow
            for i in range(len(intervals)):
                if not labs[i]:
                    continue
                for c,cid in enumerate(clone_ids):
                    y1 = c
                    y2 = c+1
                    # up-arrow
                    y_diverge1 = 0.8*y2+0.2*y1
                    y_diverge2 = 0.6*y2+0.4*y1
                    y_merge = 0.7*y2+0.3*y1
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] > df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            # bounding box
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], y1, y2, color="none", edgecolor="black", linewidth=border_linewidth)
                            # arrow
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], [y_diverge1,y_merge], [y_diverge2,y_merge], color="black", edgecolor="black")
                    # down-arrow
                    y_diverge1 = 0.2*y2+0.8*y1
                    y_diverge2 = 0.4*y2+0.6*y1
                    y_merge = 0.3*y2+0.7*y1
                    sub_intervals, sub_labs = get_intervals( df_cnv[f"clone{cid} A"].values[intervals[i][0]:intervals[i][1]] < df_cnv[f"clone{cid} B"].values[intervals[i][0]:intervals[i][1]] )
                    for j, sub_int in enumerate(sub_intervals):
                        if sub_labs[j]:
                            # bounding box
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], y1, y2, color="none", edgecolor="black", linewidth=border_linewidth)
                            # arrow
                            ax_handle.fill_between( [intervals[i][0]+sub_int[0], intervals[i][0]+sub_int[1]], [y_merge,y_diverge1], [y_merge,y_diverge2], color="black", edgecolor="black")

            # # horizontal separation between clones
            # for c,cid in enumerate(clone_ids[:-1]):
            #     ax_handle.axhline(y=c+1, color="black", lw=0.5)

    if chrbar_pos == "bottom":
        chr_ids = df_cnv.CHR
        h = len(final_clone_ids) if clone_ids is None else len(clone_ids)
        # ax_handle.add_patch(plt.Rectangle(xy=(0, h + chrbar_thickness), width=df_cnv.shape[0], height=chrbar_thickness, color='white', lw=0, transform=ax_handle.transData, clip_on=False, rasterized=rasterized))

        for i,c in enumerate(np.unique(chr_ids.values)):
            interval = np.where(chr_ids.values == c)[0]
            # add vertical separation between chromosomes
            if not np.max(interval) + 1 >= df_cnv.shape[0]:
                ax_handle.axvline(x=np.max(interval), color='black', lw=0.5, ymin=-0.5/(h+1), clip_on = False)
            mid = np.percentile(interval, 45)
            if i % 2 == 0:
                ax_handle.text(mid, h + chrbar_thickness, str(c), ha='center', transform=ax_handle.transData)
            else:
                ax_handle.text(mid, h + 2*chrbar_thickness, str(c), ha='center', transform=ax_handle.transData)
    elif chrbar_pos == "top":
        chr_ids = df_cnv.CHR
        h = len(final_clone_ids) if clone_ids is None else len(clone_ids)
        # ax_handle.add_patch(plt.Rectangle(xy=(0, h + chrbar_thickness), width=df_cnv.shape[0], height=chrbar_thickness, color='white', lw=0, transform=ax_handle.transData, clip_on=False, rasterized=rasterized))

        for i,c in enumerate(np.unique(chr_ids.values)):
            interval = np.where(chr_ids.values == c)[0]
            # add vertical separation between chromosomes
            if not np.max(interval) + 1 >= df_cnv.shape[0]:
                ax_handle.axvline(x=np.max(interval), color='black', lw=0.5, ymax=1+0.5/(h+1), clip_on = False)
            mid = np.percentile(interval, 45)
            if i % 2 == 0:
                ax_handle.text(mid, -0.1*chrbar_thickness, str(c), ha='center', transform=ax_handle.transData)
            else:
                ax_handle.text(mid, -0.8*chrbar_thickness, str(c), ha='center', transform=ax_handle.transData)

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        ax_handle.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1))
    return ax_handle



def plot_acn_legend(fig, shift_y=0.3):
    # full palette
    palette, ordered_acn = get_full_palette()

    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [palette[c] for c in ordered_acn]
    cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors))

    n_total_cn = np.max([x[0]+x[1] for x in ordered_acn]) + 1
    gs = GridSpec(2*n_total_cn-1, 1, figure=fig)

    # total cn = 0
    ax = fig.add_subplot(gs[2*n_total_cn-2, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(0,0)]]).reshape((1,-1)), columns=["{0,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    # total cn = 1
    ax = fig.add_subplot(gs[2*n_total_cn-4, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(1,0)]]).reshape((1,-1)), columns=["{1,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    # total cn = 2
    ax = fig.add_subplot(gs[2*n_total_cn-6, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(1,1)], map_cn[(2,0)]]).reshape((1,-1)), columns=["{1,1}", "{2,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,0.3))

    # total cn = 3
    ax = fig.add_subplot(gs[2*n_total_cn-8, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(2,1)], map_cn[(3,0)]]).reshape((1,-1)), columns=["{2,1}", "{3,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    # total cn = 4
    ax = fig.add_subplot(gs[2*n_total_cn-10, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(2,2)], map_cn[(3,1)], map_cn[(4,0)]]).reshape((1,-1)), columns=["{2,2}", "{3,1}", "{4,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    # total cn = 5
    ax = fig.add_subplot(gs[2*n_total_cn-12, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(3,2)], map_cn[(4,1)], map_cn[(5,0)]]).reshape((1,-1)), columns=["{3,2}", "{4,1}", "{5,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    # total cn = 6
    ax = fig.add_subplot(gs[2*n_total_cn-14, :])
    seaborn.heatmap( pd.DataFrame(np.array([map_cn[(3,3)], map_cn[(4,2)], map_cn[(5,1)], map_cn[(6,0)]]).reshape((1,-1)), columns=["{3,3}", "{4,2}", "{5,1}", "{6,0}"]), vmin=0, vmax=len(colors), cmap=cmap, cbar=False, linewidths=1, linecolor="black" )
    ax.set_yticks([])
    ax.set_xticklabels(ax.get_xticklabels(), position=(0,shift_y))

    return fig


def plot_acn_withhighlight(cn_file, df_highlight_events, ax_handle, clone_ids=None, clone_names=None, add_chrbar=True, chrbar_thickness=0.1, add_legend=True, remove_xticks=True):
    """
    df_highlight_events: dataframe with columns: ["BinSTART", "BinEND", "involved_clones"]
    """
    # full color palette
    palette,_ = get_full_palette()

    # read CN profiles
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    print(final_clone_ids)
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])

    found = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        found += list(zip(major, minor))
    found = list(set(found))
    found.sort()

    # map CN to single digit number
    map_cn = {x:i for i,x in enumerate(found)}
    cnv_mapped = []
    ploidy = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        cnv_mapped.append( [map_cn[(major[i], minor[i])] for i in range(len(major))] )
        ploidy.append( np.mean(major + minor) )
    cnv_mapped = pd.DataFrame( np.array(cnv_mapped), index=[f"clone {cid}" for cid in final_clone_ids])
    ploidy = pd.DataFrame(np.around(np.array(ploidy), decimals=2).reshape(-1,1), index=[f"clone {cid}" for cid in final_clone_ids])
    chr_ids = df_cnv.CHR

    colors = [palette[c] for c in found]
    if clone_ids is None:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in final_clone_ids]
        rename_cnv_mapped = pd.DataFrame(cnv_mapped.values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(final_clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)
    else:
        tmp_ploidy = [ploidy.loc[f"clone {cid}"].values[0] for cid in clone_ids]
        if clone_names is None:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"clone {cid}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        else:
            rename_cnv_mapped = pd.DataFrame(cnv_mapped.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"{clone_names[c]}\nploidy {tmp_ploidy[c]}" for c,cid in enumerate(clone_ids)])
        seaborn.heatmap(rename_cnv_mapped, cmap=LinearSegmentedColormap.from_list('multi-level', colors, len(colors)), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)

    for i in range(df_highlight_events.shape[0]):
        involved_clones = df_highlight_events.involved_clones.values[i]
        # interval start and end
        interval = [df_highlight_events.BinSTART.values[i], df_highlight_events.BinEND.values[i]]
        if clone_ids is None:
            for c, cid in enumerate(final_clone_ids):
                if not cid in involved_clones:
                    continue
                y1 = c
                y2 = c+1
                ax_handle.fill_between( np.arange(interval[0], interval[1]), y1, y2, color="none", edgecolor="black", linewidth=2)
        else:
            for c, cid in enumerate(clone_ids):
                if not cid in involved_clones:
                    continue
                y1 = c
                y2 = c+1
                ax_handle.fill_between( np.arange(interval[0], interval[1]), y1, y2, color="none", edgecolor="black", linewidth=2)
        
    if add_chrbar:
        # add chr color
        chr_palette = cycle(['#525252', '#969696', '#cccccc'])
        lut = {c:next(chr_palette) for c in np.unique(chr_ids.values)}
        col_colors = chr_ids.map(lut)
        for i, color in enumerate(col_colors):
            ax_handle.add_patch(plt.Rectangle(xy=(i, 1.01), width=1, height=chrbar_thickness, color=color, lw=0, transform=ax_handle.get_xaxis_transform(), clip_on=False, rasterized=True))

        for c in np.unique(chr_ids.values):
            interval = np.where(chr_ids.values == c)[0]
            mid = np.percentile(interval, 45)
            ax_handle.text(mid-10, 1.04, str(c), transform=ax_handle.get_xaxis_transform())

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, 
        color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        ax_handle.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1 - 0.1 * min(0, rename_cnv_mapped.shape[0]-6)))
    return ax_handle


def plot_total_cn(df_cnv, ax_handle, df_highlight_events=None, palette_mode=6, clone_ids=None, clone_names=None, add_chrbar=True, chrbar_thickness=0.1, add_legend=True, legend_position="upper left", remove_xticks=True):
    """
    df_cnv : pandas.DataFrame
        Each row is a genomic bin, containing columns "CHR", "clone {cid}" for each clone id.
    palette_mode : int
        Either 6 for 6-state palette, or 3 for 3-state palette.
    """
    chr_ids = df_cnv.CHR

    # create a cmap that map "amp" to #B44F3D, "bamp" to #E18073, "bdel" to #A0CEEA, "del" to #4F69DF, "loh" to #738B2D
    if palette_mode == 6:
        full_palette = {"amp":"#B44F3D", "bamp":"#E18073", "bdel":"#A0CEEA", "del":"#4F69DF", "loh":"#738B2D", "neu":"lightgrey"}
    else:
        full_palette = {"amp":"#B44F3D", "del":"#4F69DF", "neu":"lightgrey"}

    if clone_ids is None:
        found = np.unique(df_cnv.iloc[:, df_cnv.columns.str.startswith("clone")].values.flatten())
        lut = {x:i for i,x in enumerate(found)}
        palette = matplotlib.colors.ListedColormap([full_palette[x] for x in found])
        df_cnv_mapped = df_cnv.iloc[:, df_cnv.columns.str.startswith("clone")].replace(lut)
        df_cnv_mapped = df_cnv_mapped.T
        seaborn.heatmap(df_cnv_mapped, cmap=palette, linewidths=0, cbar=False, rasterized=True, ax=ax_handle)
    else:
        found = np.unique(df_cnv[[f"clone {cid}" for cid in clone_ids]].values.flatten())
        lut = {x:i for i,x in enumerate(found)}
        palette = matplotlib.colors.ListedColormap([full_palette[x] for x in found])
        df_cnv_mapped = df_cnv[[f"clone {cid}" for cid in clone_ids]].replace(lut)
        df_cnv_mapped = df_cnv_mapped.T
        if not clone_names is None:
            df_cnv_mapped.rename(index={f"clone {cid}":clone_names[i] for i,cid in enumerate(clone_ids)}, inplace=True)
        seaborn.heatmap(df_cnv_mapped, cmap=palette, linewidths=0, cbar=False, rasterized=True, ax=ax_handle)

    if not df_highlight_events is None:
        final_clone_ids = [x.split(" ")[1] for x in df_cnv.columns if x.startswith("clone")]
        for i in range(df_highlight_events.shape[0]):
            involved_clones = df_highlight_events.involved_clones.values[i]
            # interval start and end
            interval = [df_highlight_events.BinSTART.values[i], df_highlight_events.BinEND.values[i]]
            if clone_ids is None:
                for c, cid in enumerate(final_clone_ids):
                    if not cid in involved_clones:
                        continue
                    y1 = c
                    y2 = c+1
                    ax_handle.fill_between( np.arange(interval[0], interval[1]), y1, y2, color="none", edgecolor="black", linewidth=2)
            else:
                for c, cid in enumerate(clone_ids):
                    if not cid in involved_clones:
                        continue
                    y1 = c
                    y2 = c+1
                    ax_handle.fill_between( np.arange(interval[0], interval[1]), y1, y2, color="none", edgecolor="black", linewidth=2)

    if add_chrbar:
        # add chr color
        chr_palette = cycle(['#525252', '#969696', '#cccccc'])
        lut = {c:next(chr_palette) for c in np.unique(chr_ids.values)}
        col_colors = chr_ids.map(lut)
        for i, color in enumerate(col_colors):
            ax_handle.add_patch(plt.Rectangle(xy=(i, 1 + 0.02*chrbar_thickness), width=1, height=chrbar_thickness, color=color, lw=0, transform=ax_handle.get_xaxis_transform(), clip_on=False, rasterized=True))

        for c in np.unique(chr_ids.values):
            interval = np.where(chr_ids.values == c)[0]
            mid = np.percentile(interval, 45)
            ax_handle.text(mid-10, 1 + 0.2*chrbar_thickness, str(c), transform=ax_handle.get_xaxis_transform())

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    if add_legend:
        if palette_mode == 6:
            a0 = plt.arrow(0,0, 0,0, color='#B44F3D')
            a1 = plt.arrow(0,0, 0,0, color='#E18073')
            a2 = plt.arrow(0,0, 0,0, color='lightgrey')
            a3 = plt.arrow(0,0, 0,0, color='#A0CEEA')
            a4 = plt.arrow(0,0, 0,0, color='#4F69DF')
            a5 = plt.arrow(0,0, 0,0, color='#738B2D')
            if legend_position == "upper left":
                ax_handle.legend([a0, a1, a2, a3, a4, a5], ["amp", "bamp", "neu", "bdel", "del", "loh"], loc='upper left', bbox_to_anchor=(1,1 - 0.1 * min(0, df_cnv_mapped.shape[0]-5)))
            else:
                ax_handle.legend([a0, a1, a2, a3, a4, a5], ["amp", "bamp", "neu", "bdel", "del", "loh"], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=6)
        else:
            a0 = plt.arrow(0,0, 0,0, color='#B44F3D')
            a1 = plt.arrow(0,0, 0,0, color='lightgrey')
            a2 = plt.arrow(0,0, 0,0, color='#4F69DF')
            if legend_position == "upper left":
                ax_handle.legend([a0, a1, a2], ["amp", "neu", "del"], loc='upper left', bbox_to_anchor=(1,1 - 0.1 * min(0, df_cnv_mapped.shape[0]-2)))
            else:
                ax_handle.legend([a0, a1, a2], ["amp", "neu", "del"], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    return ax_handle


def plot_amp_del(cn_file, ax_handle, clone_ids=None, clone_names=None, add_chrbar=True, chrbar_thickness=0.1, add_legend=True, remove_xticks=True):
    # define color palette that maps 0 to lightgrey, -2 and -1 to blues with increasing intensity, and 1 and 2 to reds with increasing intensity
    palette_map = {-2+i:x for i,x in enumerate(seaborn.color_palette("coolwarm", 5).as_hex())}
    
    # read CN profiles
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])

    # compute the relative copy number with respect to the median copy number per clone
    df_cnv_rel = []
    for cid in final_clone_ids:
        major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
        median_copy = np.median(major + minor)
        # clamp the relative copy number major + minor - median_copy to [-2,2]
        df_cnv_rel.append( np.minimum(2, np.maximum(-2, major + minor - median_copy)) )
    df_cnv_rel = pd.DataFrame( np.array(df_cnv_rel), index=[f"clone {cid}" for cid in final_clone_ids])

    # plot heatmap
    if clone_ids is None:
        rename_cnv_mapped = pd.DataFrame(df_cnv_rel.values, index=[f"clone {cid}" for c,cid in enumerate(final_clone_ids)])
        unique_cnv_values = np.unique(rename_cnv_mapped.values)
        seaborn.heatmap(rename_cnv_mapped, cmap=ListedColormap([palette_map[x] for x in unique_cnv_values]), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)
    else:
        if clone_names is None:
            rename_cnv_mapped = pd.DataFrame(df_cnv_rel.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"clone {cid}" for c,cid in enumerate(clone_ids)])
        else:
            rename_cnv_mapped = pd.DataFrame(df_cnv_rel.loc[[f"clone {cid}" for cid in clone_ids]].values, index=[f"{clone_names[c]}" for c,cid in enumerate(clone_ids)])
        unique_cnv_values = np.unique(rename_cnv_mapped.values)
        seaborn.heatmap(rename_cnv_mapped, cmap=ListedColormap([palette_map[x] for x in unique_cnv_values]), linewidths=0, cbar=False, rasterized=True, ax=ax_handle)
    
    if add_chrbar:
        chr_ids = df_cnv.CHR
        # add chr color
        chr_palette = cycle(['#525252', '#969696', '#cccccc'])
        lut = {c:next(chr_palette) for c in np.unique(chr_ids.values)}
        col_colors = chr_ids.map(lut)
        for i, color in enumerate(col_colors):
            ax_handle.add_patch(plt.Rectangle(xy=(i, 1.01), width=1, height=chrbar_thickness, color=color, lw=0, transform=ax_handle.get_xaxis_transform(), clip_on=False, rasterized=True))

        for c in np.unique(chr_ids.values):
            interval = np.where(chr_ids.values == c)[0]
            mid = np.percentile(interval, 45)
            ax_handle.text(mid-10, 1.04, str(c), transform=ax_handle.get_xaxis_transform())

    ax_handle.set_yticklabels(ax_handle.get_yticklabels(), rotation=0)
    if remove_xticks:
        ax_handle.set_xticks([])

    # add legend corresponding to palette
    if add_legend:
        a0 = plt.arrow(0,0, 0,0, color=palette_map[-2])
        a1 = plt.arrow(0,0, 0,0, color=palette_map[-1])
        a2 = plt.arrow(0,0, 0,0, color=palette_map[0])
        a3 = plt.arrow(0,0, 0,0, color=palette_map[1])
        a4 = plt.arrow(0,0, 0,0, color=palette_map[2])
        ax_handle.legend([a0, a1, a2, a3, a4], ['-2 and below','-1','0','1', '2 and above'], ncol=1, loc='upper left', bbox_to_anchor=(1,1))

    return ax_handle



def plot_rdr_baf(configuration_file, r_hmrf_initialization, cn_file, clone_ids=None, clone_names=None, remove_xticks=True, rdr_ylim=5, chrtext_shift=-0.3, base_height=3.2, pointsize=15, linewidth=1, palette="chisel"):
    # full palette
    chisel_palette, ordered_acn = get_full_palette()
    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # load allele specific integer copy numbers
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    if not '0' in final_clone_ids:
        final_clone_ids = np.array(['0'] + list(final_clone_ids))
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])
    unique_chrs = np.unique(df_cnv.CHR.values)

    # load data
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    dat = np.load(f"{outdir}/binned_data.npz", allow_pickle=True)
    lengths = dat["lengths"]
    single_X = dat["single_X"]
    single_base_nb_mean = dat["single_base_nb_mean"]
    single_total_bb_RD = dat["single_total_bb_RD"]
    single_tumor_prop = dat["single_tumor_prop"]
    res_combine = dict( np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True) )

    n_states = res_combine["new_p_binom"].shape[0]

    assert single_X.shape[0] == df_cnv.shape[0]

    clone_index = [np.where(res_combine["new_assignment"] == c)[0] for c,cid in enumerate(final_clone_ids)]
    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop)
    n_obs = X.shape[0]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    # plotting all clones
    if clone_ids is None:
        fig, axes = plt.subplots(2*len(nonempty_clones), 1, figsize=(20, base_height*len(nonempty_clones)), dpi=200, facecolor="white")
        for s,c in enumerate(nonempty_clones):
            cid = final_clone_ids[c]
            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            # plot points
            segments, labs = get_intervals(res_combine["pred_cnv"][:,c])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,0,c]/base_nb_mean[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=1, legend=False, ax=axes[2*s])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,0,c]/base_nb_mean[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", linewidth=linewidth, alpha=1, legend=False, ax=axes[2*s])
            axes[2*s].set_ylabel(f"clone {cid}\nRDR")
            axes[2*s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2*s].set_ylim([0,rdr_ylim])
            axes[2*s].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2*s].set_xticks([])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s+1])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s+1])
            axes[2*s+1].set_ylabel(f"clone {cid}\nphased AF")
            axes[2*s+1].set_ylim([-0.1, 1.1])
            axes[2*s+1].set_yticks([0, 0.5, 1])
            axes[2*s+1].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2*s+1].set_xticks([])
            for i, seg in enumerate(segments):
                axes[2*s].plot(seg, [np.exp(res_combine["new_log_mu"][labs[i],c]), np.exp(res_combine["new_log_mu"][labs[i],c])], c="black", linewidth=2)
                axes[2*s+1].plot(seg, [res_combine["new_p_binom"][labs[i],c], res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
                axes[2*s+1].plot(seg, [1-res_combine["new_p_binom"][labs[i],c], 1-res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)

        for i in range(len(lengths)):
            median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[:(i+1)]) * 0.45
            axes[-1].text(median_len-5, chrtext_shift, unique_chrs[i], transform=axes[-1].get_xaxis_transform())
            for k in range(2*len(nonempty_clones)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()
    # plot a given clone
    else:
        fig, axes = plt.subplots(2*len(clone_ids), 1, figsize=(20, base_height*len(clone_ids)), dpi=200, facecolor="white")
        for s,cid in enumerate(clone_ids):
            c = np.where(final_clone_ids == cid)[0][0]

            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            # plot points
            segments, labs = get_intervals(res_combine["pred_cnv"][:,c])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,0,c]/base_nb_mean[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,0,c]/base_nb_mean[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s])
            axes[2*s].set_ylabel(f"clone {cid}\nRDR" if clone_names is None else f"clone {clone_names[s]}\nRDR")
            axes[2*s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2*s].set_ylim([0,5])
            axes[2*s].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2*s].set_xticks([])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s+1])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[2*s+1])
            axes[2*s+1].set_ylabel(f"clone {cid}\nphased AF" if clone_names is None else f"clone {clone_names[s]}\nphased AF")
            axes[2*s+1].set_ylim([-0.1, 1.1])
            axes[2*s+1].set_yticks([0, 0.5, 1])
            axes[2*s+1].set_xlim([0, n_obs])
            if remove_xticks:
                axes[2*s+1].set_xticks([])
            for i, seg in enumerate(segments):
                axes[2*s].plot(seg, [np.exp(res_combine["new_log_mu"][labs[i],c]), np.exp(res_combine["new_log_mu"][labs[i],c])], c="black", linewidth=2)
                axes[2*s+1].plot(seg, [res_combine["new_p_binom"][labs[i],c], res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
                axes[2*s+1].plot(seg, [1-res_combine["new_p_binom"][labs[i],c], 1-res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
        
        for i in range(len(lengths)):
            median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[:(i+1)]) * 0.45
            axes[-1].text(median_len-5, chrtext_shift, unique_chrs[i], transform=axes[-1].get_xaxis_transform())
            for k in range(2*len(clone_ids)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()

    return fig



def plot_baf(configuration_file, r_hmrf_initialization, cn_file, clone_ids=None, clone_names=None, remove_xticks=True, rdr_ylim=5, chrtext_shift=-0.3, base_height=3.2, pointsize=15, linewidth=1, palette="chisel"):
    # full palette
    chisel_palette, ordered_acn = get_full_palette()
    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # load allele specific integer copy numbers
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df_cnv.columns[3:] ])
    if not '0' in final_clone_ids:
        final_clone_ids = np.array(['0'] + list(final_clone_ids))
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])
    unique_chrs = np.unique(df_cnv.CHR.values)

    # load data
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    dat = np.load(f"{outdir}/binned_data.npz", allow_pickle=True)
    lengths = dat["lengths"]
    single_X = dat["single_X"]
    single_base_nb_mean = dat["single_base_nb_mean"]
    single_total_bb_RD = dat["single_total_bb_RD"]
    single_tumor_prop = dat["single_tumor_prop"]
    res_combine = dict( np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True) )

    n_states = res_combine["new_p_binom"].shape[0]

    assert single_X.shape[0] == df_cnv.shape[0]

    clone_index = [np.where(res_combine["new_assignment"] == c)[0] for c,cid in enumerate(final_clone_ids)]
    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop)
    n_obs = X.shape[0]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    # plotting all clones
    if clone_ids is None:
        fig, axes = plt.subplots(len(nonempty_clones), 1, figsize=(20, base_height*len(nonempty_clones)), dpi=200, facecolor="white")
        for s,c in enumerate(nonempty_clones):
            cid = final_clone_ids[c]
            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            # plot points
            segments, labs = get_intervals(res_combine["pred_cnv"][:,c])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            axes[s].set_ylabel(f"clone {cid}\nphased AF")
            axes[s].set_ylim([-0.1, 1.1])
            axes[s].set_yticks([0, 0.5, 1])
            axes[s].set_xlim([0, n_obs])
            if remove_xticks:
                axes[s].set_xticks([])
            for i, seg in enumerate(segments):
                axes[s].plot(seg, [res_combine["new_p_binom"][labs[i],c], res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
                axes[s].plot(seg, [1-res_combine["new_p_binom"][labs[i],c], 1-res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)

        for i in range(len(lengths)):
            median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[:(i+1)]) * 0.45
            axes[-1].text(median_len-5, chrtext_shift, unique_chrs[i], transform=axes[-1].get_xaxis_transform())
            for k in range(len(nonempty_clones)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()
    # plot a given clone
    else:
        fig, axes = plt.subplots(2*len(clone_ids), 1, figsize=(20, base_height*len(clone_ids)), dpi=200, facecolor="white")
        for s,cid in enumerate(clone_ids):
            c = np.where(final_clone_ids == cid)[0][0]

            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            # plot points
            segments, labs = get_intervals(res_combine["pred_cnv"][:,c])
            if palette == "chisel":
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                    palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            else:
                seaborn.scatterplot(x=np.arange(X[:,1,c].shape[0]), y=X[:,1,c]/total_bb_RD[:,c], \
                    hue=pd.Categorical(res_combine["pred_cnv"][:,c], categories=np.arange(n_states), ordered=True), \
                    palette=palette, s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            axes[s].set_ylabel(f"clone {cid}\nphased AF" if clone_names is None else f"clone {clone_names[s]}\nphased AF")
            axes[s].set_ylim([-0.1, 1.1])
            axes[s].set_yticks([0, 0.5, 1])
            axes[s].set_xlim([0, n_obs])
            if remove_xticks:
                axes[s].set_xticks([])
            for i, seg in enumerate(segments):
                axes[s].plot(seg, [res_combine["new_p_binom"][labs[i],c], res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
                axes[s].plot(seg, [1-res_combine["new_p_binom"][labs[i],c], 1-res_combine["new_p_binom"][labs[i],c]], c="black", linewidth=2)
        
        for i in range(len(lengths)):
            median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[:(i+1)]) * 0.45
            axes[-1].text(median_len-5, chrtext_shift, unique_chrs[i], transform=axes[-1].get_xaxis_transform())
            for k in range(2*len(clone_ids)):
                axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
        fig.tight_layout()

    return fig


def plot_rdr_baf_from_df(df, clone_ids=None, clone_names=None, base_height=3.2, rdr_ylim=3, baf_ylim=0.5, baf_yticks=None, linewidth=0, pointsize=30, chrtext_shift=-0.3, add_legend=False, remove_xticks=True):
    """
    Attributes
    ----------
    df : pandas.DataFrame
        dataframe with columns: CHR, clone1 RD, clone1 BAF, clone1 A, clone1 B, ... for each clone
    """
    # full palette
    chisel_palette, ordered_acn = get_full_palette()
    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]
    
    # load allele specific integer copy numbers
    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df.columns if "RD" in x ])
    assert (clone_ids is None) or np.all([ (cid in final_clone_ids) for cid in clone_ids])
    unique_chrs = np.unique(df.CHR.values)

    if clone_ids is None:
        fig, axes = plt.subplots(2*len(final_clone_ids), 1, figsize=(20, base_height*len(final_clone_ids)), dpi=200, facecolor="white")
        for s,cid in enumerate(final_clone_ids):
            # major and minor allele copies give the hue
            major = np.maximum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)
            minor = np.minimum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)
            
            seaborn.scatterplot(x=np.arange(df.shape[0]), y=df[f'clone{cid} RD'].values, \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=0.8, legend=False, ax=axes[2*s])
            axes[2*s].set_ylabel(f"clone {cid}\nRDR")
            axes[2*s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2*s].set_ylim([0,rdr_ylim])
            axes[2*s].set_xlim([0, df.shape[0]])
            if remove_xticks:
                axes[2*s].set_xticks([])
            seaborn.scatterplot(x=np.arange(df.shape[0]), y=df[f"clone{cid} BAF"].values, \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=0.8, legend=False, ax=axes[2*s+1])
            axes[2*s+1].set_ylabel(f"clone {cid}\nphased AF")
            axes[2*s+1].set_ylim([-0.1, baf_ylim])
            if baf_yticks is None:
                axes[2*s+1].set_yticks(np.arange(0, baf_ylim, 0.1))
            else:
                axes[2*s+1].set_yticks(baf_yticks)
            axes[2*s+1].set_xlim([0, df.shape[0]])
            if remove_xticks:
                axes[2*s+1].set_xticks([])

        for i in unique_chrs:
            median_len = np.percentile(np.where(df.CHR.values == i)[0], 45)
            max_len = np.max(np.where(df.CHR.values == i)[0])
            axes[-1].text(median_len-5, chrtext_shift, i, transform=axes[-1].get_xaxis_transform())
            if max_len + 1 < df.shape[0]:
                for k in range(2*len(final_clone_ids)):
                    axes[k].axvline(x=max_len, c="grey", linewidth=1)
    # plot a given clone
    else:
        fig, axes = plt.subplots(2*len(clone_ids), 1, figsize=(20, base_height*len(clone_ids)), dpi=200, facecolor="white")
        for s,cid in enumerate(clone_ids):
            # major and minor allele copies give the hue
            major = np.maximum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)
            minor = np.minimum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)

            # plot points
            seaborn.scatterplot(x=np.arange(df.shape[0]), y=df[f'clone{cid} RD'].values, \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=0.8, legend=False, ax=axes[2*s])
            axes[2*s].set_ylabel(f"clone {cid}\nRDR" if clone_names is None else f"clone {clone_names[s]}\nRDR")
            axes[2*s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[2*s].set_ylim([0,rdr_ylim])
            axes[2*s].set_xlim([0, df.shape[0]])
            if remove_xticks:
                axes[2*s].set_xticks([])
            seaborn.scatterplot(x=np.arange(df.shape[0]), y=df[f'clone{cid} BAF'].values, \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=0.8, legend=False, ax=axes[2*s+1])
            axes[2*s+1].set_ylabel(f"clone {cid}\nphased AF" if clone_names is None else f"clone {clone_names[s]}\nphased AF")
            axes[2*s+1].set_ylim([-0.1, baf_ylim])
            if baf_yticks is None:
                axes[2*s+1].set_yticks(np.arange(0, baf_ylim, 0.1))
            else:
                axes[2*s+1].set_yticks(baf_yticks)
            axes[2*s+1].set_xlim([0, df.shape[0]])
            if remove_xticks:
                axes[2*s+1].set_xticks([])
        
        for i in unique_chrs:
            median_len = np.percentile(np.where(df.CHR.values == i)[0], 45)
            max_len = np.max(np.where(df.CHR.values == i)[0])
            axes[-1].text(median_len-5, chrtext_shift, i, transform=axes[-1].get_xaxis_transform())
            if max_len + 1 < df.shape[0]:
                for k in range(2*len(clone_ids)):
                    axes[k].axvline(x=max_len, c="grey", linewidth=1)

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, 
        color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        axes[0].legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    return fig, axes


def plot_2dscatter_rdrbaf(configuration_file, r_hmrf_initialization, cn_file, clone_ids=None, rdr_ylim=5, base_width=3.2, pointsize=15):
    # full palette
    palette, ordered_acn = get_full_palette()
    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [palette[c] for c in ordered_acn]

    try:
        config = read_configuration_file(configuration_file)
    except:
        config = read_joint_configuration_file(configuration_file)
    
    # load allele specific integer copy numbers
    df_cnv = pd.read_csv(cn_file, header=0, sep="\t")
    n_final_clones = int(df_cnv.columns[-1].split(" ")[0][5:]) + 1
    assert (clone_ids is None) or np.all([cid <= n_final_clones for cid in clone_ids])
    unique_chrs = np.unique(df_cnv.CHR.values)

    # load data
    outdir = f"{config['output_dir']}/clone{config['n_clones']}_rectangle{r_hmrf_initialization}_w{config['spatial_weight']:.1f}"
    dat = np.load(f"{outdir}/binned_data.npz", allow_pickle=True)
    lengths = dat["lengths"]
    single_X = dat["single_X"]
    single_base_nb_mean = dat["single_base_nb_mean"]
    single_total_bb_RD = dat["single_total_bb_RD"]
    single_tumor_prop = dat["single_tumor_prop"]
    res_combine = dict( np.load(f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", allow_pickle=True) )

    assert single_X.shape[0] == df_cnv.shape[0]

    clone_index = [np.where(res_combine["new_assignment"] == c)[0] for c in range(len( np.unique(res_combine["new_assignment"]) ))]
    if config["tumorprop_file"] is None:
        X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(single_X, single_base_nb_mean, single_total_bb_RD, clone_index)
        tumor_prop = None
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(single_X, single_base_nb_mean, single_total_bb_RD, clone_index, single_tumor_prop)
    n_obs = X.shape[0]

    # plotting all clones
    if clone_ids is None:
        fig, axes = plt.subplots(1, X.shape[2], figsize=(base_width*X.shape[2], base_width), dpi=200, facecolor="white")
        for s in range(X.shape[2]):
            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{s} A"].values, df_cnv[f"clone{s} B"].values)
            minor = np.minimum(df_cnv[f"clone{s} A"].values, df_cnv[f"clone{s} B"].values)

            # plot points
            seaborn.scatterplot(x=X[:,1,s]/total_bb_RD[:,s], y=X[:,0,s]/base_nb_mean[:,s], \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            axes[s].set_xlabel(f"clone {s}\nphased AF")
            axes[s].set_xlim([-0.1, 1.1])
            axes[s].set_xticks([0, 0.5, 1])
            axes[s].set_ylabel(f"clone {s}\nRDR")
            axes[s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[s].set_ylim([0,5])
        fig.tight_layout()
    # plot a given clone
    else:
        fig, axes = plt.subplots(1, len(clone_ids), figsize=(base_width*len(clone_ids), base_width), dpi=200, facecolor="white")
        for s,cid in enumerate(clone_ids):
            # major and minor allele copies give the hue
            major = np.maximum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)
            minor = np.minimum(df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values)

            # plot points
            seaborn.scatterplot(x=X[:,1,cid]/total_bb_RD[:,cid], y=X[:,0,cid]/base_nb_mean[:,cid], \
                hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
                palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", alpha=0.8, legend=False, ax=axes[s])
            axes[s].set_xlabel(f"clone {cid}\nphased AF")
            axes[s].set_xlim([-0.1, 1.1])
            axes[s].set_xticks([0, 0.5, 1])
            axes[s].set_ylabel(f"clone {cid}\nRDR")
            axes[s].set_yticks(np.arange(1, rdr_ylim, 1))
            axes[s].set_ylim([0,5])
        fig.tight_layout()

    return fig


def plot_2dscatter_rdrbaf_from_df(df, axes, cid, cname=None, baf_xlim=0.51, rdr_ylim=3, pointsize=15, linewidth=1, add_legend=False):
    """
    Attributes
    ----------
    df : pandas.DataFrame
        dataframe with columns: clone1 RD, clone1 BAF, clone1 A, clone1 B, ... for each clone
    """
    # full palette
    palette, ordered_acn = get_full_palette()
    map_cn = {x:i for i,x in enumerate(ordered_acn)}
    colors = [palette[c] for c in ordered_acn]

    final_clone_ids = np.unique([ x.split(" ")[0][5:] for x in df.columns if "RD" in x ])
    assert cid in final_clone_ids
    unique_chrs = np.unique(df.CHR.values)

    # major and minor allele copies give the hue
    major = np.maximum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)
    minor = np.minimum(df[f"clone{cid} A"].values, df[f"clone{cid} B"].values)

    # plot points
    seaborn.scatterplot(x=df[f'clone{cid} BAF'].values, y=df[f'clone{cid} RD'].values, \
        hue=pd.Categorical([map_cn[(major[i], minor[i])] for i in range(len(major))], categories=np.arange(len(ordered_acn)), ordered=True), \
        palette=seaborn.color_palette(colors), s=pointsize, edgecolor="black", linewidth=linewidth, alpha=0.8, legend=False, ax=axes)
    axes.set_xlabel(f"clone {cid}\nphased AF" if cname is None else f"{cname}\nphased AF")
    axes.set_xlim([-0.02, baf_xlim])
    axes.set_xticks(np.arange(0, baf_xlim, 0.1))
    axes.set_ylabel(f"clone {cid}\nRDR" if cname is None else f"{cname}\nRDR")
    axes.set_yticks(np.arange(1, rdr_ylim, 1))
    axes.set_ylim([0,rdr_ylim])

    if add_legend:
        a00 = plt.arrow(0,0, 0,0, 
        color='darkblue')
        a10 = plt.arrow(0,0, 0,0, color='lightblue')
        a11 = plt.arrow(0,0, 0,0, color='lightgray')
        a20 = plt.arrow(0,0, 0,0, color='dimgray')
        a21 = plt.arrow(0,0, 0,0, color='lightgoldenrodyellow')
        a30 = plt.arrow(0,0, 0,0, color='gold')
        a22 = plt.arrow(0,0, 0,0, color='navajowhite')
        a31 = plt.arrow(0,0, 0,0, color='orange')
        a40 = plt.arrow(0,0, 0,0, color='darkorange')
        a32 = plt.arrow(0,0, 0,0, color='salmon')
        a41 = plt.arrow(0,0, 0,0, color='red')
        a50 = plt.arrow(0,0, 0,0, color='darkred')
        a33 = plt.arrow(0,0, 0,0, color='plum')
        a42 = plt.arrow(0,0, 0,0, color='orchid')
        a51 = plt.arrow(0,0, 0,0, color='purple')
        a60 = plt.arrow(0,0, 0,0, color='indigo')
        axes.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
        ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
        '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=2, loc='upper left', bbox_to_anchor=(1,1))



def plot_clones_in_space(coords, assignment, sample_list=None, sample_ids=None, palette="Set2", labels=None, label_coords=None, label_sample_ids=None):
    if (sample_list is None) or (len(sample_list) == 1):
        fig, axes = plt.subplots(1, 1, figsize=(5.5,4), dpi=200, facecolor="white")
        seaborn.scatterplot(x=coords[:,0], y=-coords[:,1], color="lightgrey", alpha=0.5, linewidth=0, s=15, ax=axes)
        seaborn.scatterplot(x=coords[~assignment.isnull(),0], y=-coords[~assignment.isnull(),1], \
                            hue=assignment[~assignment.isnull()], palette=palette, linewidth=0, s=15, ax=axes)
        h,l = axes.get_legend_handles_labels()
        axes.legend(h, l, loc="upper left", bbox_to_anchor=(1,1))

        if not labels is None:
            assert len(labels) == len(label_coords)
            for i,c in enumerate(labels):
                axes.text(label_coords[i][0]-4, -label_coords[i][1], c)
    else:
        unique_assignments = np.sort(np.unique(assignment[~assignment.isnull()].values))
        fig, axes = plt.subplots(1, len(sample_list), figsize=(5*len(sample_list)+0.5,4), dpi=200, facecolor="white")
        for s, sname in enumerate(sample_list):
            indexes = np.where(sample_ids == s)[0]
            seaborn.scatterplot(x=coords[indexes,0], y=-coords[indexes,1], color="lightgrey", alpha=0.5, linewidth=0, s=15, ax=axes[s])
            if s + 1 != len(sample_list):
                seaborn.scatterplot(x=coords[indexes,0][~assignment.iloc[indexes].isnull()], y=-coords[indexes,1][~assignment.iloc[indexes].isnull()], \
                                hue=pd.Categorical(assignment.iloc[indexes][~assignment.iloc[indexes].isnull()], categories=unique_assignments, ordered=True), \
                                palette=palette, linewidth=0, s=15, legend=False, ax=axes[s])
            else:
                seaborn.scatterplot(x=coords[indexes,0][~assignment.iloc[indexes].isnull()], y=-coords[indexes,1][~assignment.iloc[indexes].isnull()], \
                                hue=pd.Categorical(assignment.iloc[indexes][~assignment.iloc[indexes].isnull()], categories=unique_assignments, ordered=True), \
                                palette=palette, linewidth=0, s=15, ax=axes[s])
                h,l = axes[s].get_legend_handles_labels()
                axes[s].legend(h, l, loc="upper left", bbox_to_anchor=(1,1))

        if not labels is None:
            assert len(labels) == len(label_coords) and len(labels) == len(label_sample_ids)
            for i,c in enumerate(labels):
                s = label_sample_ids[i]
                axes[s].text(label_coords[i][0]-4, -label_coords[i][1], c)

    fig.tight_layout()

    return fig


def plot_individual_spots_in_space(coords, assignment, single_tumor_prop=None, sample_list=None, sample_ids=None, base_width=4, base_height=3, palette="Set2"):
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
        colorlist = ['lightgrey'] + seaborn.color_palette("Set2", n_final_clones-1).as_hex()
    else:
        colorlist = seaborn.color_palette("Set2", n_final_clones).as_hex()

    for c,cid in enumerate(final_clone_ids):
        idx = np.where( (assignment.values==cid) )[0]
        if single_tumor_prop is None:
            seaborn.scatterplot(x=shifted_coords[idx,0], y=-shifted_coords[idx,1], s=10, color=colorlist[c], linewidth=0, legend=None, ax=axes)
        else:
            # cmap
            this_full_cmap = seaborn.color_palette(f"blend:lightgrey,{colorlist[c]}", as_cmap=True)
            quantile_colors = this_full_cmap(np.array([0, np.min(copy_single_tumor_prop[idx]), np.max(copy_single_tumor_prop[idx]), 1]))
            quantile_colors = [matplotlib.colors.rgb2hex(x) for x in quantile_colors[1:-1]]
            this_cmap = seaborn.color_palette(f"blend:{quantile_colors[0]},{quantile_colors[-1]}", as_cmap=True)
            seaborn.scatterplot(x=shifted_coords[idx,0], y=-shifted_coords[idx,1], s=10, hue=copy_single_tumor_prop[idx], palette=this_cmap, linewidth=0, legend=None, ax=axes)

    legend_elements = [Line2D([0], [0], marker='o', color="w", markerfacecolor=colorlist[c], label=cid, markersize=10) for c,cid in enumerate(final_clone_ids)]
    axes.legend(legend_elements, final_clone_ids, handlelength=0.1, loc="upper left", bbox_to_anchor=(1,1))
    axes.axis("off")

    fig.tight_layout()
    return fig
