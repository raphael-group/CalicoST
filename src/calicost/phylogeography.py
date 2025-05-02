import scanpy as sc
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
import seaborn
from ete3 import Tree
import networkx as nx
from pathlib import Path
from calicost.utils_plotting import *

import argparse


def clone_centers(coords, clone_label, single_tumor_prop=None, sample_list=None, sample_ids=None, tumorprop_threshold=0.6):
    df_centers = []
    for l in np.unique(clone_label):
        # get spot indices of this clone
        index = np.where(clone_label == l)[0] if single_tumor_prop is None else np.where((clone_label == l) & (single_tumor_prop > tumorprop_threshold))[0]
        # if the index contains multiple slices, get the most abundance slice
        if not sample_ids is None:
            most_abundance_slice = pd.Series(sample_ids[index]).mode().values[0]
            index = index[ sample_ids[index] == most_abundance_slice ]
        # get clone cencer
        if single_tumor_prop is None:
            center = np.mean(coords[index], axis=0)
        else:
            center = single_tumor_prop[index].dot(coords[index]) / np.sum(single_tumor_prop[index])
        df_centers.append( pd.DataFrame({'clone':l, 'x':center[0], 'y':center[1]}, index=[0]) )
    df_centers = pd.concat(df_centers, ignore_index=True)
    return df_centers


def project_phylogeneny_space(newick_file, coords, clone_label, single_tumor_prop=None, sample_list=None, sample_ids=None):
    # load tree
    with open(newick_file, 'r') as fp:
        t = Tree(fp.readline())
    
    # get the 
    list_leaf_nodes = []
    list_internal_nodes = []
    rootnode = np.sort( [leaf.name.replace('clone','') for leaf in t.iter_leaves() ] )
    rootnode = "ancestor" + "_".join( rootnode )
    for node in t.traverse():
        leafnames = np.sort( [leaf.name.replace('clone','') for leaf in node.iter_leaves() ] )
        if node.name == "":
            node.name = "ancestor" + "_".join( leafnames )
        
        if node.is_leaf():
            list_leaf_nodes.append(node.name)
        else:
            list_internal_nodes.append(node.name)

    print(f"root node is {rootnode}")
    print(f"a list of leaf nodes: {list_leaf_nodes}")
    print(f"a list of internal nodes: {list_internal_nodes}")
    
    # set up multivariate Gaussian distribution to estimate internal node location
    N_nodes = len(list_leaf_nodes) + len(list_internal_nodes)
    # pairwise distance
    G = nx.Graph()
    G.add_nodes_from( list_leaf_nodes + list_internal_nodes )
    for nodename in list_leaf_nodes:
        node = t&f"{nodename}"
        while not node.is_root():
            p = node.up
            G.add_edge(node.name, p.name, weight=node.dist)
            node = p
    
    G.edges(data=True)
    nx_pdc = dict( nx.all_pairs_dijkstra(G) )

    # covariance matrix based on pairwise distance
    N_nodes = len(list_leaf_nodes) + len(list_internal_nodes)
    Sigma_square = np.zeros((N_nodes, N_nodes))
    base_var = max( np.max(np.abs(coords[:,0])), np.max(np.abs(coords[:,1])) )
    
    for n1, name1 in enumerate(list_leaf_nodes + list_internal_nodes):
        for n2, name2 in enumerate(list_leaf_nodes + list_internal_nodes):
            if n1 == n2:
                Sigma_square[n1, n2] = base_var + nx_pdc[rootnode][0][name1]
            else:
                lca_node = t.get_common_ancestor([name1, name2])
                # print( name1, name2, lca_node.name )
                if lca_node.name == rootnode:
                    Sigma_square[n1, n2] = base_var
                else:
                    Sigma_square[n1, n2] = base_var + nx_pdc[rootnode][0][lca_node.name]

    # mean position
    mu_1 = np.zeros(( len(list_leaf_nodes),2 ))
    mu_2 = np.zeros(( len(list_internal_nodes),2 ))

    # partition covariance matrix
    Sigma_11 = Sigma_square[:len(list_leaf_nodes), :len(list_leaf_nodes)]
    Sigma_12 = Sigma_square[:len(list_leaf_nodes), :][:, len(list_leaf_nodes):]
    Sigma_22 = Sigma_square[len(list_leaf_nodes):, len(list_leaf_nodes):]

    # get leaf node locations
    df_centers = clone_centers(coords, clone_label, single_tumor_prop=single_tumor_prop, 
                               sample_list=sample_list, sample_ids=sample_ids)
    obs_1 = df_centers.set_index('clone').loc[list_leaf_nodes].values

    # conditional expectation internal node position | leaf node position = mu_1
    expected_internal = mu_2 + Sigma_12.T @ (np.linalg.inv(Sigma_11) @ (obs_1 - mu_1))
    df_centers = pd.concat([ df_centers, pd.DataFrame({'clone':list_internal_nodes, 'x':expected_internal[:,0], 'y':expected_internal[:,1]}) ])

    # add to tree features
    for node in t.traverse():
        i = np.where(df_centers.clone.values == node.name)[0][0]
        node.add_features( x=df_centers.x.values[i], y=df_centers.y.values[i] )

    return df_centers, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer spatial coordinates of all nodes in a phylogenetic tree (phylogeography_clone_coordinates.tsv) and plot the phylogeography (phylogeography_plot.pdf). This script only supports a single SRT slice.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tissue_positions_file', type=str, required=True, help='Path to the spatial coordinate file under 10x visium spaceranger directory.')
    parser.add_argument('--newick_file', type=str, required=True, help='Path to the newick file.')
    parser.add_argument('--clone_label_file', type=str, required=True, help='Path to the clone label file output by CalicoST. The file name should be clone_label.tsv.')
    parser.add_argument('--outputdir', type=str, required=True, help='Path to the output directory.')
    args = parser.parse_args()

    Path(args.outputdir).mkdir(parents=True, exist_ok=True)
    
    # read spatial coordinates
    if '_list' in args.tissue_positions_file:
        df_pos = pd.read_csv(args.tissue_positions_file, sep=",", header=None, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    else:
        df_pos = pd.read_csv(args.tissue_positions_file, sep=",", header=0, names=["barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col"])
    df_pos.set_index('barcode', inplace=True)

    # read clone label
    df_clone = pd.read_csv(args.clone_label_file, sep="\t", header=0, index_col=0)
    df_clone.clone_label = 'clone' + df_clone.clone_label.astype(str)
    # reorder df_pos according to df_clone barcodes
    df_pos = df_pos.loc[df_clone.index]
    coords = df_pos[['x', 'y']].values
    single_tumor_prop=None if not 'tumor_proportion' in df_clone.columns else df_clone.tumor_proportion.values

    df_centers, t = project_phylogeneny_space(args.newick_file, coords, df_clone.clone_label.values, single_tumor_prop, sample_list=None, sample_ids=None)
    # output spatial location of observed clones and inferred ancestors to tsv file
    df_centers.to_csv(f"{args.outputdir}/phylogeography_clone_coordinates.tsv", sep="\t", index=True, header=True)

    # plot phylogeography
    fig = plot_individual_spots_in_space(coords, df_clone.clone_label, single_tumor_prop, base_height=3)
    axes = plt.gca()

    # clone centers + ancestors
    for node in t.traverse():
        axes.scatter( node.x, -node.y, marker="D", linewidth=2, edgecolor='black', facecolor="None", s=50)

    # edges
    for node in t.iter_leaves():
        while not node.is_root():
            p = node.up
            if np.abs(node.x - p.x) + np.abs(node.y - p.y) > 1:
                axes.annotate("", xy=(node.x, -node.y), xytext=(p.x, -p.y), arrowprops=dict(mutation_scale=15, lw=1, arrowstyle="->", color="black"))
            node = p
            
    fig.savefig(f"{args.outputdir}/phylogeography_plot.pdf", dpi=300, bbox_inches='tight', transparent=True)
