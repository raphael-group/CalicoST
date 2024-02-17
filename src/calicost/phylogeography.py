import scanpy as sc
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
import seaborn
from ete3 import Tree
import networkx as nx


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

    return t