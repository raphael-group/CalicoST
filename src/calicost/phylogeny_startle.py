import sys
import pandas as pd
import argparse
import itertools
import math
import subprocess
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import networkx as nx
import itertools
from collections import deque
import argparse


def get_LoH_for_phylogeny(df_seglevel_cnv, min_segments):
    """
    Treating LoH as irreversible point mutations, output a clone-by-mutation matrix for phylogeny reconstruction.
    Mutation states: 0 for no LoH, 1 for lossing A allele, 2 for lossing B allele.

    Attributes
    ----------
    df_seglevel_cnv : pd.DataFrame, (n_obs, 3+2*n_clones)
        Dataframe from cnv_*seglevel.tsv output.

    Returns
    ----------
    df_loh : pd.DataFrame, (n_clones, n_segments)
    """
    def get_shared_intervals(acn_profile):
        '''
        Takes in allele-specific copy numbers, output a segmentation of genome such that all clones are in the same CN state within each segment.

        anc_profile : array, (n_obs, 2*n_clones)
            Allele-specific integer copy numbers for each genomic bin (obs) across all clones.
        '''
        intervals = []
        seg_acn = []
        s = 0
        while s < acn_profile.shape[0]:
            t = np.where( ~np.all(acn_profile[s:,] == acn_profile[s,:], axis=1) )[0]
            if len(t) == 0:
                intervals.append( (s, acn_profile.shape[0])  )
                seg_acn.append( acn_profile[s,:] )
                s = acn_profile.shape[0]
            else:
                t = t[0]
                intervals.append( (s,s+t) )
                seg_acn.append( acn_profile[s,:] )
                s = s+t
        return intervals, seg_acn
    
    clone_ids = [x.split(" ")[0] for x in df_seglevel_cnv.columns[ np.arange(3, df_seglevel_cnv.shape[1], 2) ] ]
    
    acn_profile = df_seglevel_cnv.iloc[:,3:].values
    intervals, seg_acn = get_shared_intervals(acn_profile)
    df_loh = []
    for i, acn in enumerate(seg_acn):
        if np.all(acn != 0):
            continue
        if intervals[i][1] - intervals[i][0] < min_segments:
            continue
        idx_zero = np.where(acn == 0)[0]
        idx_clones = (idx_zero / 2).astype(int)
        is_A = (idx_zero % 2 == 0)
        # vector of mutation states
        mut = np.zeros( int(len(acn) / 2), dtype=int )
        mut[idx_clones] = np.where(is_A, 1, 2)
        df_loh.append( pd.DataFrame(mut.reshape(1, -1), index=[f"bin_{intervals[i][0]}_{intervals[i][1]}"], columns=clone_ids) )

    df_loh = pd.concat(df_loh).T
    return df_loh


def get_binary_matrix(df_character_matrix):
    
    ncells = len(df_character_matrix)
    binary_col_dict = {}
    for column in df_character_matrix.columns:
        state_list = list(df_character_matrix[column].unique())
        for s in state_list:
            if s != -1 and s != 0:
                state_col = np.zeros((ncells))
                state_col[df_character_matrix[column] == s] = 1
                state_col[df_character_matrix[column] == -1] = -1

                binary_col_dict[f'{column}_{s}'] = state_col

    df_binary = pd.DataFrame(binary_col_dict, index = df_character_matrix.index, dtype=int)
    return df_binary


def generate_perfect_phylogeny(df_binary):

    solT_mut = nx.DiGraph()
    solT_mut.add_node('root')

    solT_cell = nx.DiGraph()
    solT_cell.add_node('root')

    df_binary = df_binary[df_binary.sum().sort_values(ascending=False).index]    

    for cell_id, row in df_binary.iterrows():
        if cell_id == 'root':
            continue

        curr_node = 'root'
        for column in df_binary.columns[row.values == 1]:
            if column in solT_mut[curr_node]:
                curr_node = column
            else:
                if column in solT_mut.nodes:
                    raise NameError(f'{column} is being repeated')
                solT_mut.add_edge(curr_node, column)
                solT_cell.add_edge(curr_node, column)
                curr_node = column

        solT_cell.add_edge(curr_node, cell_id)   

    return solT_mut, solT_cell


def tree_to_newick(T, root=None):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, T.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    while len(T[root]) == 1:
        root = list(T[root])[0]
    for child in T[root]:
        pathlen = 0
        while len(T[child]) == 1:
            child = list(T[child])[0]
            pathlen += 1
        if len(T[child]) > 0:
            pathlen += 1
            subgs.append(tree_to_newick(T, root=child) + f":{pathlen}")
        else:
            subgs.append( f"{child}:{pathlen}" )
    return "(" + ','.join(map(str, subgs)) + ")"


def output_startle_input_files(calicostdir, outdir, midfix="", startle_bin="startle", min_segments=3):
    # get LoH data frame
    # rows are clones, columns are bins, entries are 0 (no LoH) or 1 (A allele LoH) of 2 (B allele LoH)
    df_seglevel_cnv = pd.read_csv(f"{calicostdir}/cnv{midfix}_seglevel.tsv", header=0, sep="\t")
    df_loh = get_LoH_for_phylogeny(df_seglevel_cnv, min_segments)
    df_loh.to_csv(f"{outdir}/loh_matrix.tsv", header=True, index=True, sep="\t")
    
    # binarize
    df_binary = get_binary_matrix(df_loh)

    cell_list = list(df_binary.index)
    mutation_list = list(df_binary.columns)
    mutation_to_index = {x: idx for idx, x in enumerate(mutation_list)}

    # one and missing indices
    # one indices
    one_cell_mut_list = []
    for cell_idx, cell in enumerate(cell_list):
        for mut_idx, mut in enumerate(mutation_list):
            if df_binary.loc[cell][mut] == 1:
                one_cell_mut_list.append((cell_idx, mut_idx))
    with open(f'{outdir}/loh_one_indices.txt', 'w') as out:
        for cell_idx, mut_idx in one_cell_mut_list:
            out.write(f'{cell_idx} {mut_idx}\n')
    # missimg imdices
    character_list = list(set(['_'.join(x.split('_')[:-1]) for x in df_binary.columns]))
    missing_cell_character_list = []
    for character_idx, character in enumerate(character_list):
        for cell_idx, cell in enumerate(cell_list):
            if df_loh.loc[cell][character] == -1:
                missing_cell_character_list.append((cell_idx, character_idx))
    with open(f'{outdir}/loh_missing_indices.txt', 'w') as out:
        for cell_idx, character_idx in missing_cell_character_list:
            out.write(f'{cell_idx} {character_idx}\n')

    # character mutation mapping
    with open(f'{outdir}/loh_character_mutation_mapping.txt', 'w') as out:
        for _, character in enumerate(character_list):
            character_mutation_list = [mutation_to_index[x] for x in mutation_list if x.startswith(f'{character}_')]
            out.write(' '.join(map(str, character_mutation_list)) + '\n')

    # count of character states of mutations
    max_allowed_homoplasy = {}
    for mutation in mutation_list:
        max_allowed_homoplasy[mutation] = 2
    with open(f'{outdir}/loh_counts.txt', 'w') as out:
        for mutation in mutation_list:
            out.write(f'{max_allowed_homoplasy[mutation]}\n')
    
    # weights
    with open(f'{outdir}/loh_weights.txt', 'w') as out:
        for mutation in mutation_list:
            out.write(f"1\n")

    ##### run startle #####
    m_mutations = df_binary.shape[1]
    n_clones = df_binary.shape[0]
    command = f"{startle_bin} -m {m_mutations} -n {n_clones} {outdir}/loh_one_indices.txt {outdir}/loh_missing_indices.txt {outdir}/loh_counts.txt {outdir}/loh_character_mutation_mapping.txt {outdir}/loh_weights.txt {outdir}/loh_cpp_output.txt"
    print( command )
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out,err = p.communicate()

    # parse output
    df_cpp_output = pd.read_csv(f'{outdir}/loh_cpp_output.txt', header=None, sep=' ')
    df_cpp_output = df_cpp_output.rename(columns={0:'cell_idx', 1:'mut_idx', 2:'state_idx', 3:'entry'})
    df_cpp_output['name'] = df_cpp_output.apply(lambda x: f"{mutation_list[x['mut_idx']]}_{x['state_idx']}", axis =1)
    
    sol_columns = list(df_cpp_output['name'].unique())
    nsol_columns = len(sol_columns)
    sol_entries = np.zeros((n_clones, nsol_columns), dtype=int)
    for mut_idx, mut in enumerate(sol_columns):
        for cell_idx in df_cpp_output[(df_cpp_output['entry'] == 1) & (df_cpp_output['name'] == mut)]['cell_idx']:
            sol_entries[cell_idx][mut_idx] = 1
    df_sol_binary = pd.DataFrame(sol_entries, columns=sol_columns, index=cell_list)

    solT_mut, solT_cell = generate_perfect_phylogeny(df_sol_binary)
    with open(f'{outdir}/loh_tree.newick', 'w') as out:
        out.write(f"{tree_to_newick(solT_cell)};")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calicost_dir", help="Directory of a specific random initialization of CalicoST", type=str)
    parser.add_argument("-s", "--startle_bin", help="The startle executable path", default="startle", type=str)
    parser.add_argument("-p", "--ploidy", help="Ploidy of allele-specific integer copy numbers.", default="", type=str)
    parser.add_argument("--min_segments", help="Minimum number of genome segment to keep an LOH event in phylogenetic tree reconstruction.", default=3, type=int)
    parser.add_argument("-o", "--outputdir", help="output directory", type=str)
    args = parser.parse_args()

    output_startle_input_files(args.calicost_dir, args.outputdir, midfix=args.ploidy, startle_bin=args.startle_bin, min_segments=args.min_segments)