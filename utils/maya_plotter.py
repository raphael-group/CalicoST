#!/usr/bin/env python3.7

import sys, os
import argparse
import random
import warnings

from itertools import cycle
from collections import defaultdict

import numpy as np
import scipy
import scipy.cluster
import scipy.cluster.hierarchy as hier
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from Utils import *
from Clusterizer import kclustering

from matplotlib.colors import LinearSegmentedColormap

orderchrs = (lambda x : int(''.join([l for l in x if l.isdigit()])))
order = (lambda b : (orderchrs(b[0]), int(b[1]), int(b[2])))


def parse_args():
    description = "Generate plots for the analysis of estimated RDRs and BAFs, inferred allele- and haplotype-specific copy numbers, and clones."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("INPUT", type=str, help="Input file with combined RDR and BAF per bin and per cell")
#    parser.add_argument("-m", "--clonemap", required=True, type=str, default=None, help="Clone map (default: not used, the cells will be clustered for plotting purposes)")
    parser.add_argument("-m", "--clonemap", required=False, type=str, default=None, help="Clone map (default: not used, the cells will be clustered for plotting purposes)")
    parser.add_argument("-f", "--figformat", required=False, type=str, default='png', help="Format of output figures (default: png, the only other option is pdf)")
    parser.add_argument("-s", "--sample", required=False, type=int, default=20, help="Number of cells to sample (default: 20)")
    parser.add_argument("--excludenoisy", required=False, default=False, action='store_true', help="Exclude noisy cells from plots (default: False)")
    parser.add_argument("--gridsize", required=False, type=str, default='12,6', help="Grid dimenstions specified as comma-separated numbers (default: 12,6)")
    parser.add_argument("--plotsize", required=False, type=str, default='5,1.5', help="Plot dimenstions for RDR-BAF plots, specified as comma-separated numbers (default: 5,1.5)")
    parser.add_argument("--clussize", required=False, type=str, default='5,3', help="Grid dimenstions for clustered plots, specified as comma-separated numbers (default: 5,3)")
    parser.add_argument("--xmax", required=False, type=float, default=None, help="Maximum x-axis value (default: None)")
    parser.add_argument("--xmin", required=False, type=float, default=None, help="Minimum x-axis value (default: None)")
    parser.add_argument("--ymax", required=False, type=float, default=None, help="Maximum x-axis value (default: None)")
    parser.add_argument("--ymin", required=False, type=float, default=None, help="Minimum x-axis value (default: None)")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: none)")
    args = parser.parse_args()

    if not os.path.isfile(args.INPUT):
        raise ValueError('ERROR: input file does not exist!')
    if args.clonemap and not os.path.isfile(args.clonemap):
        raise ValueError('ERROR: the provided clone map does not exist!')
    if args.figformat not in ['pdf', 'png']:
        raise ValueError('ERROR: figure format must be either pdf or png!')
    if args.sample < 1:
        raise ValueError('ERROR: number of sampled cells must be positive!')
    if args.seed and args.seed < 0:
        raise ValueError("Random seed must be positive or zero!")
    else:
        np.random.seed(args.seed)

    def get_size(s):
        p = s.split(',')
        if len(p) != 2:
            raise ValueError('ERROR: wrong format for figure sizes!')
        return tuple(map(float, p))

    return {
        'input' : args.INPUT,
        'clonemap' : args.clonemap,
        'format' : args.figformat,
        'sample' : args.sample,
        'nonoisy' : args.excludenoisy,
        'gridsize' : get_size(args.gridsize),
        'plotsize' : get_size(args.plotsize),
        'clussize' : get_size(args.clussize),
        'xmax' : args.xmax,
        'xmin' : args.xmin,
        'ymax' : args.ymax,
        'ymin' : args.ymin
    }


def main():
    log('Parsing and checking arguments')
    args = parse_args()
    log('\n'.join(['Arguments:'] + ['\t{} : {}'.format(a, args[a]) for a in args]), level='INFO')

    log('Reading input')
    bins, pos, cells, iscorr = read_cells(args['input'])
    log('Number of cells: {}'.format(len(cells)), level='INFO')
    log('Number of bins: {}'.format(len(pos)), level='INFO')

    log('Setting style')
    set_style(args)

    if args['clonemap']:
        log('Reading clonemap')
        index, clones, selected = clonemap_to_index(args['clonemap'], cells)
        if all(selected[e] == 'None' for e in selected):
            log('Cell will be re-clustered as no clone has been previously identified', level='WARN')
            index, clones = clustering_tot(bins, pos, cells)
            selected = dict(clones)
    else:
        log('Clustering cells')
        index, clones = clustering_tot(bins, pos, cells)
        selected = dict(clones)

    if args['nonoisy']:
        log('Excluding noisy cells')
        bins, pos, cells, index, clones, selected = exclude_noisy(bins, pos, cells, index, clones, selected)
        log('Number of cells: {}'.format(len(cells)), level='INFO')
        log('Number of bins: {}'.format(len(pos)), level='INFO')

    chosen = random.sample(list(enumerate(cells)), args['sample'])
    chosen = [p[1] for p in sorted(chosen, key=(lambda x : x[0]))]

    log('Plotting RDR and mirrored BAF plots for {} random cells in rbplot_mirrored.{}'.format(args['sample'], args['format']))
    rbplot_mirrored(bins, chosen, args)

    log('Plotting clustered RDR plots for {} random cells in crdr.{}'.format(args['sample'], args['format']))
    crdr(bins, pos, chosen, args)

    log('Plotting clustered-mirrored BAF plots for {} random cells in cbaf.{}'.format(args['sample'], args['format']))
    cbaf(bins, pos, chosen, args)

    log('Plotting read-depth ratios in {}'.format('rdrs.' + args['format']))
    gridrdrs(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    log('Plotting B-allele frequencies in {}'.format('bafs.' + args['format']))
    gridbafs(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)
    
    log('Plotting total copy numbers in {}'.format('totalcn.' + args['format']))
    totalcns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting total copy numbers corrected by clones in {}'.format('totalcn-corrected.' + args['format']))
        totalcns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='totalcn-corrected.', val='CORR-CNS')

    log('Plotting LOH in {}'.format('loh.' + args['format']))
    loh(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting LOH corrected by clones in {}'.format('loh-corrected.' + args['format']))
        loh(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='loh-corrected.', val='CORR-CNS')

    log('Plotting A-specific copy numbers in {}'.format('Aspecificcn.' + args['format']))
    acns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting A-specific copy numbers corrected by clones in {}'.format('Aspecificcn-corrected.' + args['format']))
        acns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='Aspecificcn-corrected.', val='CORR-CNS')

    log('Plotting B-specific copy numbers in {}'.format('Bspecificcn.' + args['format']))
    bcns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting B-specific copy numbers corrected by clones in {}'.format('Bspecificcn-corrected.' + args['format']))
        bcns(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='Bspecificcn-corrected.', val='CORR-CNS')
    
    log('Plotting allele-specific copy numbers in {}'.format('allelecn.' + args['format']))
    states(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting allele-specific copy numbers corrected by clones in {}'.format('allelecn-corrected.' + args['format']))
        states(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='allelecn-corrected.', val='CORR-CNS')
    
    log('Plotting haplotype-specific copy numbers in {}'.format('haplotypecn.' + args['format']))
    minor(bins, pos, cells, index=index, clones=clones, selected=selected, args=args)

    if iscorr:
        log('Plotting haplotype-specific copy numbers corrected by clones in {}'.format('haplotypecn-corrected.' + args['format']))
        minor(bins, pos, cells, index=index, clones=clones, selected=selected, args=args, out='haplotypecn-corrected.', val='CORR-CNS')
    
    log('KTHKBYE!')


def read_cells(f):
    bins = defaultdict(lambda : dict())
    cells = set()
    with open(f, 'r') as i:
        p = i.readline().strip().split()
    if len(p) == 12:    
        form = (lambda p : ((p[0], int(p[1]), int(p[2])), p[3], float(p[6]), float(p[9]), p[10], tuple(map(int, p[11].split('|')))))
        with open(f, 'r') as i:
            for l in i:
                if l[0] != '#' and len(l) > 1:
                    b, e, rdr, baf, c, cns = form(l.strip().split())
                    bins[b][e] = {'RDR' : rdr, 'BAF' : baf, 'Cluster' : c, 'CNS' : cns}
                    cells.add(e)
        pos = sorted(bins.keys(), key=order)
        for x, b in enumerate(pos):
            for e in cells:
                bins[b][e]['Genome'] = x
        return bins, pos, sorted(cells), False
    elif len(p) == 13:
        form = (lambda p : ((p[0], int(p[1]), int(p[2])), p[3], float(p[6]), float(p[9]), p[10], tuple(map(int, p[11].split('|'))), tuple(map(int, p[12].split('|')))))
        with open(f, 'r') as i:
            for l in i:
                if l[0] != '#' and len(l) > 1:
                    b, e, rdr, baf, c, cns, corr = form(l.strip().split())
                    bins[b][e] = {'RDR' : rdr, 'BAF' : baf, 'Cluster' : c, 'CNS' : cns, 'CORR-CNS' : corr}
                    cells.add(e)
        pos = sorted(bins.keys(), key=order)
        for x, b in enumerate(pos):
            for e in cells:
                bins[b][e]['Genome'] = x
        return bins, pos, sorted(cells), True        
    else:
        raise ValueError("Input format is wrong: 12 or 13 fields expected but {} were found".format(len(p)))


def set_style(args):
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.edgecolor"] = "k"
    plt.rcParams["axes.linewidth"]  = 1.5


def clonemap_to_index(f, cells):
    clonemap = {}
    selected = {}
    with open(f, 'r') as i:
        for l in (g for g in i if g[0] != '#' and len(g) > 1):
            p = l.strip().split()
            assert p[0] not in clonemap
            clonemap[p[0]] = int(p[1])
            selected[p[0]] = p[2]
    mapc = [(clonemap[e], e) for e in cells]
    return [p[1] for p in sorted(mapc, key=(lambda x : x[0]))], clonemap, selected


def clustering_tot(bins, pos, cells):
    data = [[bins[b][e]['CNS'][d] for b in pos for d in [0, 1]] for e in cells]
    linkage = hier.linkage(data, method='average', metric='hamming', optimal_ordering=True)
    clus = hier.fcluster(linkage, t=len(cells), criterion='maxclust')
    mapc = [(clus[i], e) for i, e in enumerate(cells)]
    return [p[1] for p in sorted(mapc, key=(lambda x : x[0]))], {e : clus[i] for i, e in enumerate(cells)}


def exclude_noisy(_bins, _pos, _cells, _index, _clones, _selected):
    check = {e : _selected[e] != 'None' for e in _cells}
    cells = [e for e in _cells if check[e]]
    selected = {e : _selected[e] for e in _selected if check[e]}
    clones = {e : _clones[e] for e in _clones if check[e]}
    index = [e for e in _index if check[e]]
    bins = {b : {e : _bins[b][e] for e in _bins[b] if check[e]} for b in _bins}
    pos = sorted(bins.keys(), key=order)
    return bins, pos, cells, index, clones, selected


def rbplot_unphased(bins, chosen, args):
    form = (lambda d, e : {'RDR' : d['RDR'], 'BAF' : d['BAF'], 'Cluster' : d['Cluster'], 'Cell' : e})
    df = [form(bins[b][e], e) for b in bins for e in chosen]

    par= {}
    par['data'] = pd.DataFrame(df)
    par['x'] = 'RDR'
    par['y'] = 'BAF'
    par['hue'] = 'Cluster'
    par['row'] = 'Cell'
    par['fit_reg'] = False
    par['legend'] = False
    par['palette'] = 'tab20'
    par['size'] = args['plotsize'][0]
    par['aspect'] = args['plotsize'][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.lmplot(**par)
        g.despine(top=False, bottom=False, left=False, right=False)
        g.set(ylim=(-0.01, 1.01))
        g.set(xlim=(args['xmin'], args['xmax']))
    plt.savefig('rbplot_unphased.{}'.format(args['format']), bbox_inches='tight')
    plt.close()


def rbplot_mirrored(bins, chosen, args):
    form = (lambda d, e : {'RDR' : d['RDR'], '|0.5 - BAF|' : 0.5-min(d['BAF'], 1-d['BAF']), 'Cluster' : d['Cluster'], 'Cell' : e})
    df = [form(bins[b][e], e) for b in bins for e in chosen]

    par= {}
    par['data'] = pd.DataFrame(df)
    par['x'] = 'RDR'
    par['y'] = '|0.5 - BAF|'
    par['hue'] = 'Cluster'
    par['row'] = 'Cell'
    par['fit_reg'] = False
    par['legend'] = False
    par['palette'] = 'tab20'
    par['size'] = args['plotsize'][0]
    par['aspect'] = args['plotsize'][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.lmplot(**par)
        g.despine(top=False, bottom=False, left=False, right=False)
        g.set(ylim=(-0.01, 0.51))
        g.set(xlim=(args['xmin'], args['xmax']))
    plt.savefig('rbplot_mirrored.{}'.format(args['format']), bbox_inches='tight')
    plt.close()


def crdr(bins, pos, chosen, args):
    form = (lambda d, e : {'Genome' : d['Genome'], 'RDR' : d['RDR'], 'Cluster' : d['Cluster'], 'Cell' : e})
    df = [form(bins[b][e], e) for b in bins for e in chosen]

    par= {}
    par['data'] = pd.DataFrame(df)
    par['x'] = 'Genome'
    par['y'] = 'RDR'
    par['hue'] = 'Cluster'
    par['row'] = 'Cell'
    par['fit_reg'] = False
    par['legend'] = False
    par['palette'] = 'tab20'
    par['size'] = args['clussize'][0]
    par['aspect'] = args['clussize'][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.lmplot(**par)
        g.despine(top=False, bottom=False, left=False, right=False)
        for ax in g.axes:
            for x, p in enumerate(pos):
                if x > 0 and pos[x-1][0] != pos[x][0]:
                    ax[0].plot((x, x), (0, 2), '--b', linewidth=1.0)
            ax[0].margins(x=0, y=0)
        addchrplt(pos)
        g.set(xlim=(0, len(pos)))
        g.set(xlim=(args['ymin'], args['ymax']))
    plt.savefig('crdr.{}'.format(args['format']), bbox_inches='tight')
    plt.close()


def cbaf(bins, pos, chosen, args):
    form = (lambda d, e : {'Genome' : d['Genome'], '|0.5 - BAF|' : 0.5-min(d['BAF'], 1-d['BAF']), 'Cluster' : d['Cluster'], 'Cell' : e})
    df = [form(bins[b][e], e) for b in bins for e in chosen]

    par= {}
    par['data'] = pd.DataFrame(df)
    par['x'] = 'Genome'
    par['y'] = '|0.5 - BAF|'
    par['hue'] = 'Cluster'
    par['row'] = 'Cell'
    par['fit_reg'] = False
    par['legend'] = False
    par['palette'] = 'tab20'
    par['size'] = args['clussize'][0]
    par['aspect'] = args['clussize'][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.lmplot(**par)
        g.despine(top=False, bottom=False, left=False, right=False)
        for ax in g.axes:
            for x, p in enumerate(pos):
                if x > 0 and pos[x-1][0] != pos[x][0]:
                    ax[0].plot((x, x), (0, 0.5), '--b', linewidth=1.0)
            ax[0].margins(x=0, y=0)
        addchrplt(pos)
        g.set(xlim=(0, len(pos)))
        g.set(xlim=(args['ymin'], args['ymax']))
    plt.savefig('cbaf.'.format(args['format']), bbox_inches='tight')
    plt.close()


def gridrdrs(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='rdrs.', val='RDR'):
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'RDR' : min(2, bins[b][e][val])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='RDR', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'Read-depth ratios'
    draw(table, bins, pos, cells, index, mapc, palette='coolwarm', center=1, method='single', metric='hamming', title=title, out=out, args=args)


def gridbafs(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='bafs.', val='BAF'):
    df = []
    mapc = {}
    mirror = (lambda v : min(v, 1 - v))
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Mirrored BAF' : mirror(bins[b][e][val])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='Mirrored BAF', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'Mirrored B-allele frequencies'
    draw(table, bins, pos, cells, index, mapc, palette='YlGnBu_r', center=None, method='single', metric='hamming', title=title, out=out, args=args)


def totalcns(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='totalcn.', val='CNS'):
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Total CN' : min(6, sum(bins[b][e][val]))} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='Total CN', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'Total copy numbers'
    palette = {}
    palette.update({0 : 'darkblue'})
    palette.update({1 : 'lightblue'})
    palette.update({2 : 'lightgray'})
    palette.update({3 : 'lightgoldenrodyellow'})
    palette.update({4 : 'navajowhite'})
    palette.update({5 : 'red'})
    palette.update({6 : 'orchid'})
    colors = [palette[x] for x in range(7) if x in set(df['Total CN'])]
    cmap = LinearSegmentedColormap.from_list('multi-level', colors, len(colors))
    draw(table, bins, pos, cells, index, mapc, palette=cmap, center=None, method='single', metric='hamming', title=title, out=out, args=args)


def loh(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='loh.', val='CNS'):
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'LOH' : 1 if 0 in bins[b][e][val] else 0} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='LOH', columns=['Genome'], index=['Cell'], aggfunc='first')
    myColors = sns.cubehelix_palette(2, start=2, rot=0, dark=0, light=.95)
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    title = 'Loss of heterozigosity (LOH)'
    draw(table, bins, pos, cells, index, mapc, palette=cmap, center=None, method='median', metric='cityblock', title=title, out=out, args=args)


def acns(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='Aspecificcn.', val='CNS'):
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'A-specific CN' : min(8, bins[b][e][val][0])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='A-specific CN', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'A-specific copy numbers'
    draw(table, bins, pos, cells, index, mapc, palette='coolwarm', center=2, method='single', metric='hamming', title=title, out=out, args=args)


def bcns(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='Bspecificcn.', val='CNS'):
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'B-specific CN' : min(8, bins[b][e][val][1])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='B-specific CN', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'B-specific copy numbers'
    draw(table, bins, pos, cells, index, mapc, palette='coolwarm', center=2, method='single', metric='hamming', title=title, out=out, args=args)


def states(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='allelecn.', val='CNS'):
    avail = [(t - i, i) for t in range(7) for i in reversed(range(t+1)) if i <= t - i]
    order = (lambda p : (max(p), min(p)))
    convert = (lambda p : order(p) if sum(p) <= 6 else min(avail, key=(lambda x : abs(p[0] - x[0]) + abs(p[1] - x[1]))))
    df = []
    mapc = {}
    found = set()
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Value' : convert(bins[b][e][val])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    found = [v for v in avail if v in set(df['Value'])]
    smap = {v : x for x, v in enumerate(found)}
    df['CN states'] = df.apply(lambda r : smap[r['Value']], axis=1)
    table = pd.pivot_table(df, values='CN states', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'Copy-number states'
    #found = set(df['CN states'] for i, r in df.iterrows())
    palette = {}
    palette.update({(0, 0) : 'darkblue'})
    palette.update({(1, 0) : 'lightblue'})
    palette.update({(1, 1) : 'lightgray', (2, 0) : 'dimgray'})
    palette.update({(2, 1) : 'lightgoldenrodyellow', (3, 0) : 'gold'})
    palette.update({(2, 2) : 'navajowhite', (3, 1) : 'orange', (4, 0) : 'darkorange'})
    palette.update({(3, 2) : 'salmon', (4, 1) : 'red', (5, 0) : 'darkred'})
    palette.update({(3, 3) : 'plum', (4, 2) : 'orchid', (5, 1) : 'purple', (6, 0) : 'indigo'})
    colors = [palette[c] for c in found]
    cmap = LinearSegmentedColormap.from_list('multi-level', colors, len(colors))
    draw(table, bins, pos, cells, index, mapc, palette=cmap, center=None, method='single', metric='cityblock', title=title, out=out, args=args)


def minor(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='haplotypecn.', val='CNS'):
    get_minor = (lambda s : (3 - min(2, min(s))) * (0 if s[0] == s[1] else (-1 if s[0] < s[1] else 1)))
    df = []
    mapc = {}
    for x, e in enumerate(index):
        df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Minor allele' : get_minor(bins[b][e][val])} for b in pos])
        mapc[x] = (clones[e], selected[e])
    df = pd.DataFrame(df)
    table = pd.pivot_table(df, values='Minor allele', columns=['Genome'], index=['Cell'], aggfunc='first')
    title = 'Minor alleles'
    colormap = 'PiYG'
    draw(table, bins, pos, cells, index, mapc, palette=colormap, center=0, method='single', metric='hamming', title=title, out=out, args=args)


def draw(table, bins, pos, cells, index, clones, palette, center, method, metric, title, out, args):
    chr_palette = cycle(['#525252', '#969696', '#cccccc'])
    chr_colors = {c : next(chr_palette) for c in sorted(set(b[0] for b in bins), key=orderchrs)}
    seen = set()
    seen_add = seen.add
    ordclones = [clones[x] for x in table.index if not (clones[x][0] in seen or seen_add(clones[x][0]))]
    cell_palette = cycle(sns.color_palette("muted", len(set(ordclones))))
    disc_palette = cycle(sns.color_palette("Greys", 8))
    clone_colors = {i[0] : next(cell_palette) if i[1] != 'None' else '#f0f0f0' for i in ordclones}
    cell_colors = {x : clone_colors[clones[x][0]] for x in table.index}

    para = {}
    para['data'] = table
    para['cmap'] = palette
    if center:
        para['center'] = center
    para['yticklabels'] = False
    para['row_cluster'] = False
    para['xticklabels'] = False
    para['col_cluster'] = False
    para['figsize'] = args['gridsize']
    para['rasterized'] = True
    para['col_colors'] = pd.DataFrame([{'index' : s, 'chromosomes' : chr_colors[pos[x][0]]} for x, s in enumerate(table.columns)]).set_index('index')
    para['row_colors'] = pd.DataFrame([{'index' : x, 'Clone' : cell_colors[x]} for x in table.index]).set_index('index')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.clustermap(**para)
        addchr(g, pos)
        g.fig.suptitle(title)
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
    plt.legend([a00, a10, a11, a20, a21, a30, a22, a31, a40, a32, a41, a50, a33, a42, a51, a60], \
    ['(0, 0)','(1, 0)','(1, 1)','(2, 0)', '(2, 1)','(3, 0)', '(2, 2)','(3, 1)','(4, 0)','(3, 2)', \
    '(4, 1)','(5, 0)', '(3, 3)','(4, 2)','(5, 1)','(6, 0)'], ncol=4, loc='upper left')
    plt.savefig(out + args['format'], bbox_inches='tight', dpi=600)
    plt.close()


def addchr(g, pos, color=None):
    corners = []
    prev = 0
    for x, b in enumerate(pos):
        if x != 0 and pos[x-1][0] != pos[x][0]:
            corners.append((prev, x))
            prev = x
    corners.append((prev, x))
    ax = g.ax_heatmap
    ticks = []
    for o in corners:
        ax.set_xticks(np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        ticks.append(pos[o[0]][0])
    ax.set_xticklabels(ticks, rotation=45, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def addchrplt(pos):
    corners = []
    prev = 0
    val = pos[0][0]
    for x, s in enumerate(pos):
        if x != 0 and pos[x-1][0] != pos[x][0]:
            corners.append((prev, x, val))
            prev = x
            val = s[0]
    corners.append((prev, x, val))
    ticks = [(int(float(o[1] + o[0] + 1) / 2.0), o[2]) for o in corners]
    plt.xticks([x[0] for x in ticks], [x[1] for x in ticks], rotation=45, ha='center')
    plt.yticks(rotation=0)

def generate_discrete_legend(levels, colors=None, ax=None):
    """"""
    # levels = [(2, 2), (3, 1)], [(2, 1)], [(1, 1), (2, 2)]
    # levels = [[(1, 2), (1, 1)], [(0, 2)], [(1, 3), (1, 0)], [(1, 1)]]
    if ax is None:
        ax = plt.gca()

    if colors is None:
        n_colors = len(levels)
        cmap = plt.cm.tab20
        colors = [(cmap(idx * 2), cmap(idx * 2 + 1)) for idx in range(n_colors)]

    legend_elements = []
    for level, color in zip(levels, colors):
        sub_level = level[0]
        sub_label = "{" + str(sub_level[0]) + ", " + str(sub_level[1]) + "}"
        if len(level) > 1:
            patch_1 = Patch(facecolor=color[1], edgecolor='k', label=sub_label)
            sub_level_2 = level[1]
            sub_label_2 = "{" + str(sub_level_2[0]) + ", " + str(sub_level_2[1]) + "}"
            patch_2 = Patch(facecolor=color[0], edgecolor='k', label=sub_label_2)
        else:
            # Moves this patch one space over.
            patch_1 = Line2D([0], [0], color="w", label="")
            patch_2 = Patch(facecolor=color[0], edgecolor='k', label=sub_label)
        legend_elements.append([patch_1, patch_2])

    legend_elements_ordered = [e[0] for e in legend_elements] + [e[1] for e in legend_elements]
    #
    fig, ax = plt.subplots()
    legend = ax.legend(handles=legend_elements_ordered, ncol=2, loc='upper right', frameon=False,
                       columnspacing=-3.9, labelspacing=1.5, handlelength=3.0, handleheight=1.1, borderpad=0.9,
                       bbox_to_anchor=(1.13, 1.0))
    for txt in legend.get_texts():
        txt.set_ha("center")
        txt.set_x(-52)
        txt.set_y(17)


if __name__ == '__main__':
    main()
