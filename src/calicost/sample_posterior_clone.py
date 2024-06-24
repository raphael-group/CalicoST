import sys
import copy
import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import seaborn

from calicost.utils_hmm import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.utils_phase_switch import *


def compute_potential(single_llf, labels, spatial_weight, adjacency_mat):
    """
    Attributes
    ----------
    single_llf : np.array, size (num_spots, num_labels)
        The log-likelihood of each pixel being in each label.
    labels : np.array, size (num_spots)
        The label of each pixel.
    spatial_weight : float
        The weight of the spatial coherence term.
    adjacency_mat : np.array, size (num_spots, num_spots)
        The adjacency matrix of the spatial graph. Note this is a symmetric matrix with 0 diagonal.
    """
    N = len(labels)
    # emission part
    neg_potential = single_llf[(np.arange(N), labels)].sum()
    # spatial adjacency part
    # make sure diagonal is 0
    if not np.all(adjacency_mat.diagonal() == 0):
        adjacency_mat.setdiag(0)
    # neg_potential += 0.5 * spatial_weight * np.sum(adjacency_mat * (labels[:, None] == labels[None, :]))
    idx1, idx2 = adjacency_mat.nonzero()
    neg_potential += spatial_weight * np.sum(labels[idx1] == labels[idx2])
    return neg_potential / N


def block_gibbs_sampling_labels(emission_llf, log_clonesize, adjacency_mat, block_ids, spatial_weight, num_draws, random_state, temperature, initial_logprob=None):
    """
    Attributes
    ----------
    emission_llf : np.array, size (num_spots, n_clones)
        The log-likelihood of each pixel being in each clone label.
    log_clonesize : np.array, size (num_spots, n_clones)
        The prior probability of selecting each clone label for each spot.
    adjacency_mat : np.array, size (num_spots, num_spots)
        The adjacency matrix of the spatial graph. Note this is a symmetric matrix with 0 diagonal.
    block_ids : np.array, size (num_spots)
        The block id of each spot. Each block is updated together in a block Gibbs sampling step.
    spatial_weight : float
        The weight of the spatial coherence term.
    num_draws : int
        The number of draws to draw.
    random_state : int
        The random seed.
    initial_logprob : np.array, size (num_spots, n_clones)
        Initial log probability to sample clone labels.
    """
    from calicost.utils_hmm import multinoulli_sampler_high_dimensional
    np.random.seed(random_state)
    # number of spots, blocks, and clones
    N, n_clones = emission_llf.shape
    n_blocks = len(np.unique(block_ids))

    # list of labels of each sampling time point
    # with initial label by random sampling
    if not initial_logprob is None:
        list_labels = [ multinoulli_sampler_high_dimensional(initial_logprob) ]
    else:
        list_labels = [ multinoulli_sampler_high_dimensional(log_clonesize) ]
    list_potentials = [ compute_potential(emission_llf + log_clonesize, list_labels[-1], spatial_weight, adjacency_mat) ]
    
    # block Gibbs sampling
    # prepare sub-adjacency matrix where rows correspond to one block, columns correspond to the other
    A_sub_list = [adjacency_mat[block_ids==b, :][:, block_ids!=b] for b in range(n_blocks)]
    for iter in range(num_draws):
        this_l = np.zeros(N, dtype=int)
        for b in range(n_blocks):
            # for spots with block_id of b, compute the posterior distribution
            post = copy.copy((emission_llf + log_clonesize)[block_ids==b, :])
            for c in range(n_clones):
                post[:,c] += spatial_weight * A_sub_list[b] @ (list_labels[-1][block_ids!=b] == c)

            post -= scipy.special.logsumexp(post, axis=1, keepdims=True)
            post = post / temperature
            this_l[block_ids==b] = multinoulli_sampler_high_dimensional(post)
        
        list_labels.append(this_l)
        # potential of the new state
        list_potentials.append( compute_potential(emission_llf + log_clonesize, list_labels[-1], spatial_weight, adjacency_mat) )
    
    return list_labels, list_potentials


def posterior_distribution_clone_labels(emission_llf, log_clonesize, adjacency_mat, coords, spatial_weight, num_chains, burnin, platform='visium', fun=block_gibbs_sampling_labels, temperature=1.0):
    n_spots, n_clones = emission_llf.shape
    list_labels = []
    list_potentials = []

    # if block Gibbs sampling
    if platform == 'visium':
        block_ids = coords[:,1]%3
    else:
        block_ids = (coords[:,0] + coords[:,1]) % 2 

    # number of chains (starting points)
    for r in range(num_chains):
        this_list_labels, this_list_potential = fun(emission_llf, log_clonesize, adjacency_mat, block_ids, spatial_weight, num_draws=150, random_state=r, temperature=temperature)
        list_labels.append(this_list_labels)
        list_potentials.append(this_list_potential)

    # stack lists
    list_labels = np.stack(list_labels)
    list_potentials = np.stack(list_potentials)

    # make a plot of the potential of the first chain
    for r in range(num_chains):
        plt.plot(list_potentials[r], alpha=0.5)
    plt.xlabel('iteration')
    plt.ylabel('potential')
    plt.show()

    # remove burnin
    list_labels = list_labels[:,burnin:, :]
    list_potentials = list_potentials[:, burnin:]

    # probability of getting each clone at each spot
    clone_prob = []
    for r in range(num_chains):
        this_clone_prob = np.array([ np.bincount(l, minlength=n_clones) for l in list_labels[r].T ])
        this_clone_prob = this_clone_prob / list_labels[r].shape[0]
        clone_prob.append( this_clone_prob )

    # aggregate across chains to get the final posterior distribution per spot
    agg_clone_prob = np.mean(clone_prob, axis=0)
    return agg_clone_prob, list_labels.reshape((-1, n_spots)), list_potentials.flatten()


def infer_all(single_X, lengths, single_base_nb_mean, single_total_bb_RD, single_tumor_prop, initial_clone_label, n_states,
    coords, adjacency_mat, tumorprop_threshold, spatial_weight, platform, max_iter_outer, num_chains, burnin, sampling_tol, temperature,
    hmmclass, hmm_params, hmm_t, hmm_random_state, hmm_max_iter, hmm_tol, hmm_num_draws,
    smooth_mat=None, sample_ids=None, init_log_mu=None, init_p_binom=None, init_alphas=None, init_taus=None,
    fix_NB_dispersion=False, shared_NB_dispersion=True, fix_BB_dispersion=False, shared_BB_dispersion=True):

    n_clones = len(np.unique(initial_clone_label))
    n_obs, n_spots = single_total_bb_RD.shape

    # aggregated counts according to smooth_mat
    if not smooth_mat is None:
        agg_single_X = np.stack([ single_X[i,:,:] @ smooth_mat for i in range(single_X.shape[0]) ])
        agg_single_base_nb_mean = single_base_nb_mean @ smooth_mat
        agg_single_total_bb_RD = single_total_bb_RD @ smooth_mat
        agg_single_tumor_prop = (single_tumor_prop @ smooth_mat) / np.sum(smooth_mat.A, axis=0)
    else:
        # aggregated matrices are the same as their original ones
        agg_single_X = single_X
        agg_single_base_nb_mean = single_base_nb_mean
        agg_single_total_bb_RD = single_total_bb_RD
        agg_single_tumor_prop = single_tumor_prop

    # make a fake log_sitewise_transmat because some function calls have that parameter for formating purpose, but it's not actually used
    log_sitewise_transmat = np.zeros(n_obs)

    # the initial posterior distribution of clone labels is a binary matrix derived from initial_clone_label
    posterior_clones = np.zeros((n_spots, n_clones))
    posterior_clones[ (np.arange(n_spots), initial_clone_label) ] = 1
    log_clonesize = np.ones(n_clones) * np.log(1.0 / n_clones)
    log_clonesize = np.repeat( log_clonesize[None,:], single_X.shape[2], axis=0 )

    # initial emission parameters
    last_log_mu = init_log_mu
    last_alphas = init_alphas
    last_p_binom = init_p_binom
    last_taus = init_taus

    list_posterior_clones = [posterior_clones]
    list_cna_states = []
    list_log_mu = []
    list_p_binom = []
    list_elbo = []

    for r in range(max_iter_outer):
        ##### Fit HMM using posterior_clones #####
        # aggregate into pseudobulk for each clone weighted by posterior_clones
        X = (single_X[:,:, single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :])
        base_nb_mean = single_base_nb_mean[:, single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :]
        total_bb_RD = single_total_bb_RD[:, single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :]
        tumor_prop = single_tumor_prop[single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :] / posterior_clones[single_tumor_prop>tumorprop_threshold, :].sum(axis=0)
        
        # initialize parameters
        if (last_log_mu is None and "m" in hmm_params) or (last_p_binom is None and "p" in hmm_params):
            tmp_log_mu, tmp_p_binom = initialization_by_gmm(n_states, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), \
                base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1), hmm_params, random_state=hmm_random_state, in_log_space=False, only_minor=False)
            
            last_log_mu = tmp_log_mu if init_log_mu is None and "m" in hmm_params else None
            last_p_binom = tmp_p_binom if init_p_binom is None and "p" in hmm_params else None

        # fit HMM
        res = pipeline_baum_welch(None, np.vstack([X[:,0,:].flatten("F"), X[:,1,:].flatten("F")]).T.reshape(-1,2,1), np.tile(lengths, X.shape[2]), n_states, \
                        base_nb_mean.flatten("F").reshape(-1,1), total_bb_RD.flatten("F").reshape(-1,1),  np.tile(log_sitewise_transmat, X.shape[2]), np.repeat(tumor_prop, X.shape[0]).reshape(-1,1), \
                        hmmclass=hmmclass, params=hmm_params, t=hmm_t, random_state=hmm_random_state, \
                        fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion, fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion, \
                        is_diag=True, init_log_mu=last_log_mu, init_p_binom=last_p_binom, init_alphas=last_alphas, init_taus=last_taus, max_iter=hmm_max_iter, tol=hmm_tol)
        # estimate log emission probability for each spot belong to each clone based on the fitted HMM res
        list_h = FFBS_faster(hmm_num_draws, res['log_alpha'], res['log_gamma'], res['new_log_transmat'], np.tile(lengths, X.shape[2]))
        # emission probability of each spot under the posterior distribution of hidden states of each clone
        log_prob_rdr, log_prob_baf = hmmclass.compute_emission_probability_nb_betabinom_mix(np.vstack([agg_single_X[:,0,:].flatten("F"), agg_single_X[:,1,:].flatten("F")]).T.reshape(-1,2,1), 
                                                                                                    agg_single_base_nb_mean.flatten("F").reshape(-1,1), res['new_log_mu'], res['new_alphas'], 
                                                                                                    agg_single_total_bb_RD.flatten("F").reshape(-1,1), res['new_p_binom'], res['new_taus'], 
                                                                                                    np.repeat(agg_single_tumor_prop, agg_single_X.shape[0]).reshape(-1,1))
        log_per_state_emission = log_prob_rdr + log_prob_baf

        emission_llf = np.zeros((agg_single_X.shape[2], n_clones))
        for s in range(agg_single_X.shape[2]):
            this_log_emission = log_per_state_emission[:,(s*n_obs):(s*n_obs+n_obs), 0]
            for c in range(n_clones):
                sampled_h = list_h[:, (c*n_obs):(c*n_obs+n_obs)]
                # emission probability
                emission_llf[s,c] = this_log_emission[(sampled_h, np.arange(n_obs))].sum()
                # take average
                emission_llf[s,c] /= hmm_num_draws

        # save results
        list_log_mu.append(res['new_log_mu'])
        list_p_binom.append(res['new_p_binom'])
        list_cna_states.append( np.stack([res['pred_cnv'][(i*n_obs):(i*n_obs+n_obs)] for i in range(n_clones)]) )

        # update last_log_mu, last_p_binom, last_alphas, last_taus
        last_log_mu = res['new_log_mu']
        last_p_binom = res['new_p_binom']
        last_alphas = res['new_alphas']
        last_taus = res['new_taus']

        ##### Infer clone labels using the estimated log emission probability #####
        posterior_clones, list_labels, list_potentials = posterior_distribution_clone_labels(emission_llf, log_clonesize, adjacency_mat, coords, spatial_weight, num_chains, burnin, platform=platform, temperature=temperature)
        list_posterior_clones.append(posterior_clones)

        plot_posterior_clones_single(list_posterior_clones, coords, len(list_posterior_clones)-1, sample_ids)
        plt.show()
        # update log clone size
        log_clonesize = np.mean(posterior_clones, axis=0)
        log_clonesize = np.repeat( log_clonesize[None,:], single_X.shape[2], axis=0 )

        # update elbo
        prior_log_prob = np.zeros(n_clones)
        for c in range(n_clones):
            sampled_h = list_h[:, (c*n_obs):(c*n_obs+n_obs)]
            prior_log_prob[c] = compute_state_prior_prob_highdim(sampled_h, res['new_log_startprob'], res['new_log_transmat'], lengths).mean()

        eventual_llf = [compute_potential(emission_llf + log_clonesize, l, spatial_weight, adjacency_mat) + prior_log_prob.sum() for l in list_labels]
        list_elbo.append( np.mean(eventual_llf) )

        # compare the difference between the current and previous posterior_clones
        if np.square(list_posterior_clones[-1] - list_posterior_clones[-2]).mean() < sampling_tol:
            break

        ##### temperature annealing #####
        temperature = max(1.0, 0.95 * temperature)
        # if r > 2:
        #     temperature = max(1.0, temperature-2)

    return list_posterior_clones, list_cna_states, list_log_mu, list_p_binom, list_elbo


def plot_posterior_clones_single(list_posterior_clones, coords, idx, sample_ids=None):
    """
    Scatterplot of the probability of each spot being in each clone for iteration idx.
    Attributes
    ----------
    list_posterior_clones : list of np.arrays, each one has size (num_spots, num_clones)
        The posterior probability of each spot being in each clone at each iteration, output of infer_all
    coords : np.array, size (num_spots, 2)
        The spatial coordinates of each spot.
    sample_ids : np.array, size (num_spots)
        The sample id of each spot.
    idx : int
        The iteration index to plot.
    """
    shifted_coords = copy.copy(coords)
    if not sample_ids is None:
        offset = 0
        for s in np.sort(np.unique(sample_ids)):
            min_x = np.min(shifted_coords[sample_ids==s, 0])
            shifted_coords[sample_ids==s, 0] = shifted_coords[sample_ids==s, 0] - min_x + offset
            offset = np.max(shifted_coords[sample_ids==s, 0]) + 1

    n_clones = list_posterior_clones[idx].shape[1]
    fig, axes = plt.subplots(1, n_clones, figsize=(n_clones * 5*len(np.unique(sample_ids)), 5), facecolor='white', dpi=150)
    for c in range(list_posterior_clones[idx].shape[1]):
        axes[c].scatter(x=shifted_coords[:,0], y=-shifted_coords[:,1], c=list_posterior_clones[idx][:,c], cmap='magma_r', s=15, linewidth=0)
        axes[c].set_title('Clone %d' % c)
        norm = plt.Normalize(list_posterior_clones[idx][:,c].min(), list_posterior_clones[idx][:,c].max())
        axes[c].figure.colorbar( plt.cm.ScalarMappable(cmap='magma_r', norm=norm), ax=axes[c] )
        axes[c].axis('off')
    fig.tight_layout()
    return fig


def plot_posterior_clones_interactive(list_posterior_clones, coords, giffile, sample_ids=None, base_duration=500):
    import io
    import imageio

    shifted_coords = copy.copy(coords)
    if not sample_ids is None:
        offset = 0
        for s in np.sort(np.unique(sample_ids)):
            min_x = np.min(shifted_coords[sample_ids==s, 0])
            shifted_coords[sample_ids==s, 0] = shifted_coords[sample_ids==s, 0] - min_x + offset
            offset = np.max(shifted_coords[sample_ids==s, 0]) + 1
    
    n_clones = list_posterior_clones[0].shape[1]
    # List to store images and durations
    images = []
    # durations = base_duration * np.arange(len(list_posterior_clones))  # Different durations for each frame

    # Generate scatter plots and store them in memory
    for idx in range(len(list_posterior_clones)):
        fig, axes = plt.subplots(1, n_clones, figsize=(n_clones * 5*len(np.unique(sample_ids)), 5), dpi=150)
        for c in range(list_posterior_clones[idx].shape[1]):
            axes[c].scatter(x=shifted_coords[:,0], y=-shifted_coords[:,1], c=list_posterior_clones[idx][:,c], cmap='magma_r', s=15, linewidth=0)
            axes[c].set_title('Clone %d' % c)
            norm = plt.Normalize(list_posterior_clones[idx][:,c].min(), list_posterior_clones[idx][:,c].max())
            axes[c].figure.colorbar( plt.cm.ScalarMappable(cmap='magma_r', norm=norm), ax=axes[c] )
            axes[c].axis('off')
        fig.suptitle(f"interation {idx}")
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close()

    # Create a GIF from the in-memory images with different durations for each frame
    imageio.mimsave(giffile, images, duration=base_duration)


# def plot_posterior_clones_interactive(list_posterior_clones, coords, sample_ids=None):
#     import mpl_interactions.ipyplot as iplt

#     def c_func(x, y, ite):
#         return list_posterior_clones[ite][:,c]

#     shifted_coords = copy.copy(coords)
#     if not sample_ids is None:
#         offset = 0
#         for s in np.sort(np.unique(sample_ids)):
#             min_x = np.min(shifted_coords[sample_ids==s, 0])
#             shifted_coords[sample_ids==s, 0] = shifted_coords[sample_ids==s, 0] - min_x + offset
#             offset = np.max(shifted_coords[sample_ids==s, 0]) + 1

#     # make a gif of the posterior probability of each clone in multiple figure panels
#     n_clones = list_posterior_clones[0].shape[1]
#     fig, axes = plt.subplots(1, n_clones, figsize=(n_clones * 5*len(np.unique(sample_ids)), 5), facecolor='white', dpi=150)
#     for c in range(list_posterior_clones[0].shape[1]):
#         _ = iplt.scatter(x=shifted_coords[:,0], y=-shifted_coords[:,0], ite=np.arange(len(list_posterior_clones)), c=c_func, cmap='magma_r', s=15, linewidth=0, ax=axes[c])
#     fig.tight_layout()
#     return fig


def plot_posterior_baf_cnstate_single(single_X, single_total_bb_RD, single_tumor_prop, lengths, tumorprop_threshold, list_posterior_clones, list_cna_states, list_p_binom, idx, unique_chrs):
    """
    Scatterplot BAF along the genome colored by the hidden states at iteration idx.
    """
    posterior_clones = list_posterior_clones[idx]
    X = (single_X[:,:, single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :])
    total_bb_RD = single_total_bb_RD[:, single_tumor_prop>tumorprop_threshold] @ posterior_clones[single_tumor_prop>tumorprop_threshold, :]

    n_clones = list_posterior_clones[idx].shape[1]
    n_obs = single_X.shape[0]

    fig, axes = plt.subplots(n_clones, 1, figsize=(20,1.8*n_clones), sharex=True, sharey=True, dpi=150, facecolor="white")
    for s in np.arange(n_clones):
        cid = s
        this_pred = list_cna_states[idx][s,:]
        segments, labs = get_intervals(this_pred)
        seaborn.scatterplot(x=np.arange(X[:,1,cid].shape[0]), y=X[:,1,cid]/total_bb_RD[:,cid], \
            hue=pd.Categorical(this_pred, categories=np.arange(20), ordered=True), palette="tab10", s=15, linewidth=0, alpha=0.8, legend=False, ax=axes[s])
        axes[s].set_ylabel(f"clone {cid}\nphased AF")
        axes[s].set_ylim([-0.1, 1.1])
        axes[s].set_yticks([0, 0.5, 1])
        axes[s].set_xticks([])
        for i, seg in enumerate(segments):
            axes[s].plot(seg, [list_p_binom[idx][labs[i],0], list_p_binom[idx][labs[i],0]], c="black", linewidth=3)
            axes[s].plot(seg, [1-list_p_binom[idx][labs[i],0], 1-list_p_binom[idx][labs[i],0]], c="black", linewidth=3)

    for i in range(len(lengths)):
        median_len = (np.sum(lengths[:(i)]) + np.sum(lengths[:(i+1)])) / 2
        axes[-1].text(median_len, -0.21, unique_chrs[i])
        for k in range(n_clones):
            axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig