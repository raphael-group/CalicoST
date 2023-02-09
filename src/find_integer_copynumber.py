# from cProfile import label
import numpy as np
import pandas as pd
import scipy
# import gurobipy as gp
# from gurobipy import GRB
import copy

def old_hill_climbing_integer_copynumber_oneclone(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, max_allele_copy=4):
    n_states = len(new_log_mu)
    lambd = base_nb_mean / np.sum(base_nb_mean)
    weight_per_state = np.array([ np.sum(lambd[pred_cnv == s]) for s in range(n_states)])
    mu = np.exp(new_log_mu)
    def f(params, ploidy):
        # params of size (n_states, 2)
        if np.any( np.sum(params, axis=1) == 0 ):
            return len(pred_cnv) * 1e6
        denom = weight_per_state.dot( np.sum(params, axis=1) )
        frac_rdr = np.sum(params, axis=1) / denom
        frac_baf = params[:,0] / np.sum(params, axis=1)
        points_per_state = np.bincount(pred_cnv, minlength=params.shape[0] )
        neg_llf_expon = -scipy.stats.expon.logpdf(points_per_state, loc=0, scale=len(pred_cnv) / 50)
        return np.abs(mu - frac_rdr).dot(points_per_state) + 5 * np.abs(new_p_binom - frac_baf).dot(points_per_state) + 5 * np.maximum(0, np.sum(params, axis=1) - 2*ploidy-1).dot(neg_llf_expon)
    def hill_climb(initial_params, ploidy, max_iter=10):
        best_obj = f(initial_params, ploidy)
        params = copy.copy(initial_params)
        increased = True
        for counter in range(max_iter):
            increased = False
            for k in range(params.shape[0]):
                this_best_obj = best_obj
                this_best_k = copy.copy(params[k,:])
                for candi in candidates:
                    params[k,:] = candi
                    obj = f(params, ploidy)
                    if obj < this_best_obj:
                        print(k, candi, obj, this_best_obj, 2*ploidy+1, 0.1 * np.maximum(0, np.sum(params[k,:]) - 2*ploidy-1) * np.sum(pred_cnv==k))
                        this_best_obj = obj
                        this_best_k = candi
                increased = (increased | (this_best_obj < best_obj))
                params[k,:] = this_best_k
                best_obj = this_best_obj
            if not increased:
                break
        return params, best_obj
    # candidate integer copy states
    candidates = np.array([ [i,j] for i in range(max_allele_copy + 1) for j in range(max_allele_copy) if not (i == 0 and j == 0)])
    # find the best copy number states starting from various ploidy
    best_obj = np.inf
    best_integer_copies = np.zeros((n_states, 2), dtype=int)
    for ploidy in range(1, 3):
        initial_params = np.ones((n_states, 2), dtype=int) * ploidy
        params, obj = hill_climb(initial_params, ploidy)
        if obj < best_obj:
            best_obj = obj
            best_integer_copies = copy.copy(params)
    return best_integer_copies, best_obj


def hill_climbing_integer_copynumber_oneclone(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, max_allele_copy=4):
    n_states = len(new_log_mu)
    lambd = base_nb_mean / np.sum(base_nb_mean)
    weight_per_state = np.array([ np.sum(lambd[pred_cnv == s]) for s in range(n_states)])
    mu = np.exp(new_log_mu)
    def f(params, ploidy):
        # params of size (n_states, 2)
        if np.any( np.sum(params, axis=1) == 0 ):
            return len(pred_cnv) * 1e6
        denom = weight_per_state.dot( np.sum(params, axis=1) )
        frac_rdr = np.sum(params, axis=1) / denom
        frac_baf = params[:,0] / np.sum(params, axis=1)
        points_per_state = np.bincount(pred_cnv, minlength=params.shape[0] )
        neg_llf_expon = -scipy.stats.expon.logpdf(points_per_state, loc=0, scale=len(pred_cnv) / 50)
        return np.abs(mu - frac_rdr).dot(points_per_state) + 5 * np.abs(new_p_binom - frac_baf).dot(points_per_state) #+ 5 * np.maximum(0, np.sum(params, axis=1) - ploidy-1).dot(neg_llf_expon)
    def hill_climb(initial_params, ploidy, idx_med, max_iter=10):
        best_obj = f(initial_params, ploidy)
        params = copy.copy(initial_params)
        increased = True
        for counter in range(max_iter):
            increased = False
            for k in range(params.shape[0]):
                this_best_obj = best_obj
                this_best_k = copy.copy(params[k,:])
                for candi in candidates:
                    if k == idx_med and np.sum(candi) != ploidy:
                        continue
                    params[k,:] = candi
                    obj = f(params, ploidy)
                    if obj < this_best_obj:
                        print(k, candi, obj, this_best_obj, ploidy+1, 0.1 * np.maximum(0, np.sum(params[k,:]) - ploidy-1) * np.sum(pred_cnv==k))
                        this_best_obj = obj
                        this_best_k = candi
                increased = (increased | (this_best_obj < best_obj))
                params[k,:] = this_best_k
                best_obj = this_best_obj
            if not increased:
                break
        return params, best_obj
    # candidate integer copy states
    candidates = np.array([ [i,j] for i in range(max_allele_copy + 1) for j in range(max_allele_copy) if not (i == 0 and j == 0)])
    # find the best copy number states starting from various ploidy
    best_obj = np.inf
    best_integer_copies = np.zeros((n_states, 2), dtype=int)
    # fix the genomic bin with the median new_log_mu to have exactly ploidy genomes
    bidx_med = np.argsort(new_log_mu[pred_cnv])[ int(len(pred_cnv)/2) ]
    idx_med = pred_cnv[bidx_med]
    for ploidy in range(1, 5):
        initial_params = np.ones((n_states, 2), dtype=int) * int(ploidy / 2)
        initial_params[:, 1] = ploidy - initial_params[:, 0]
        params, obj = hill_climb(initial_params, ploidy, idx_med)
        if obj < best_obj:
            best_obj = obj
            best_integer_copies = copy.copy(params)
    return best_integer_copies, best_obj


def get_genelevel_cnv_oneclone(A_copy, B_copy, x_gene_list):
    map_gene_bin = {}
    for i,x in enumerate(x_gene_list):
        this_genes = [z for z in x.split(" ") if z != ""]
        for g in this_genes:
            map_gene_bin[g] = i
    gene_list = np.sort(np.array(list(map_gene_bin.keys())))
    gene_level_copies = np.zeros( (len(gene_list), 2), dtype=int )
    for i,g in enumerate(gene_list):
        idx = map_gene_bin[g]
        gene_level_copies[i, 0] = A_copy[idx]
        gene_level_copies[i, 1] = B_copy[idx]
    return pd.DataFrame({"A":gene_level_copies[:,0], "B":gene_level_copies[:,1]}, index=gene_list)


def convert_copy_to_states(A_copy, B_copy):
    tmp = A_copy + B_copy
    tmp = tmp[~np.isnan(tmp)]
    base_ploidy = np.median(tmp)
    coarse_states = np.array(["neutral"] * A_copy.shape[0])
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy != B_copy) ] = "del"
    coarse_states[ (A_copy + B_copy < base_ploidy) & (A_copy == B_copy) ] = "bdel"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy != B_copy) ] = "amp"
    coarse_states[ (A_copy + B_copy > base_ploidy) & (A_copy == B_copy) ] = "bamp"
    coarse_states[ (A_copy + B_copy == base_ploidy) & (A_copy != B_copy) ] = "loh"
    coarse_states[coarse_states == "neutral"] = "neu"
    return coarse_states


"""
def optimize_integer_copynumber_oneclone(new_log_mu, base_nb_mean, total_bb_RD, new_p_binom, pred_cnv, max_copynumber=6):
    '''
    For each single clone, input are all vectors instead of matrices
    '''
    m = gp.Model("ilp")
    ##### Create variables #####
    var_copies_1 = []
    var_copies_2 = []
    # allele-specific copy numbers
    for k in range(len(new_log_mu)):
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}1")
        var_copies_1.append( tmp )
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}2")
        var_copies_2.append( tmp )
    # absolute value of the expressions in objective function
    var_abs_rdr = []
    var_abs_baf = []
    for k in range(len(new_log_mu)):
        tmp = m.addVar(lb=0, name=f"rdr{k}")
        var_abs_rdr.append( tmp )
        tmp = m.addVar(lb=0, name=f"baf{k}")
        var_abs_baf.append( tmp )
    ##### Set objective #####
    obj = gp.LinExpr([np.sum((pred_cnv==k) & (base_nb_mean>0)) for k in range(len(new_log_mu))], var_abs_rdr)
    obj.addTerms([np.sum((pred_cnv==k) & (total_bb_RD>0)) for k in range(len(new_p_binom))], var_abs_baf)
    m.setObjective(obj, GRB.MINIMIZE)
    ##### Add constraint #####
    # total copy >= 1
    for k in range(len(new_log_mu)):
        m.addConstr(var_copies_1[k] + var_copies_2[k] >= 1, f"min_cn_{k}")
    # total copy not exceeding max_copynumber
    for k in range(len(new_log_mu)):
        m.addConstr(var_copies_1[k] + var_copies_2[k] <= max_copynumber, f"max_cn_{k}")
    # RDR
    lambd = base_nb_mean / np.sum(base_nb_mean)
    mu = np.exp(new_log_mu)
    weight_total_copy = gp.LinExpr( np.append(lambd, lambd), [var_copies_1[pred_cnv[g]] for g in range(len(base_nb_mean))] + [var_copies_2[pred_cnv[g]] for g in range(len(base_nb_mean))])
    for k in range(len(new_log_mu)):
        m.addConstr(mu[k] * weight_total_copy - var_copies_1[k] - var_copies_2[k] <= var_abs_rdr[k], f"const_rdr_{k}_1" )
        m.addConstr(-mu[k] * weight_total_copy + var_copies_1[k] + var_copies_2[k] <= var_abs_rdr[k], f"const_rdr_{k}_1" )
    # BAF
    for k in range(len(new_log_mu)):
        m.addConstr( (new_p_binom[k] - 1) * var_copies_1[k] + new_p_binom[k] * var_copies_2[k] <= var_abs_baf[k], f"const_baf_{k}_1" )
        m.addConstr( -(new_p_binom[k] - 1) * var_copies_1[k] - new_p_binom[k] * var_copies_2[k] <= var_abs_baf[k], f"const_baf_{k}_1" )
    ##### Optimize model #####
    m.Params.LogToConsole = 0
    m.optimize()
    ##### get A allele and B allele integer copies corresponding to each HMM state #####
    B_copy = np.array([ m.getVarByName(f"c{k}1").X for k in range(len(new_log_mu)) ]).astype(int)
    A_copy = np.array([ m.getVarByName(f"c{k}2").X for k in range(len(new_log_mu)) ]).astype(int)
    # theoretical RDR and BAF per state
    total_copy_per_locus = A_copy[pred_cnv] + B_copy[pred_cnv]
    theoretical_mu = 1.0 * (A_copy + B_copy) / (lambd.dot(total_copy_per_locus))
    theoretical_p_binom = 1.0 * B_copy / (B_copy + A_copy)
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, m.ObjVal


def optimize_integer_copynumber_oneclone_v2(new_log_mu, base_nb_mean, total_bb_RD, new_p_binom, pred_cnv, base_copynumber=4, max_copynumber=6):
    '''
    For each single clone, input are all vectors instead of matrices
    '''
    m = gp.Model("ilp")
    ##### Create variables #####
    var_copies_1 = []
    var_copies_2 = []
    # allele-specific copy numbers
    for k in range(len(new_log_mu)):
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}1")
        var_copies_1.append( tmp )
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}2")
        var_copies_2.append( tmp )
    # absolute value of the expressions in objective function
    var_abs_rdr = []
    var_abs_baf = []
    var_abs_total = []
    for k in range(len(new_log_mu)):
        tmp = m.addVar(lb=0, name=f"rdr{k}")
        var_abs_rdr.append( tmp )
        tmp = m.addVar(lb=0, name=f"baf{k}")
        var_abs_baf.append( tmp )
        tmp = m.addVar(lb=0, name=f"total{k}")
        var_abs_total.append( tmp )
    ##### Set objective #####
    obj = gp.LinExpr([np.sum((pred_cnv==k) & (base_nb_mean>0)) for k in range(len(new_log_mu))], var_abs_rdr)
    obj.addTerms([np.sum((pred_cnv==k) & (total_bb_RD>0)) for k in range(len(new_p_binom))], var_abs_baf)
    obj.addTerms([0.02 * np.sum((pred_cnv==k) & (base_nb_mean>0)) for k in range(len(new_log_mu))], var_abs_total)
    obj.addTerms([0.02 * np.sum((pred_cnv==k) & (total_bb_RD>0)) for k in range(len(new_p_binom))], var_abs_total)
    m.setObjective(obj, GRB.MINIMIZE)
    ##### Add constraint #####
    # total copy >= 1
    for k in range(len(new_log_mu)):
        m.addConstr(var_copies_1[k] + var_copies_2[k] >= 1, f"min_cn_{k}")
    # total copy not exceeding max_copynumber
    for k in range(len(new_log_mu)):
        m.addConstr(var_copies_1[k] + var_copies_2[k] <= max_copynumber, f"max_cn_{k}")
    # total copy similar to base_copynumber
    for k in range(len(new_log_mu)):
        m.addConstr(var_copies_1[k] + var_copies_2[k] - base_copynumber <= var_abs_total[k], f"total_cn_{k}_1")
        m.addConstr(base_copynumber - var_copies_1[k] - var_copies_2[k] <= var_abs_total[k], f"total_cn_{k}_2")
    # RDR
    lambd = base_nb_mean / np.sum(base_nb_mean)
    mu = np.exp(new_log_mu)
    weight_total_copy = gp.LinExpr( np.append(lambd, lambd), [var_copies_1[pred_cnv[g]] for g in range(len(base_nb_mean))] + [var_copies_2[pred_cnv[g]] for g in range(len(base_nb_mean))])
    for k in range(len(new_log_mu)):
        m.addConstr(mu[k] * weight_total_copy - var_copies_1[k] - var_copies_2[k] <= var_abs_rdr[k], f"const_rdr_{k}_1" )
        m.addConstr(-mu[k] * weight_total_copy + var_copies_1[k] + var_copies_2[k] <= var_abs_rdr[k], f"const_rdr_{k}_1" )
    # BAF
    for k in range(len(new_log_mu)):
        m.addConstr( (new_p_binom[k] - 1) * var_copies_1[k] + new_p_binom[k] * var_copies_2[k] <= var_abs_baf[k], f"const_baf_{k}_1" )
        m.addConstr( -(new_p_binom[k] - 1) * var_copies_1[k] - new_p_binom[k] * var_copies_2[k] <= var_abs_baf[k], f"const_baf_{k}_1" )
    ##### Optimize model #####
    m.Params.LogToConsole = 0
    m.optimize()
    ##### get A allele and B allele integer copies corresponding to each HMM state #####
    B_copy = np.array([ m.getVarByName(f"c{k}1").X for k in range(len(new_log_mu)) ]).astype(int)
    A_copy = np.array([ m.getVarByName(f"c{k}2").X for k in range(len(new_log_mu)) ]).astype(int)
    # theoretical RDR and BAF per state
    total_copy_per_locus = A_copy[pred_cnv] + B_copy[pred_cnv]
    theoretical_mu = 1.0 * (A_copy + B_copy) / (lambd.dot(total_copy_per_locus))
    theoretical_p_binom = 1.0 * B_copy / (B_copy + A_copy)
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, m.ObjVal


def get_integer_copynumber(new_log_mu, base_nb_mean, total_bb_RD, new_p_binom, pred_cnv, max_copynumber):
    num_clones = new_p_binom.shape[1]
    B_copy = np.ones(new_p_binom.shape, dtype=int)
    A_copy = np.ones(new_p_binom.shape, dtype=int)
    theoretical_mu = np.ones(new_p_binom.shape)
    theoretical_p_binom = np.ones(new_p_binom.shape)
    sum_objective = 0
    for c in range(num_clones):
        tmp_B_copy, tmp_A_copy, tmp_theoretical_mu, tmp_theoretical_p_binom, tmp_obj = optimize_integer_copynumber_oneclone(new_log_mu[:,c], base_nb_mean[:,c], total_bb_RD[:,c], new_p_binom[:,c], pred_cnv, max_copynumber)
        B_copy[:,c] = tmp_B_copy
        A_copy[:,c] = tmp_A_copy
        theoretical_mu[:,c] = tmp_theoretical_mu
        theoretical_p_binom[:,c] = tmp_theoretical_p_binom
        sum_objective += tmp_obj
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, sum_objective


def eval_objective(new_log_mu, base_nb_mean, total_bb_RD, new_p_binom, pred_cnv, B_copy, A_copy):
    num_clones = new_p_binom.shape[1]
    objectives_rdr = []
    objectives_baf = []
    for c in range(num_clones):
        # RDR
        idx_nonzero = np.where(base_nb_mean[:,c] > 0)[0]
        total_copy = (A_copy + B_copy)[pred_cnv, c]
        lambd = base_nb_mean[:,c] / np.sum(base_nb_mean[:,c])
        weight_total_copy = lambd.dot(total_copy)
        obj_rdr = np.sum(np.abs( np.exp(new_log_mu[pred_cnv,c][idx_nonzero]) * weight_total_copy - total_copy[idx_nonzero] ))
        objectives_rdr.append( obj_rdr )
        # BAF
        idx_nonzero = np.where(total_bb_RD[:,c] > 0)[0]
        obj_baf = np.sum(np.abs( total_copy[idx_nonzero] * new_p_binom[pred_cnv, c][idx_nonzero] - B_copy[pred_cnv, c][idx_nonzero] ))
        objectives_baf.append( obj_baf )
    return objectives_rdr, objectives_baf


def composite_hmm_optimize_integer_copynumber(base_nb_mean, total_bb_RD, new_log_mu, new_scalefactors, new_p_binom, state_tuples, pred_cnv, max_copynumber=6):
    '''
    Attributes
    ----------
    base_nb_mean : array, (n_obs, n_spots)
        Expected read counts per bin (or SNP) under diploid genome assumption.

    total_bb_RD : array, (n_obs, n_spots)
        Total SNP-covering reads per SNP.

    new_log_mu : array, (n_individual_states, )
        Log fold change of RDR for each copy number state

    new_scalefactors : array, (n_spots, )
        Log normalization factor due to total copy number change along the whole genome of that clone.

    new_p_binom : array, (n_individual_states, )    
        BAF of each copy number state.

    state_tuples : array, (n_composite_states, n_spots)
        Each composite state is a omposition of copy numnber states across all clones.

    pred_cnv : array, (n_obs, )
        Categorical variables to indicate the composite state each bin (or SNP) is in. 
    '''
    n_obs = base_nb_mean.shape[0]
    n_spots = base_nb_mean.shape[1]
    n_individual_states = int(len(new_log_mu) / 2)
    n_composite_states = int(len(state_tuples) / 2)
    # gurobi ILP to infer integer copy numbers
    m = gp.Model("ilp")
    ##### Create variables #####
    var_copies_1 = []
    var_copies_2 = []
    # allele-specific copy numbers
    for k in range(n_individual_states):
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}1")
        var_copies_1.append( tmp )
        tmp = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"c{k}2")
        var_copies_2.append( tmp )
    # absolute value of the expressions in objective function, per clone per individual copy number state
    var_abs_rdr = [[] for c in range(n_spots)]
    var_abs_baf_B = [[] for c in range(n_spots)]
    var_abs_baf_A = [[] for c in range(n_spots)]
    for c in range(n_spots):
        for k in range(n_individual_states):
            tmp = m.addVar(lb=0, name=f"rdr_{c}_{k}")
            var_abs_rdr[c].append( tmp )
            tmp = m.addVar(lb=0, name=f"bafB_{c}_{k}")
            var_abs_baf_B[c].append( tmp )
            tmp = m.addVar(lb=0, name=f"bafA_{c}_{k}")
            var_abs_baf_A[c].append( tmp )
    ##### Set objective #####
    obj = gp.LinExpr(0)
    for c in range(n_spots):
        this_pred_cnv = state_tuples[pred_cnv, c]
        # RDR
        coef = [np.sum(((this_pred_cnv==k) | (this_pred_cnv==k+n_individual_states)) & (base_nb_mean[:,c]>0)) for k in range(n_individual_states)]
        obj.addTerms(coef, var_abs_rdr[c])
        # BAF
        coef = [np.sum((this_pred_cnv==k) & (total_bb_RD[:,c]>0)) for k in range(n_individual_states)]
        obj.addTerms(coef, var_abs_baf_B[c])
        coef = [np.sum((this_pred_cnv==k+n_individual_states) & (total_bb_RD[:,c]>0)) for k in range(n_individual_states)]
        obj.addTerms(coef, var_abs_baf_A[c])
    m.setObjective(obj, GRB.MINIMIZE)
    ##### Add constraint #####
    # total copy >= 1
    for k in range(n_individual_states):
        m.addConstr(var_copies_1[k] + var_copies_2[k] >= 1, f"min_cn_{k}")
    # total copy not exceeding max_copynumber
    for k in range(n_individual_states):
        m.addConstr(var_copies_1[k] + var_copies_2[k] <= max_copynumber, f"max_cn_{k}")
    # RDR
    for c in range(n_spots):
        this_pred_cnv = state_tuples[pred_cnv, c]
        this_pred_cnv = this_pred_cnv % n_individual_states
        lambd = base_nb_mean[:,c] / np.sum(base_nb_mean[:,c])
        mu = np.exp(new_log_mu[:n_individual_states]) if c==0 else np.exp(new_log_mu[:n_individual_states] + new_scalefactors[c-1])
        weight_total_copy = gp.LinExpr( np.append(lambd, lambd), [var_copies_1[this_pred_cnv[g]] for g in range(n_obs)] + [var_copies_2[this_pred_cnv[g]] for g in range(n_obs)])
        for k in range(n_individual_states):
            m.addConstr(mu[k] * weight_total_copy - var_copies_1[k] - var_copies_2[k] <= var_abs_rdr[c][k], f"const_rdr__{c}_{k}_1" )
            m.addConstr(-mu[k] * weight_total_copy + var_copies_1[k] + var_copies_2[k] <= var_abs_rdr[c][k], f"const_rdr__{c}_{k}_2" )
    # BAF
    for c in range(n_spots):
        this_pred_cnv = state_tuples[pred_cnv, c]
        # B allele
        for k in range(n_individual_states):
            m.addConstr( (new_p_binom[k] - 1) * var_copies_1[k] + new_p_binom[k] * var_copies_2[k] <= var_abs_baf_B[c][k], f"const_baf_{c}_{k}_1" )
            m.addConstr( -(new_p_binom[k] - 1) * var_copies_1[k] - new_p_binom[k] * var_copies_2[k] <= var_abs_baf_B[c][k], f"const_baf_{c}_{k}_2" )
        # A allele
        for k in range(n_individual_states):
            m.addConstr( new_p_binom[k] * var_copies_1[k] + (1-new_p_binom[k]) * var_copies_2[k] <= var_abs_baf_A[c][k], f"const_baf_A_{c}_{k}_1" )
            m.addConstr( -new_p_binom[k] * var_copies_1[k] - (1-new_p_binom[k]) * var_copies_2[k] <= var_abs_baf_A[c][k], f"const_baf_A_{c}_{k}_2" )
    ##### Optimize model #####
    m.Params.LogToConsole = 0
    m.optimize()
    ##### get A allele and B allele integer copies corresponding to each HMM state #####
    B_copy = np.array([ m.getVarByName(f"c{k}1").X for k in range(n_individual_states) ]).astype(int).reshape(-1,1)
    A_copy = np.array([ m.getVarByName(f"c{k}2").X for k in range(n_individual_states) ]).astype(int).reshape(-1,1)
    # theoretical RDR and BAF per state
    theoretical_mu = []
    theoretical_p_binom = []
    for c in range(n_spots):
        this_pred_cnv = state_tuples[pred_cnv, c]
        lambd = base_nb_mean[:,c] / np.sum(base_nb_mean[:,c])
        total_copy_per_locus = A_copy[this_pred_cnv % n_individual_states,0] + B_copy[this_pred_cnv % n_individual_states,0]
        theoretical_mu.append( 1.0 * (A_copy + B_copy) / (lambd.dot(total_copy_per_locus)) )
        theoretical_p_binom.append( 1.0 * B_copy / (B_copy + A_copy) )
    theoretical_mu = np.hstack(theoretical_mu)
    theoretical_p_binom = np.hstack(theoretical_p_binom)
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, m.ObjVal


def composite_hmm_eval_objective(base_nb_mean, total_bb_RD, new_log_mu, new_scalefactors, new_p_binom, state_tuples, pred_cnv, B_copy, A_copy):
    n_spots = base_nb_mean.shape[1]
    n_individual_states = int(len(new_log_mu) / 2)
    objectives_rdr = np.zeros( (n_individual_states, n_spots) )
    objectives_baf_B = np.zeros( (n_individual_states, n_spots) )
    objectives_baf_A = np.zeros( (n_individual_states, n_spots) )
    for c in range(n_spots):
        this_pred_cnv = state_tuples[pred_cnv, c]
        total_copy = (A_copy + B_copy)[this_pred_cnv % n_individual_states, 0]
        # RDR
        lambd = base_nb_mean[:,c] / np.sum(base_nb_mean[:,c])
        weight_total_copy = lambd.dot(total_copy)
        for k in range(n_individual_states):
            num_entries = np.sum( (base_nb_mean[:,c] > 0) & (this_pred_cnv % n_individual_states == k) )
            if c == 0:
                objectives_rdr[k,c] = num_entries * np.abs( np.exp(new_log_mu[k]) * weight_total_copy - A_copy[k,0] - B_copy[k,0] )
            else:
                objectives_rdr[k,c] = num_entries * np.abs( np.exp(new_log_mu[k] + new_scalefactors[c-1]) * weight_total_copy - A_copy[k,0] - B_copy[k,0] )
        # BAF
        for k in range(n_individual_states):
            num_entries = np.sum( (total_bb_RD[:,c] > 0) & (this_pred_cnv == k) )
            objectives_baf_B[k,c] = num_entries * np.abs( new_p_binom[k] * (A_copy[k,0] + B_copy[k,0]) - B_copy[k, 0] )
            num_entries = np.sum( (total_bb_RD[:,c] > 0) & (this_pred_cnv == k + n_individual_states) )
            objectives_baf_A[k,c] = num_entries * np.abs( new_p_binom[k] * (A_copy[k,0] + B_copy[k,0]) - A_copy[k, 0] )
    return objectives_rdr, objectives_baf_B, objectives_baf_A

##### below are gurobi example #####
# try:

#     # Create a new model
#     m = gp.Model("mip1")

#     # Create variables
#     x = m.addVar(vtype=GRB.BINARY, name="x")
#     y = m.addVar(vtype=GRB.BINARY, name="y")
#     z = m.addVar(vtype=GRB.BINARY, name="z")

#     # Set objective
#     m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

#     # Add constraint: x + 2 y + 3 z <= 4
#     m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

#     # Add constraint: x + y >= 1
#     m.addConstr(x + y >= 1, "c1")

#     # Optimize model
#     m.optimize()

#     for v in m.getVars():
#         print('%s %g' % (v.VarName, v.X))

#     print('Obj: %g' % m.ObjVal)

# except gp.GurobiError as e:
#     print('Error code ' + str(e.errno) + ': ' + str(e))

# except AttributeError:
#     print('Encountered an attribute error')
"""