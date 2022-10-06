from cProfile import label
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def optimize_integer_copynumber_oneclone(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, max_copynumber=6):
    """
    For each single clone, input are all vectors instead of matrices
    """
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
    obj = gp.LinExpr(np.tile([ np.sum(pred_cnv==k) for k in range(len(new_log_mu))], 2), var_abs_rdr + var_abs_baf)
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
    m.optimize()
    ##### get A allele and B allele integer copies corresponding to each HMM state #####
    B_copy = np.array([ m.getVarByName(f"c{k}1").X for k in range(len(new_log_mu)) ]).astype(int)
    A_copy = np.array([ m.getVarByName(f"c{k}2").X for k in range(len(new_log_mu)) ]).astype(int)
    # theoretical RDR and BAF per state
    total_copy_per_locus = np.array([A_copy[pred_cnv[g]] + B_copy[pred_cnv[g]] for g in range(len(base_nb_mean))])
    theoretical_mu = 1.0 * (A_copy + B_copy) / (lambd.dot(total_copy_per_locus))
    theoretical_p_binom = 1.0 * B_copy / (B_copy + A_copy)
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, m.ObjVal


def get_integer_copynumber(new_log_mu, base_nb_mean, new_p_binom, pred_cnv, max_copynumber):
    num_clones = new_p_binom.shape[1]
    B_copy = np.ones(new_p_binom.shape, dtype=int)
    A_copy = np.ones(new_p_binom.shape, dtype=int)
    theoretical_mu = np.ones(new_p_binom.shape)
    theoretical_p_binom = np.ones(new_p_binom.shape)
    sum_objective = 0
    for c in range(num_clones):
        tmp_B_copy, tmp_A_copy, tmp_theoretical_mu, tmp_theoretical_p_binom, tmp_obj = optimize_integer_copynumber_oneclone(new_log_mu[:,c], base_nb_mean[:,c], new_p_binom[:,c], pred_cnv, max_copynumber)
        B_copy[:,c] = tmp_B_copy
        A_copy[:,c] = tmp_A_copy
        theoretical_mu[:,c] = tmp_theoretical_mu
        theoretical_p_binom[:,c] = tmp_theoretical_p_binom
        sum_objective += tmp_obj
    return B_copy, A_copy, theoretical_mu, theoretical_p_binom, sum_objective


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