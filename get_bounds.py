from gurobipy import *

from gurobi_functions import create_mip_model, optimize
from recommendation import solve_recommendation_problem
from utils import get_u0


def get_mmu_ub(items, u0_type):
    """
    get the UB for the MMU problem for a set of items and U0

    we estimate this by solving the problem   min_{u \in U0} max_{x\in R} u^T x

    where U0 = {u | B * u >= b}
    """

    num_features = len(items[0].features)

    b_mat, b_vec = get_u0(u0_type, num_features)

    m = create_mip_model()

    u_vars = m.addVars(num_features, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")

    # objective
    tau = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")

    # add U0 constraints
    for j in range(len(b_vec)):
        m.addConstr(
            quicksum(b_mat[j, i] * u_vars[i] for i in range(num_features)) >= b_vec[j]
        )

    # add constraints for each x
    for i, item in enumerate(items):
        m.addConstr(
            tau >= quicksum(item.features[f] * u_vars[f] for f in range(num_features)),
            name=f"tau_constr_{i}",
        )

    m.setObjective(tau, sense=GRB.MINIMIZE)

    optimize(m)

    assert m.status == GRB.OPTIMAL

    return m.objVal


def get_mmu_lb(items, u0_type):
    """
    get the LB for the MMU problem for a set of items and U0

    this is simply the objval of the rec. problem, before asking any queries
    """

    objval, _ = solve_recommendation_problem(
        [],
        items,
        "maximin",
        gamma=0.0,
        verbose=False,
        fixed_rec_item=None,
        u0_type=u0_type,
        logger=None,
    )

    return objval


def get_mmr_ub(items, u0_type):
    """
    get the UB for the MMR problem for a set of items and U0

    this is simply the objval of the rec. problem, before asking any queries
    """

    objval, _ = solve_recommendation_problem(
        [],
        items,
        "mmr",
        gamma=0.0,
        verbose=False,
        fixed_rec_item=None,
        u0_type=u0_type,
        logger=None,
    )

    return objval


def get_mmr_lb(items, u0_type):
    """
    get the LB for the MMR problem for a set of items and U0

    we assume this is 0
    """
    return 0.0