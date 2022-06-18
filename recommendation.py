# function for making a robust recommendation to an agent, after observing some answered queries
from gurobi_functions import create_mip_model, optimize
from preference_classes import Item
from utils import find_analytic_center, get_u0
from gurobipy import *
import numpy as np


def solve_recommendation_problem(
    answered_queries,
    items,
    problem_type,
    gamma=0,
    verbose=False,
    fixed_rec_item=None,
    u0_type="box",
    logger=None,
):
    """solve the robust recommendation problem, and return the recommended item and worst-case utility vector"""

    # print([q.response for q in answered_queries])

    valid_responses = [-1, 1]

    assert set([q.response for q in answered_queries]).issubset(set(valid_responses))
    assert problem_type in ["maximin", "mmr"]
    assert gamma >= 0

    # some constants
    K = len(answered_queries)
    for i in range(len(answered_queries)):
        print(answered_queries[i].item_A.id, answered_queries[i].item_B.id, answered_queries[i].response)
    num_features = len(items[0].features)
    z_vectors = [q.z for q in answered_queries]
    responses = [q.response for q in answered_queries]

    # polyhedral definition for U^0, b_mat and b_vec
    b_mat, b_vec = get_u0(u0_type, num_features)

    # define beta vars (more dual variables)
    m_const = len(b_vec)

    if logger is not None:
        log_file = logger.handlers[0].baseFilename
        logger.debug("writing gurobi logs for recommendation problem")
    else:
        log_file = None

    # set up the Gurobi model
    m = create_mip_model(verbose=verbose, log_file=log_file)

    # if the recommended item is fixed, don't create y vars
    if fixed_rec_item is not None:
        assert isinstance(fixed_rec_item, Item)
        y_vars = None
    else:
        # y vars : to select x^r, the recommended item in scenario r
        y_vars = m.addVars(len(items), vtype=GRB.BINARY, name="y")
        m.addSOS(GRB.SOS_TYPE1, [y_vars[i] for i in range(len(items))])
        m.addConstr(
            quicksum(y_vars[i] for i in range(len(items))) == 1, name="y_constr"
        )
        fixed_rec_item = None

    # add dual variables
    if problem_type == "maximin":
        mu_var, alpha_vars, beta_vars = add_rec_dual_variables(
            m,
            K,
            gamma,
            problem_type,
            m_const,
            y_vars,
            num_features,
            items,
            b_mat,
            responses,
            z_vectors,
            fixed_rec_item,
        )
    if problem_type == "mmr":
        theta_var = m.addVar(
            vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta"
        )
        beta_vars = {}
        alpha_vars = {}
        mu_vars = {}
        for item in items:
            (
                mu_vars[item.id],
                alpha_vars[item.id],
                beta_vars[item.id],
            ) = add_rec_dual_variables(
                m,
                K,
                gamma,
                problem_type,
                m_const,
                y_vars,
                num_features,
                items,
                b_mat,
                responses,
                z_vectors,
                fixed_rec_item,
                mmr_item=item,
            )
            m.addConstr(
                theta_var
                >= quicksum([b_vec[j] * beta_vars[item.id][j] for j in range(m_const)])
                + gamma * mu_vars[item.id]
            )

    if problem_type == "maximin":
        obj = (
            quicksum([b_vec[j] * beta_vars[j] for j in range(m_const)]) + gamma * mu_var
        )
        m.setObjective(obj, sense=GRB.MAXIMIZE)
    elif problem_type == "mmr":
        m.setObjective(theta_var, sense=GRB.MINIMIZE)

    m.Params.DualReductions = 0
    optimize(m, raise_warnings=True)

    # --- gather results ---

    # if the model is unbounded (uncertainty set it empty), return None
    if m.status == GRB.OPTIMAL:
        if m.status == GRB.INF_OR_UNBD:
            lp_file = os.path.join(
                os.getenv("HOME"), "recommendation_problem_infeas_unbd.lp"
            )
            ilp_file = os.path.join(
                os.getenv("HOME"), "recommendation_problem_infeas_unbd.ilp"
            )
            print(
                f"badly-behaved model. writing lp to: {lp_file}, writing ilp to: {ilp_file}"
            )
            m.computeIIS()
            m.write(lp_file)
            m.write(ilp_file)
            raise Exception("model infeasible or unbounded")
        if m.status == GRB.UNBOUNDED:
            lp_file = os.path.join(
                os.getenv("HOME"), "recommendation_problem_infeas_unbd.lp"
            )
            print(f"badly-behaved model. writing lp to: {lp_file}")
            m.write(lp_file)
            raise Exception("model is unbounded")

        assert m.status == GRB.OPTIMAL

        if fixed_rec_item is not None:
            return m.objVal, fixed_rec_item
        else:
            # find the recommended item
            y_vals = np.array([var.x for var in y_vars.values()])
            selected_items = np.argwhere(y_vals > 0.5)

            # there can only be one recommended item
            assert len(selected_items) == 1
            recommended_item = items[selected_items[0][0]]

            # # finally, find the minimum u-vector
            return m.objVal, recommended_item

    else:
        return 1, items[np.random.randint(0, len(items) - 1)]


def add_rec_dual_variables(
    m,
    K,
    gamma,
    problem_type,
    m_const,
    y_vars,
    num_features,
    items,
    b_mat,
    responses,
    z_vectors,
    fixed_rec_item,
    mmr_item=None,
):

    if gamma > 0:
        # dual variable for inconsistencies constraint
        if problem_type == "maximin":
            mu_var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0.0, name="mu")
        if problem_type == "mmr":
            mu_var = m.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name=f"mu_{mmr_item.id}"
            )
    else:
        mu_var = 0

    # the dual variables have a different sign for mmr and maximin
    if problem_type == "maximin":
        dual_lb = 0.0
        dual_ub = GRB.INFINITY
        beta_name = "beta"
        alpha_name = "alpha"
    if problem_type == "mmr":
        dual_lb = -GRB.INFINITY
        dual_ub = 0.0
        beta_name = f"beta_{mmr_item.id}"
        alpha_name = f"alpha_{mmr_item.id}"

    beta_vars = m.addVars(
        m_const, vtype=GRB.CONTINUOUS, lb=dual_lb, ub=dual_ub, name=beta_name
    )
    alpha_vars = m.addVars(
        K, vtype=GRB.CONTINUOUS, lb=dual_lb, ub=dual_ub, name=alpha_name
    )

    if gamma > 0:
        if problem_type == "maximin":
            for k in range(K):
                m.addConstr(alpha_vars[k] + mu_var <= 0, name=f"alpha_constr_k{k}")
        if problem_type == "mmr":
            for k in range(K):
                m.addConstr(
                    alpha_vars[k] + mu_var >= 0, name=f"alpha_{mmr_item.id}_constr_k{k}"
                )

    # define an expression for each feature of x
    if fixed_rec_item is not None:
        x_features = fixed_rec_item.features
    else:
        x_features = [
            quicksum([y_vars[i] * items[i].features[j] for i in range(len(items))])
            for j in range(num_features)
        ]

    # the big constraint ...
    for f in range(num_features):
        lhs_1 = quicksum(
            [responses[k] * z_vectors[k][f] * alpha_vars[k] for k in range(K)]
        )
        lhs_2 = quicksum([b_mat[j, f] * beta_vars[j] for j in range(m_const)])

        if problem_type == "maximin":
            rhs = x_features[f]
            feature_name = f"feature_{f}"
        if problem_type == "mmr":
            assert isinstance(mmr_item, Item)
            rhs = mmr_item.features[f] - x_features[f]
            feature_name = f"feature_{mmr_item.id}_{f}"

        m.addConstr(lhs_1 + lhs_2 == rhs, name=feature_name)

    return mu_var, alpha_vars, beta_vars


def recommend_item_ac(answered_queries, items, gamma, u0_type):
    """make a recommendation to the agent, assuming their u-vector is their current AC"""
    ac = find_analytic_center(answered_queries, len(items[0].features), gamma, u0_type)
    print("rac:", ac)

    true_util = np.array([np.dot(ac, item.features) for item in items])
    # print(true_util)

    # select any item with max utility
    max_inds = list(np.where(np.array(true_util) == max(true_util))[0])
    # print(max_inds)

    if len(max_inds) > 1:
        return items[np.random.choice(max_inds, 1)[0]]
    else:
        return items[max_inds[0]]


def recommend_item_ac_robust(answered_queries, items, problem_type, gamma, u0_type, verbose=False, logger=None):
    """make a recommendation to the agent, assuming their u-vector is their current AC"""
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1, 0]

    assert problem_type in ["mmu", "mmr"]
    assert u0_type in ["box", "positive_normed"]

    true_util = []
    eps = 0.1
    n_par = 10 * len(items[0].features)
    num_features = len(items[0].features)

    for item in items:

        if logger is not None:
            log_file = logger.handlers[0].baseFilename
            logger.debug("writing gurobi logs for recommendation problem")
        else:
            log_file = None

        # set up the Gurobi model
        m = create_mip_model(verbose=verbose, log_file=log_file)

        u_vars = m.addVars(
            num_features,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="u",
        )
        phi_vars = m.addVars(len(answered_queries), vtype=GRB.BINARY, name="phi")

        if problem_type == "mmu":
            m.setObjective(quicksum(u_vars[i] * item.features[i] for i in range(num_features)), sense=GRB.MINIMIZE)
        if problem_type == "mmr":
            m.params.NonConvex = 2
            x_vars = m.addVars(len(items), vtype=GRB.BINARY, name="x")
            inner_max_expr = [quicksum(u_vars[i] * items[j].features[i] for i in range(num_features)) for j in range(len(items))]
            m.setObjective(quicksum(inner_max_expr[i] * x_vars[i] for i in range(len(items))) - quicksum(u_vars[i] * item.features[i] for i in range(num_features)), sense=GRB.MAXIMIZE)

        # u-constraints defined by queries
        if len(answered_queries) > 0:
            for k, q in enumerate(answered_queries):
                assert len(q.z) == num_features
                utz_expr = quicksum(u_vars[i] * q.z[i] for i in range(num_features))
                if q.response == 1:
                    m.addConstr(utz_expr + (n_par + eps) * phi_vars[k] >= eps)
                    m.addConstr(utz_expr + (n_par - eps) * phi_vars[k] <= n_par)
                if q.response == -1:
                    m.addConstr(utz_expr - (n_par + eps) * phi_vars[k] <= -eps)
                    m.addConstr(utz_expr - (n_par - eps) * phi_vars[k] >= -n_par)
                if q.response == 0:
                    m.addConstr(utz_expr >= -eps)
                    m.addConstr(utz_expr <= eps)
                    m.addConstr(phi_vars[k] == 0)

        if u0_type == "box":
            m.setAttr("LB", u_vars, -1.0)
            m.setAttr("UB", u_vars, 1.0)

        if u0_type == "positive_normed":
            u_vars.lb = 0.0
            m.addConstr(quicksum(u_vars[i] for i in range(num_features)) == 1.0)

        m.addConstr(quicksum(phi_vars[i] for i in range(len(answered_queries))) <= gamma * len(answered_queries))

        if problem_type == "mmr":
            m.addConstr(quicksum(x_vars[i] for i in range(len(items))) == 1)

        m.update()
        m.optimize()

        if m.status == GRB.OPTIMAL:
            true_util.append(m.objVal)
        else:
            # raise("Problem not optimal!")
            true_util.append(0)

    if problem_type == "mmu":
        # select any item with max utility
        max_inds = list(np.where(np.array(true_util) == max(true_util))[0])
        # print(max_inds)

        if len(max_inds) > 1:
            return items[np.random.choice(max_inds, 1)[0]]
        else:
            return items[max_inds[0]]
    if problem_type == "mmr":
        # select any item with max utility
        min_inds = list(np.where(np.array(true_util) == min(true_util))[0])
        # print(max_inds)

        if len(min_inds) > 1:
            return items[np.random.choice(min_inds, 1)[0]]
        else:
            return items[min_inds[0]]


def recommend_item_mean(mu, items):
    """recommend an item assuming the agent's utility vector is mu"""

    true_util = np.array([np.dot(mu, item.features) for item in items])

    # select any item with max utility
    max_inds = list(np.where(np.array(true_util) == max(true_util))[0])

    if len(max_inds) > 1:
        return items[np.random.choice(max_inds)]
    else:
        return items[max_inds[0]]
