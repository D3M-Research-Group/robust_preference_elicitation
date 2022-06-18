import logging
import time
import os
import itertools
import numpy as np
import sys
from mosek.fusion import *
import gurobipy as grbpy
import copy


def get_logger(logfile=None):
    format = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # logging.basicConfig(filename=logfile, level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    # print(output_dir, output_string)
    return os.path.join(output_dir, output_string)


def get_generator_item(generator, i):
    # return the i^th item of a generator
    return next(itertools.islice(generator, i, None))


def array_to_string(x):
    if x is None:
        return str(x)
    return "[" + " ".join(["%.3e" % i for i in x]) + "]"


def generate_random_point_nsphere(n, rs=None):
    # generate a random point on the n-dimensional sphere
    if rs is None:
        rs = np.random.RandomState(0)
    x = rs.normal(size=n)
    return x / np.linalg.norm(x, ord=2)


def dist_to_point(z, p):
    # return the distance of z (a hyperplane) to the point p
    # np.linalg.norm is 2-norm, by default
    return np.abs(np.dot(p, z)) / np.linalg.norm(z)


def U0_box(num_features):
    # create the B matrix and b vector for the box u \in [-1, 1]^n
    # that is: U^0 = {u | B * u >= b}, in this case:
    B_mat = np.concatenate((np.eye(num_features), -np.eye(num_features)))
    b_vec = -np.ones(2 * num_features)
    return B_mat, b_vec


def U0_positive_normed(num_features):
    # create the B matrix and b vector for the set u \in [0,1]^n, and ||u||_1 = 1
    # that is: U^0 = {u | B * u >= b}, in this case:
    B_mat = np.concatenate(
        (
            np.eye(num_features),
            -np.eye(num_features),
            np.stack((np.repeat(1.0, num_features), np.repeat(-1.0, num_features))),
        )
    )
    b_vec = np.concatenate(
        (np.repeat(0.0, num_features), np.repeat(-1.0, num_features), [1.0], [-1.0])
    )
    return B_mat, b_vec


# def U0_positive_normed_axis(num_features):
#     # create the B matrix and b vector for the set u \in [0,1]^n, and ||u||_1 = 1
#     # that is: U^0 = {u | B * u >= b}, in this case:
#     B_mat = np.concatenate(
#         (
#             np.eye(num_features),
#             -np.eye(num_features),
#             np.stack((np.repeat(1.0, num_features), np.repeat(-1.0, num_features))),
#         )
#     )
#     b_vec = np.concatenate(
#         (np.repeat(0.0, num_features), np.repeat(-1.0, num_features), [0.9], [-1.1])
#     )
#     return B_mat, b_vec


def get_u0(u0_type, num_features):
    """return a polyhedral definition for U^0, B_mat and b_vec"""

    assert u0_type in ["box", "positive_normed"]

    if u0_type == "box":
        B_mat, b_vec = U0_box(num_features)
    if u0_type == "positive_normed":
        B_mat, b_vec = U0_positive_normed(num_features)

    return B_mat, b_vec


def find_analytic_center(answered_queries, num_features, gamma, u0_type, verbose=False):
    """use MOSEK to find the analytic center of the uncertainty set, given a set of answered queries"""

    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    assert u0_type in ["box", "positive_normed"]

    with Model("Analytic Center") as m:

        if verbose:
            m.setLogHandler(sys.stdout)

        # -- u-set variables --
        u_vars = m.variable("u", num_features, Domain.unbounded())
        xi_vars = m.variable("xi", len(answered_queries), Domain.greaterThan(0.0))
        if gamma == 0.0:
            m.constraint(xi_vars, Domain.equalsTo(0.0))

        # -- slack variables --
        # l variables - only used if there are answered_queries
        if len(answered_queries) > 0:
            l_vars = m.variable("l", len(answered_queries), Domain.greaterThan(0.0))
            log_l_vars = m.variable(
                "log(l)", len(answered_queries)
            )  # , Domain.unbounded())
            for k in range(len(answered_queries)):
                mosek_log(m, log_l_vars.index(k), l_vars.index(k))

        # w, v variables - always used
        w_vars = m.variable("w", num_features, Domain.greaterThan(0.0))
        log_w_vars = m.variable("log(w)", num_features)  # , Domain.unbounded())
        v_vars = m.variable("v", num_features, Domain.greaterThan(0.0))
        log_v_vars = m.variable("log(v)", num_features)  # , Domain.unbounded())
        for i in range(num_features):
            mosek_log(m, log_v_vars.index(i), v_vars.index(i))
            mosek_log(m, log_w_vars.index(i), w_vars.index(i))

        # p_var - only used if gamma > 0
        if gamma > 0:
            p_var = m.variable("p", Domain.greaterThan(0.0))
            log_p_var = m.variable("log(p)")  # , Domain.unbounded())
            mosek_log(m, log_p_var, p_var)

        # u-constraints defined by queries
        if len(answered_queries) > 0:
            for k, q in enumerate(answered_queries):
                assert len(q.z) == num_features
                utz_expr = Expr.dot(q.z, u_vars)
                # lhs = Var.vstack([utz_expr, xi_vars.index(k), l_vars.index(k)])
                lhs = Var.vstack([xi_vars.index(k), l_vars.index(k)])
                if q.response == 1:
                    # u^T z_k + xi_k - l_k = 0 (s_k = 1)
                    m.constraint(
                        "uz_{}".format(k),
                        Expr.add(utz_expr, Expr.dot([1, -1], lhs),),
                        Domain.equalsTo(0.0),
                    )
                if q.response == -1:
                    # u^T z_k - xi_k + l_k = 0 (s_k = -1)
                    m.constraint(
                        "uz_{}".format(k),
                        Expr.add(utz_expr, Expr.dot([-1, 1], lhs),),
                        Domain.equalsTo(0.0),
                    )

        if u0_type == "box":
            # u-constraints - within bounding box [-1, 1]^(num_features)
            # m.constraint('u + w = 1',
            #              Expr.add(u_vars, w_vars), Domain.equalsTo(1.0))
            # m.constraint('u - v = -1',
            #              Expr.sub(u_vars, v_vars), Domain.equalsTo(-1.0))
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = -1 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(-1.0),
                )

        if u0_type == "positive_normed":
            # fix the 1-norm of u
            m.constraint(
                "||u||_1 = 1", Expr.sum(u_vars), Domain.equalsTo(1.0),
            )
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = 0 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(0.0),
                )

        # constraints bounding xi
        if gamma > 0:
            m.constraint(
                "sum(xi) + p = Gamma",
                Expr.add(Expr.sum(xi_vars), p_var),
                Domain.equalsTo(gamma),
            )

        # objective value
        if len(answered_queries) > 0:
            obj_u = Expr.add(
                [Expr.sum(log_l_vars), Expr.sum(log_w_vars), Expr.sum(log_v_vars),]
            )
        else:
            obj_u = Expr.add([Expr.sum(log_w_vars), Expr.sum(log_v_vars),])

        if gamma > 0:
            obj = Expr.add(obj_u, log_p_var)
        else:
            obj = obj_u

        m.objective(ObjectiveSense.Maximize, obj)

        # optimize
        m.solve()

        sol_status = m.getPrimalSolutionStatus()

        # return the u-vars
        if sol_status == SolutionStatus.Optimal:

            return u_vars.level()

        else:

            X_mat, d_vec = get_u0(u0_type, num_features)
            for k, q in enumerate(answered_queries):
                d_vec = np.append(d_vec, 0)
                if q.response == 1:
                    X_mat = np.row_stack((X_mat, q.z))
                elif q.response == -1:
                    X_mat = np.row_stack((X_mat, -q.z))
            num_constrs = len(d_vec)

            # flag_0 = 1
            # flag_feas = 1

            for i in range(num_features):
                model_checkfeas = grbpy.Model("feas")
                model_checkfeas.params.logtoconsole = 0

                x_vars = model_checkfeas.addVars(num_features, vtype=grbpy.GRB.CONTINUOUS, name=("x"))
                d_vars = model_checkfeas.addVars(num_features, vtype=grbpy.GRB.CONTINUOUS, name=("d"))
                di_abs = model_checkfeas.addVar(vtype=grbpy.GRB.CONTINUOUS, name=("d_i_abs"))

                model_checkfeas.setObjective(di_abs, sense=grbpy.GRB.MAXIMIZE)

                for j in range(num_constrs):
                    model_checkfeas.addConstr(
                        grbpy.quicksum(X_mat[j, l] * (x_vars[l] + d_vars[l]) for l in range(num_features)) >= d_vec[j],
                        name=f"constraint_{j}"
                    )
                model_checkfeas.addConstr(di_abs >= d_vars[i], name="di_abs1")
                model_checkfeas.addConstr(di_abs >= -d_vars[i], name="di_abs2")

                model_checkfeas.update()
                model_checkfeas.optimize()

                if model_checkfeas.status != grbpy.GRB.Status.OPTIMAL:
                    if u0_type == "positive_normed":
                        return None
                    elif u0_type == "box":
                        return np.array([0] * num_features)

                else:
                    if model_checkfeas.objVal != 0:
                        raise Exception(
                            f"mosek model: infeasible; gurobi model: feasible but there exist nonzero feasible point {model_checkfeas.objVal}"
                        )

            return np.array([0] * num_features)


# MOSEK: Logarithm
# t <= log(x), x>=0
def mosek_log(M, t, x):
    M.constraint(Expr.hstack(x, 1, t), Domain.inPExpCone())


def find_analytic_center_robust(answered_queries, num_features, gamma, u0_type, verbose=False):
    """
    use MOSEK to find the analytic center of the uncertainty set, given a set of answered queries
    This function corresponds to the method used in Bertsimas and O'hair 2013
    """

    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1, 0]

    assert u0_type in ["box", "positive_normed"]

    eps = 0.1
    n_par = 10 * num_features
    big_M = 1000

    with Model("Analytic Center") as m:

        if verbose:
            m.setLogHandler(sys.stdout)

        # -- u-set variables --
        u_vars = m.variable("u", num_features, Domain.unbounded())

        # -- slack variables --
        # l, p, q variables - only used if there are answered_queries
        if len(answered_queries) > 0:
            l_vars = m.variable("l", len(answered_queries), Domain.greaterThan(0.0))
            log_l_vars = m.variable(
                "log(l)", len(answered_queries)
            )  # , Domain.unbounded())
            p_vars = m.variable("p", len(answered_queries), Domain.greaterThan(0.0))
            log_p_vars = m.variable(
                "log(p)", len(answered_queries)
            )  # , Domain.unbounded())
            q_vars = m.variable("q", len(answered_queries), Domain.greaterThan(0.0))
            log_q_vars = m.variable(
                "log(q)", len(answered_queries)
            )  # , Domain.unbounded())
            for k in range(len(answered_queries)):
                mosek_log(m, log_l_vars.index(k), l_vars.index(k))
                mosek_log(m, log_p_vars.index(k), p_vars.index(k))
                mosek_log(m, log_q_vars.index(k), q_vars.index(k))

        # w, v variables - always used
        w_vars = m.variable("w", num_features, Domain.greaterThan(0.0))
        log_w_vars = m.variable("log(w)", num_features)  # , Domain.unbounded())
        v_vars = m.variable("v", num_features, Domain.greaterThan(0.0))
        log_v_vars = m.variable("log(v)", num_features)  # , Domain.unbounded())
        for i in range(num_features):
            mosek_log(m, log_v_vars.index(i), v_vars.index(i))
            mosek_log(m, log_w_vars.index(i), w_vars.index(i))

        # phi_vars - binary
        phi_vars = m.variable("phi", len(answered_queries), Domain.binary())

        # u-constraints defined by queries
        if len(answered_queries) > 0:
            for k, q in enumerate(answered_queries):
                # print(k, q.z)
                assert len(q.z) == num_features
                utz_expr = Expr.dot(q.z, u_vars)
                # lhs = Var.vstack([utz_expr, xi_vars.index(k), l_vars.index(k)])
                # lhs = Var.vstack([xi_vars.index(k), l_vars.index(k)])
                if q.response == 1:
                    # u^T z_k + xi_k - l_k = 0 (s_k = 1)
                    m.constraint(
                        "uz1_{}".format(k),
                        Expr.sub(Expr.add(utz_expr, Expr.mul(n_par + eps, phi_vars.index(k))), p_vars.index(k)),
                        Domain.equalsTo(eps),
                    )
                    m.constraint(
                        "uz2_{}".format(k),
                        Expr.add(Expr.add(utz_expr, Expr.mul(n_par - eps, phi_vars.index(k))), q_vars.index(k)),
                        Domain.equalsTo(n_par),
                    )
                if q.response == -1:
                    # u^T z_k - xi_k + l_k = 0 (s_k = -1)
                    m.constraint(
                        "uz1_{}".format(k),
                        Expr.add(Expr.sub(utz_expr, Expr.mul(n_par + eps, phi_vars.index(k))), p_vars.index(k)),
                        Domain.equalsTo(-eps),
                    )
                    m.constraint(
                        "uz2_{}".format(k),
                        Expr.sub(Expr.sub(utz_expr, Expr.mul(n_par - eps, phi_vars.index(k))), q_vars.index(k)),
                        Domain.equalsTo(-n_par),
                    )
                if q.response == 0:
                    # u^T z_k - xi_k + l_k = 0 (s_k = -1)
                    m.constraint(
                        "uz1_{}".format(k),
                        Expr.add(utz_expr, p_vars.index(k)),
                        Domain.equalsTo(eps),
                    )
                    m.constraint(
                        "uz2_{}".format(k),
                        Expr.sub(utz_expr, q_vars.index(k)),
                        Domain.equalsTo(-eps),
                    )

            for k, q in enumerate(answered_queries):
                if q.response == 0:
                    m.constraint("phi0_{}".format(k), phi_vars.index(k), Domain.equalsTo(0.0))
                else:
                    m.constraint(
                        "phi1_{}".format(k),
                        Expr.add(Expr.sub(l_vars.index(k), p_vars.index(k)), Expr.mul(big_M, phi_vars.index(k))),
                        Domain.greaterThan(0),
                    )
                    m.constraint(
                        "phi2_{}".format(k),
                        Expr.sub(Expr.sub(l_vars.index(k), p_vars.index(k)), Expr.mul(big_M, phi_vars.index(k))),
                        Domain.lessThan(0),
                    )
                    m.constraint(
                        "phi3_{}".format(k),
                        Expr.add(Expr.sub(l_vars.index(k), q_vars.index(k)), Expr.mul(big_M, Expr.sub(1, phi_vars.index(k)))),
                        Domain.greaterThan(0),
                    )
                    m.constraint(
                        "phi4_{}".format(k),
                        Expr.sub(Expr.sub(l_vars.index(k), q_vars.index(k)), Expr.mul(big_M, Expr.sub(1, phi_vars.index(k)))),
                        Domain.lessThan(0),
                    )


        if u0_type == "box":
            # u-constraints - within bounding box [-1, 1]^(num_features)
            # m.constraint('u + w = 1',
            #              Expr.add(u_vars, w_vars), Domain.equalsTo(1.0))
            # m.constraint('u - v = -1',
            #              Expr.sub(u_vars, v_vars), Domain.equalsTo(-1.0))
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = -1 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(-1.0),
                )

        if u0_type == "positive_normed":
            # fix the 1-norm of u
            m.constraint(
                "||u||_1 = 1", Expr.sum(u_vars), Domain.equalsTo(1.0),
            )
            for i in range(num_features):
                m.constraint(
                    "u + w = 1 : {}".format(i),
                    Expr.add(u_vars.index(i), w_vars.index(i)),
                    Domain.equalsTo(1.0),
                )
                m.constraint(
                    "u - v = 0 : {}".format(i),
                    Expr.sub(u_vars.index(i), v_vars.index(i)),
                    Domain.equalsTo(0.0),
                )

        # constraints bounding phi
        m.constraint(
            "sum(phi) <= gamma * K",
            Expr.sum(phi_vars),
            Domain.lessThan(gamma * len(answered_queries)),
        )

        # objective value
        if len(answered_queries) > 0:
            obj = Expr.add(
                [Expr.sum(log_l_vars), Expr.sum(log_w_vars), Expr.sum(log_v_vars),]
            )
        else:
            obj = Expr.add([Expr.sum(log_w_vars), Expr.sum(log_v_vars),])


        m.objective(ObjectiveSense.Maximize, obj)

        # optimize
        m.solve()

        sol_status = m.getPrimalSolutionStatus()

        # return the u-vars
        if sol_status == SolutionStatus.Optimal:
            return u_vars.level()
        else:
            return np.array([0] * num_features)


def random_vector_bounded_sum_rejection(k, vector_sum, rs, max_samples=100000000):
    """
    return a uniform random vector of length k which has 1-norm <= vector_sum, with each element bounded by
     vector sum (otherwise the support of this distribution is unbounded)

    do this using rejection sampling
    """

    for i in range(max_samples):
        vec = rs.rand(int(k)) * vector_sum
        if sum(vec) <= vector_sum:
            signs = rs.choice([-1.0, 1.0], int(k))
            return vec * signs

    return Exception("could not find a vector in this distribution")


def random_vector_bounded_sum_dirichlet(k, vector_sum, rs):
    """
    return a uniform random vector of length k which has 1-norm <= vector_sum, with each element bounded by
     vector sum (otherwise the support of this distribution is unbounded)

    do this by (a) sampling the length of the vector uniformly on [0, vector_sum], and (b) drawing each element from a
    dirichlet distribution (alpha=1) and scaling each element to have sum equal to the random sum
    """
    gen = np.random.default_rng(rs.randint(100))
    vec_len = rs.rand() * vector_sum
    elements = gen.dirichlet([1.0] * int(k)) * vec_len
    signs = rs.choice([-1.0, 1.0], int(k))
    return elements * signs


def rand_fixed_sum(n, seed):
    """
    Roger Stafford's randfixedsum algorithm, adapted from original matlab code
    (https://www.mathworks.com/matlabcentral/fileexchange/9700-random-vectors-with-fixed-sum)

    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.

    each element of the vector is on [0, 1], and the sum is fixed to 1

    args:
    - n (int): length of the vector
    - seed (int): random seed
    """
    assert n > 1
    assert isinstance(n, int)

    rand_state = np.random.RandomState(seed)

    s = 1
    k = 1

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    s1 = s - np.arange(k, k - n, -1.0)
    s2 = np.arange(k + n, k, -1.0) - s

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (
            np.logical_not(tmp4)
        )

    x = np.zeros(n)
    rt = rand_state.uniform(size=(n - 1))  # rand simplex type
    rs = rand_state.uniform(size=(n - 1))  # rand position in simplex
    j = k + 1
    tmp_sum = 0.0
    tmp_prob = 1.0

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = 1 if rt[(n - i) - 1] <= t[i - 1, j - 1] else 0
        sx = rs[(n - i) - 1] ** (1.0 / i)  # next simplex coord
        tmp_sum = tmp_sum + (1.0 - sx) * tmp_prob * s / (i + 1)
        tmp_prob = sx * tmp_prob
        x[(n - i) - 1] = tmp_sum + tmp_prob * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1] = tmp_sum + tmp_prob * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    x_new = x[rand_state.permutation(n)]

    return x_new


def find_longest_axis(analytic_center, answered_queries, num_features, u0_type, items):
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    assert u0_type in ["box"]

    if not np.any(analytic_center):
        return np.random.uniform(-1, 1, len(analytic_center))

    if u0_type == "box":
        A_mat, b_vec = get_u0(u0_type, num_features)
        assert A_mat.shape[1] == num_features
    # elif u0_type == "positive_normed":
    #     A_mat, b_vec = get_u0("positive_normed_axis", num_features)

    A_mat = -A_mat
    b_vec = -b_vec  # Ax <= b rather than Bu >= b

    for k, q in enumerate(answered_queries):
        b_vec = np.append(b_vec, 0)
        if q.response == 1:
            A_mat = np.row_stack((A_mat, -q.z))
        elif q.response == -1:
            A_mat = np.row_stack((A_mat, q.z))

    hessian = np.zeros((A_mat.shape[1], A_mat.shape[1]))
    # print("ac:", analytic_center)
    for k in range(A_mat.shape[0]):
        for i in range(A_mat.shape[1]):
            for j in range(A_mat.shape[1]):
                # print(b_vec[k] - np.dot(A_mat[k], analytic_center))
                hessian[i, j] += (A_mat[k, i] * A_mat[k, j]) / (b_vec[k] - np.dot(A_mat[k], analytic_center)) ** 2

    # print("A_mat = ", A_mat)
    # print("new123", b_vec[k] - np.dot(A_mat[k], analytic_center))
    # print("hessian =", hessian)
    # if not np.all(np.linalg.eigvals(hessian) > 0):
    #     print("hessian =", hessian)
    Minv = np.linalg.inv(np.linalg.cholesky(hessian))
    cov = Minv[0:len(analytic_center), 0:len(analytic_center)]
    eigenvalue, eigenvector = np.linalg.eig(cov)

    return eigenvector[list(eigenvalue).index(max(eigenvalue))]  # return the eigenvector corrsponding to the largest eigenvalue


def find_longest_axis_pn(analytic_center, answered_queries, num_features, u0_type, items):
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    assert u0_type in ["positive_normed"]

    U_bar = np.diag(analytic_center)
    X_mat, _ = get_u0(u0_type, num_features)
    # X_mat = X_mat[0:num_features, :]
    X_mat = X_mat[2 * num_features:2 * num_features+1,]
    # np.random.seed(1)
    # answered_queries.append(Query(items[0], items[1], 1))
    # X_mat = np.row_stack((X_mat, items[0].features - items[1].features))
    for k, q in enumerate(answered_queries):
        if q.response == 1:
            X_mat = np.row_stack((X_mat, q.z))
        elif q.response == -1:
            X_mat = np.row_stack((X_mat, -q.z))
    tmp1 = []
    tmp2 = []
    for row in X_mat:
        if tuple(row) not in tmp2:
            tmp1.append(row)
        tmp2.append(tuple(row))
        tmp2.append(tuple(-row))
    X_mat = np.array(tmp1)
    # print("X =", X_mat)
    # print("X X.T =", X_mat @ X_mat.T)
    # M_mat = np.linalg.matrix_power(U_bar, -2) - X_mat.T @ np.linalg.matrix_power(X_mat @ X_mat.T, -1) @ X_mat @ np.linalg.matrix_power(U_bar, -2)
    U_bar_2 = np.linalg.matrix_power(U_bar, -2)
    # print("U_bar-2 =", U_bar_2)
    # print("middle =", np.linalg.inv(X_mat @ X_mat.T))
    # print(X_mat.T @ np.linalg.inv(X_mat @ X_mat.T) @ X_mat)
    # print("U:", U_bar)
    # print("X_mat:", X_mat)
    if np.linalg.det(X_mat @ X_mat.T):
        M_mat = np.linalg.matrix_power(U_bar, -2) - X_mat.T @ np.linalg.inv(X_mat @ X_mat.T) @ X_mat @ np.linalg.matrix_power(U_bar, -2)
    else:
        # print("Singular Matrix!")
        return generate_random_point_nsphere(num_features)
    # print("M_1st:", M_mat)
    # if not np.allclose(M_mat, M_mat.T, rtol=1e-05, atol=1e-08):
    #     print("nonX:", X_mat)
    #     print("nonsymmetric:", M_mat)
    #     M_mat = np.triu(M_mat)
    #     M_mat += M_mat.T - np.diag(M_mat.diagonal())
    # else:
    #     raise Exception("nonsymmetric")
    M_mat = np.triu(M_mat)
    M_mat += M_mat.T - np.diag(M_mat.diagonal())
    eigenvalue, eigenvector = np.linalg.eig(M_mat)
    # print(M_mat)
    # print("M =", M_mat)
    # print("eigenvalue =", eigenvalue)
    # print(eigenvalue)

    return eigenvector[list(eigenvalue).index(min(eigenvalue[eigenvalue > 0]))]



def find_two_partworth_vectors(analytic_center, longest_axis, answered_queries, num_features, u0_type):
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    assert u0_type in ["box", "positive_normed"]

    X_mat, d_vec = get_u0(u0_type, num_features)
    for k, q in enumerate(answered_queries):
        d_vec = np.append(d_vec, 0)
        if q.response == 1:
            X_mat = np.row_stack((X_mat, q.z))
        elif q.response == -1:
            X_mat = np.row_stack((X_mat, -q.z))

    tmp = []
    t = 1000000
    for i in range(X_mat.shape[0]):
        denom = np.dot(X_mat[i], longest_axis)
        if np.isclose(denom, 0):
            continue
        dist = np.absolute(d_vec[i] - np.dot(X_mat[i], analytic_center))
        t = min(t, dist / denom)

    intersections = []
    intersections.append(analytic_center + t * longest_axis)
    intersections.append(analytic_center - t * longest_axis)

    assert len(intersections) == 2

    return intersections[0], intersections[1]


def reverse_answered_queries(answered_queries, reverse):
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1]

    aq_copy = copy.deepcopy(answered_queries)

    for k, q in enumerate(aq_copy):
        if reverse[k] == 1:
            q.response = -q.response

    return aq_copy


def estimateProbPoly(answered_queries, num_features, u0_type, items, gamma):
    count = 0
    totalweight = 0
    alpha = 1 - gamma
    numqueries = len(answered_queries)

    ac = find_analytic_center(
        answered_queries, num_features, gamma, u0_type
    )
    if u0_type == "box":
        la = find_longest_axis(
            ac, answered_queries, num_features, u0_type, items
        )
    elif u0_type == "positive_normed":
        la = find_longest_axis_pn(
            ac, answered_queries, num_features, u0_type, items
        )

    Vs = [la]
    pis = [alpha ** numqueries]
    ac *= alpha ** numqueries
    totalweight += alpha ** numqueries
    count += 1

    for z in range(numqueries - 1):
        stop = False
        for op in list(itertools.combinations(list(range(numqueries)), z)):
            reverse = np.zeros(numqueries)
            reverse[list(op)] = 1
            reversed_quries = reverse_answered_queries(answered_queries, reverse)
            ac_temp = find_analytic_center(
                reversed_quries, num_features, gamma, u0_type
            )
            if ac_temp is None:
                break
            else:

                if np.all(ac_temp == 0):
                    la_temp = np.random.rand(num_features)
                else:
                    if u0_type == "box":
                        la_temp = find_longest_axis(
                            ac, reversed_quries, num_features, u0_type, items
                        )
                    elif u0_type == "positive_normed":
                        la_temp = find_longest_axis_pn(
                            ac, reversed_quries, num_features, u0_type, items
                        )
                Vs.append(la_temp)
                pis.append(alpha ** (numqueries - z) * (1 - alpha) ** z)
                ac += alpha ** (numqueries - z) * (1 - alpha) ** z * ac_temp
                totalweight += alpha ** (numqueries - z) * (1 - alpha) ** z
                count += 1
                if count > 32:
                    stop = True
                    break
        if stop:
            break

    ac /= totalweight
    Pi_mat = np.diag(pis) / totalweight
    V_mat = np.array(Vs)

    A_mat = V_mat.T @ Pi_mat @ V_mat
    if (A_mat != A_mat.T).all():
        for i in range(A_mat.shape[0]):
            for j in range(A_mat.shape[1]):
                A_mat[i, j] = A[j, i]

    eigenvalue, eigenvector = np.linalg.eig(A_mat)

    la = eigenvector[list(eigenvalue).index(max(eigenvalue))]

    return ac, la