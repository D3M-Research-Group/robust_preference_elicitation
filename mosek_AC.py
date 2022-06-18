from mosek.fusion import *
import numpy as np
import sys
from preference_classes import Agent, generate_items, Query


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
    big_M = 100

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
            raise Exception("mosek model: infeasible")


# verbose = True
# gamma = 0.2
# u0_type = "box"
# num_features = 5
# num_items = 30

# agent_sphere_size = 1.0
# item_sphere_size = 10.0
# agent_seed = 1
# problem_seed = 1
# agent = Agent.random(num_features, id=agent_seed, sphere_size=agent_sphere_size, seed=agent_seed,)
# items = generate_items(
#     num_features,
#     num_items,
#     item_sphere_size=item_sphere_size,
#     seed=problem_seed,
# )
# for i in range(20):
#     rs = np.random.RandomState(i)
#     a, b = rs.choice(len(items), 2, replace=False)
#     q =  Query(items[min(a, b)], items[max(a, b)])
#     agent.answer_query(q, error=0.2)

# ac = find_analytic_center_robust(agent.answered_queries, num_features, gamma, u0_type, verbose=verbose)
# print(ac)