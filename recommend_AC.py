import numpy as np
from gurobipy import *
from preference_classes import Agent, Query, Item, generate_items
from gurobi_functions import create_mip_model, optimize


def recommend_item_ac_robust(answered_queries, items, problem_type, gamma, u0_type, verbose=False, logger=None):
    """make a recommendation to the agent, assuming their u-vector is their current AC"""
    # validate input
    for q in answered_queries:
        assert q.response in [-1, 1, 0]

    assert u0_type in ["box", "positive_normed"]
    assert problem_type in ["mmu", "mmr"]

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

        # m.setObjective(0, sense=GRB.MINIMIZE)
        
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
            # u_vars.lb = -1.0
            # u_vars.ub = 1.0
            m.setAttr("LB", u_vars, -1.0)
            m.setAttr("UB", u_vars, 1.0)
            # for i in range(num_features):
            #     m.addConstr(u_vars[i] >= -1.0)
            #     m.addConstr(u_vars[i] <= 1.0)

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
            raise("Problem not optimal!")

    # select any item with max utility
    max_inds = list(np.where(np.array(true_util) == max(true_util))[0])
    print(max_inds)

    if len(max_inds) > 1:
        return items[np.random.choice(max_inds, 1)[0]]
    else:
        return items[max_inds[0]]


verbose = True
gamma = 0.2
u0_type = "box"
num_features = 10
num_items = 30

agent_sphere_size = 1.0
item_sphere_size = 10.0
agent_seed = 1
problem_seed = 1
agent = Agent.random(num_features, id=agent_seed, sphere_size=agent_sphere_size, seed=agent_seed,)
items = generate_items(
    num_features,
    num_items,
    item_sphere_size=item_sphere_size,
    seed=problem_seed,
)
# for i in range(10):
#     rs = np.random.RandomState(i)
#     a, b = rs.choice(len(items), 2, replace=False)
#     q =  Query(items[min(a, b)], items[max(a, b)])
#     agent.answer_query(q, error=0.0)

rec = recommend_item_ac_robust(agent.answered_queries, items, "mmr", gamma, u0_type, verbose=verbose)
print(rec.id)
print(len(items))