import argparse
import time

import numpy as np
from gurobipy import *

from preference_classes import Query, Agent, generate_items
from read_csv_to_items import get_data_items

from utils import get_u0


items = get_data_items(
    "AdultHMIS_20210906_preprocessed_final_Robust_41.csv", max_items=50, standardize_features=False, normalize_features=True, drop_cols=["IsInterpretable_int"]
)
num_features = len(items[0].features)
# num_features = 5
# items = generate_items(
#     num_features,
#     10,
#     item_sphere_size=10,
#     seed=3
# )
B_mat, b_vec = get_u0("positive_normed", num_features)
# B_mat, b_vec = get_u0("box", num_features)

ubs = []
for item in items:
    m = Model("ub")
    m.params.LogToConsole = 0

    u_vars = m.addVars(num_features, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")

    # objective
    m.setObjective(quicksum(item.features[i] * u_vars[i] for i in range(num_features)), sense=GRB.MAXIMIZE)

    # add U0 constraints
    for j in range(len(b_vec)):
        m.addConstr(
            quicksum(B_mat[j, i] * u_vars[i] for i in range(num_features)) >= b_vec[j]
        )

    m.optimize()

    ubs.append(m.objVal)

lbs = []
for item in items:
    m = Model("lb")
    m.params.LogToConsole = 0

    u_vars = m.addVars(num_features, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")

    # objective
    m.setObjective(quicksum(item.features[i] * u_vars[i] for i in range(num_features)), sense=GRB.MINIMIZE)

    # add U0 constraints
    for j in range(len(b_vec)):
        m.addConstr(
            quicksum(B_mat[j, i] * u_vars[i] for i in range(num_features)) >= b_vec[j]
        )

    m.optimize()

    lbs.append(m.objVal)

print(ubs, max(ubs))
print(lbs, min(lbs))