# Implement the adaptive preference elicitation methods of "Robust Active Preference Learning", for a fixed set of items
# that constitute both the recommendation and query sets.

import numpy as np
import scipy
from gurobipy import *
from scipy import misc
from random import shuffle

from gurobi_functions import GurobiTimeLimit
from preference_classes import Query, EPS_ANSWER
from recommendation import solve_recommendation_problem
from static_elicitation import static_mip_optimal
from utils import (
    get_generator_item, dist_to_point,
    find_analytic_center,
    find_analytic_center_robust,
    find_longest_axis,
    find_longest_axis_pn,
    find_two_partworth_vectors,
    estimateProbPoly
)


def next_optimal_query_iterative(
    answered_queries, items, query_list, problem_type, gamma
):
    """
    iteratively search through all queries (except for those already answered, to find the optimal query, according to
    the problem type

    check all queries in query_list. don't check for duplicates in answered_queries
    """

    valid_responses = [-1, 1]
    assert problem_type in ["mmr", "maximin"]

    # for maximin, we want to maximize the minimum robust rec. utility. for mmr, we want to minimize the maximum regret
    if problem_type == "maximin":
        obj_sign = 1.0
    if problem_type == "mmr":
        obj_sign = -1.0

    M = 1e8
    opt_objval = -M
    next_query = None
    for q in query_list:
        answered_queries_new = answered_queries + [q]
        min_response_objval = M
        for i, r in enumerate(valid_responses):
            q.response = r
            objval, _ = solve_recommendation_problem(
                answered_queries_new, items, problem_type, gamma=gamma
            )
            if (obj_sign * objval) < min_response_objval:
                min_response_objval = obj_sign * objval

        if min_response_objval > opt_objval:
            opt_objval = min_response_objval
            next_query = q

    return (obj_sign * opt_objval), next_query


def next_optimal_query_mip(
    answered_queries,
    items,
    problem_type,
    gamma,
    time_limit=10800,
    log_problem_size=False,
    logger=None,
    u0_type="box",
):
    """
    use the static elicitation MIP to find the next optimal query to ask. the next query can be constructed using any
    of the items.
    """

    valid_responses = [-1, 1]
    assert problem_type in ["mmr", "maximin"]

    for q in answered_queries:
        assert q.item_A.id < q.item_B.id

    response_list = [q.response for q in answered_queries]

    assert set(response_list).issubset(set(valid_responses))

    scenario_list = [tuple(response_list + [r]) for r in valid_responses]

    K = len(answered_queries) + 1

    if logger is not None:
        logger.debug("calling static_mip_optimal, writing logs to file")
    queries, objval, _, _ = static_mip_optimal(
        items,
        K,
        valid_responses,
        cut_1=True,
        cut_2=False,
        fixed_queries=answered_queries,
        subproblem_list=scenario_list,
        gamma_inconsistencies=gamma,
        problem_type=problem_type,
        time_lim=time_limit,
        raise_gurobi_time_limit=False,
        log_problem_size=log_problem_size,
        logger=logger,
        u0_type=u0_type,
    )

    return objval, queries[-1]


def get_next_query_ac(answered_queries, items, gamma, u0_type):
    """return the query vector (created using the set of items), with hyperplane closest to the AC of the u-set formed
    by answered_queries"""
    num_features = len(items[0].features)
    ac = find_analytic_center(answered_queries, num_features, gamma, u0_type)

    min_dist = 9999
    query_opt = None
    for item_a, item_b in itertools.combinations(items, 2):
        if item_a.id < item_b.id:
            q = Query(item_a, item_b)
        else:
            q = Query(item_b, item_a)
        if q in answered_queries:
            continue
        z = q.z
        dist = np.dot(z, ac) / np.dot(z, z)
        if dist < min_dist:
            min_dist = dist
            query_opt = q

    return query_opt


def get_next_query_ac_robust(answered_queries, items, gamma, u0_type, rs):
    """return the query vector (created using the set of items), with hyperplane closest to the AC of the u-set formed
    by answered_queries"""
    num_features = len(items[0].features)
    ac = find_analytic_center_robust(answered_queries, num_features, gamma, u0_type)

    if not np.any(ac):
        query_opt = get_random_query(items, rs)
        return query_opt

    min_dist = 9999
    query_opt = None
    for item_a, item_b in itertools.combinations(items, 2):
        if item_a.id < item_b.id:
            q = Query(item_a, item_b)
        else:
            q = Query(item_b, item_a)
        if q in answered_queries:
            continue
        z = q.z
        dist = np.dot(z, ac) / np.dot(z, z)
        if dist < min_dist:
            min_dist = dist
            query_opt = q

    return query_opt


def get_random_query(items, rs):
    """select a random query"""
    a, b = rs.choice(len(items), 2, replace=False)
    return Query(items[min(a, b)], items[max(a, b)])


def get_next_query_polyhedral(answered_queries, items, gamma, u0_type, rs):
    """return the query vector (created using the set of items), with polyhedral method in Toubia(2004)"""
    num_features = len(items[0].features)

    ac = find_analytic_center(
        answered_queries, num_features, gamma, u0_type
        )
    if not np.any(ac):
        question = get_random_query(items, rs)
        return question
    if ac is None:
        raise Exception(
            "Uncertainty set is infeasible"
        )
    if u0_type == "box":
        la = find_longest_axis(
            ac, answered_queries, num_features, u0_type, items
        )
    elif u0_type == "positive_normed":
        la = find_longest_axis_pn(
            ac, answered_queries, num_features, u0_type, items
        )
    partworth_1, partworth_2 = find_two_partworth_vectors(
        ac, la, answered_queries, num_features, u0_type
    )
    question_found = 0
    it = 0

    # test_tmp = [np.dot(item_tmp.features, ac) for item_tmp in items]
    # print(test_tmp)
    # print("mean:", sum(test_tmp) / len(test_tmp))
    shuffled_list = list(range(len(items)))

    while question_found == 0 and it < 10:
        it += 1
        for i in range(101):
            np.random.seed(i + it * 100)
            if u0_type == "box":
                M = np.random.uniform(-1, 10)
            if u0_type == "positive_normed":
                M = np.random.uniform(-0.25, 0.75)
                # M = np.random.uniform(-1, 10)
            q = []
            for j, partworth in enumerate([partworth_1, partworth_2]):
                shuffle(shuffled_list)
                item = items[shuffled_list[0]]
                if j == 0:
                    # shuffle(shuffled_list)
                    # for item_tmp in items:
                    for k in shuffled_list:
                        # print(np.dot(items[k].features, ac))
                        if np.dot(items[k].features, ac) <= M:
                            if np.dot(items[k].features, partworth) > np.dot(item.features, partworth):
                                item = items[k]
                    q.append(item)
                else:
                    # shuffle(shuffled_list)
                    # if q[0] == items[0]:
                    #     item = items[1]
                    if q[0] == items[shuffled_list[0]]:
                        item = items[shuffled_list[1]]
                    # for item_tmp in items:
                    for k in shuffled_list:
                        if items[k] != q[0]:
                            # print(np.dot(items[k].features, ac))
                            if np.dot(items[k].features, ac) <= M:
                                if np.dot(items[k].features, partworth) > np.dot(item.features, partworth):
                                    item = items[k]
                    q.append(item)
            if q[0].id == q[1].id:
                continue
            elif q[0].id < q[1].id:
                question = Query(q[0], q[1])
            else:
                question = Query(q[1], q[0])
            if question in answered_queries:
                continue
            question_found == 1

    # print(question.item_A.id, question.item_B.id)
    return question


def get_next_query_probpoly(answered_queries, items, gamma, u0_type):
    """return the query vector (created using the set of items), with probabilistic polyhedral method in Toubia(2007)"""
    num_features = len(items[0].features)

    ac, la = estimateProbPoly(answered_queries, num_features, u0_type, items, gamma)

    partworth_1, partworth_2 = find_two_partworth_vectors(
        ac, la, answered_queries, num_features, u0_type
    )
    question_found = 0
    it = 0

    # test_tmp = [np.dot(item_tmp.features, ac) for item_tmp in items]
    # print(test_tmp)
    # print("mean:", sum(test_tmp) / len(test_tmp))
    shuffled_list = list(range(len(items)))

    while question_found == 0 and it < 10:
        it += 1

        for i in range(101):
            np.random.seed(i + it * 100)
            if u0_type == "box":
                M = np.random.uniform(-1, 10)
            if u0_type == "positive_normed":
                M = np.random.uniform(-0.25, 0.75)
            q = []
            for j, partworth in enumerate([partworth_1, partworth_2]):
                shuffle(shuffled_list)
                item = items[shuffled_list[0]]
                if j == 0:
                    # shuffle(shuffled_list)
                    # for item_tmp in items:
                    for k in shuffled_list:
                        if np.dot(items[k].features, ac) <= M:
                            if np.dot(items[k].features, partworth) > np.dot(item.features, partworth):
                                item = items[k]
                    q.append(item)
                else:
                    # shuffle(shuffled_list)
                    # if q[0] == items[0]:
                    #     item = items[1]
                    if q[0] == items[shuffled_list[0]]:
                        item = items[shuffled_list[1]]
                    # for item_tmp in items:
                    for k in shuffled_list:
                        if items[k] != q[0]:
                            if np.dot(items[k].features, ac) <= M:
                                if np.dot(items[k].features, partworth) > np.dot(item.features, partworth):
                                    item = items[k]
                    q.append(item)
            if q[0].id == q[1].id:
                continue
            elif q[0].id < q[1].id:
                question = Query(q[0], q[1])
            else:
                question = Query(q[1], q[0])
            if question in answered_queries:
                continue
            question_found == 1

    return question
