# compare the static elicitation heuristic to many random samples, on different problems. output each random sample and
# the associated objective. finally, output two rows with the probabilities of random beating the heuristic
# should replace all gamma with sigma

import argparse
import time
from collections import namedtuple

import numpy as np
from gurobipy import *
from scipy.special import erfinv

from get_bounds import get_mmu_ub, get_mmu_lb, get_mmr_ub, get_mmr_lb
from preference_classes import Query, generate_items
from scenario_decomposition import (
    evaluate_query_list_decomp,
    solve_scenario_decomposition,
)
from static_heuristics import solve_warm_start_decomp_heuristic
from utils import generate_filepath, get_logger


def experiment(args):
    """
    compare the static incremental heuristic to many different sets of random queries

    record the problem instance (problem_seed) and, for each method:
    - obj. val (worst-case rec. utility)
    - normalized utility of the recommended item (on [0, 1])
    - query list (list of tuples containing item indices)
    - probability that random wins (only for the final aggregate result row)
    - all data needed to re-create the problem instance

    args must have the following parameters set:
    - problem_seed: (int).  random seed for generating the problem instance
    - num_features (int).  number of item features
    - num_items (int).  number of items
    - num_random_samples (int). number of sets of random queries to compare to
    - max_K: (int). total number of queries. results will be reported for all K = 1, ..., max_K.
    - output_dir: (str). output directory where result file will be created
    """

    # some fixed parameters
    valid_responses = [1, -1]
    item_sphere_size = 10.0
    query_seed = 0
    p_confidence = 0.9

    if args.u0_type == "positive_normed":
        # have not corrected the objval normalization for this
        raise NotImplemented

    rs = np.random.RandomState(0)

    # generate an output file
    output_file = generate_filepath(args.output_dir, "static_experiment", "csv")
    log_file = generate_filepath(args.output_dir, "static_experiment_LOGS", "txt")
    logger = get_logger(logfile=log_file)

    logger.info("generating output file: {}".format(output_file))
    logger.info("generating log file: {}".format(log_file))

    # param_sets = [(args.num_items[0], args.num_features[0], args.gamma[0])]

    # # create additional parameter sets
    # if len(args.num_items) > 0:
    #     for num_items in args.num_items[1:]:
    #         param_sets.append((num_items, args.num_features[0], args.gamma[0]))

    # if len(args.num_features) > 0:
    #     for num_features in args.num_features[1:]:
    #         param_sets.append((args.num_items[0], num_features, args.gamma[0]))

    # if len(args.gamma) > 0:
    #     for gamma in args.gamma[1:]:
    #         param_sets.append((args.num_items[0], args.num_features[0], gamma))

    param_sets = (
        args.param_sets
    )  # list of tuples of the form (num_items, num_features, sigma)
    print(param_sets)
    # print(set(param_sets))

    # remove duplicate parameter sets
    col_list = [
        "problem_seed",
        "num_features",
        "num_items",
        "item_sphere_size",
        "method",
        "query_seed",
        "K",
        "gamma",
        "gamma_normalized",
        "method_objval",
        "mmu_objval",
        "mmu_objval_ub",
        "mmu_objval_lb",
        "mmr_objval",
        "mmr_objval_ub",
        "mmr_objval_lb",
        "mmu_prob_random_wins",
        "mmr_prob_random_wins",
        "query_list",
        "solve_time",
        "mmu_objective_eval_time",
        "mmr_objective_eval_time",
    ]
    # param_sets_deduped = list(set(param_sets))
    param_sets_deduped = param_sets

    # write the file header: write experiment parameters to the first line of the output file
    delimiter = ";"
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((delimiter.join(len(col_list) * ["%s"]) + "\n") % tuple(col_list))

    # store all results in a namedtuple
    Result = namedtuple("Result", col_list)

    def result_to_str(result):
        """return a string representation of a result, for writing to a csv"""
        result_str = (delimiter.join(len(col_list) * ["{}"]) + "\n").format(
            result.problem_seed,
            result.num_features,
            result.num_items,
            result.item_sphere_size,
            result.method,
            result.query_seed,
            result.K,
            result.gamma,
            result.gamma_normalized,
            result.method_objval,
            result.mmu_objval,
            result.mmu_objval_ub,
            result.mmu_objval_lb,
            result.mmr_objval,
            result.mmr_objval_ub,
            result.mmr_objval_lb,
            result.mmu_prob_random_wins,
            result.mmr_prob_random_wins,
            result.query_list,
            result.solve_time,
            result.mmu_objective_eval_time,
            result.mmr_objective_eval_time,
        )
        return result_str

    def get_results(
        queries,
        name,
        items,
        num_features,
        method_objval_list,
        mmu_heuristic_objval_dict,
        mmr_heuristic_objval_dict,
        solve_times,
        gamma,
        gamma_normalized,
        query_seed=None,
    ):
        assert len(queries) == args.max_K
        results = []
        for k in range(args.max_K):
            # find the actual obj. val of the first k queries
            query_subset = queries[: (k + 1)]

            if args.evaluate_mmu_objective:
                t0 = time.time()
                logger.debug("evaluating query list for MMU")
                mmu_objval = evaluate_query_list_decomp(
                    query_subset,
                    items,
                    valid_responses,
                    args.u0_type,
                    gamma_inconsistencies=gamma_normalized,
                    problem_type="maximin",
                    logger=logger,
                )
                mmu_objective_eval_time = time.time() - t0
                mmu_random_wins = (
                    1 if mmu_objval > mmu_heuristic_objval_dict.get(k + 1, -999) else 0
                )

                # find the UB and LB for the MMU objval
                mmu_objval_ub = get_mmu_ub(items, args.u0_type)
                mmu_objval_lb = get_mmu_lb(items, args.u0_type)
            else:
                mmu_objval = None
                mmu_objval_ub = None
                mmu_objval_lb = None
                mmu_objective_eval_time = None
                mmu_random_wins = None

            if args.evaluate_mmr_objective:
                t0 = time.time()
                logger.debug("evaluating query list for MMR")
                mmr_objval = evaluate_query_list_decomp(
                    query_subset,
                    items,
                    valid_responses,
                    args.u0_type,
                    gamma_inconsistencies=gamma_normalized,
                    problem_type="mmr",
                    logger=logger,
                )
                mmr_objective_eval_time = time.time() - t0
                mmr_random_wins = (
                    1 if mmr_objval > mmr_heuristic_objval_dict.get(k + 1, -999) else 0
                )

                # find the UB and LB for the MMR objval
                mmr_objval_ub = get_mmr_ub(items, args.u0_type)
                mmr_objval_lb = get_mmr_lb(items, args.u0_type)

            else:
                mmr_objval = None
                mmr_objval_ub = None
                mmr_objval_lb = None
                mmr_objective_eval_time = None
                mmr_random_wins = None

            results.append(
                Result(
                    problem_seed=args.problem_seed,
                    num_features=num_features,
                    num_items=len(items),
                    item_sphere_size=item_sphere_size,
                    method=name,
                    query_seed=query_seed,
                    K=k + 1,
                    gamma=gamma,
                    gamma_normalized=gamma_normalized,
                    method_objval=method_objval_list[k],
                    mmu_objval=mmu_objval,
                    mmu_objval_ub=mmu_objval_ub,
                    mmu_objval_lb=mmu_objval_lb,
                    mmr_objval=mmr_objval,
                    mmr_objval_ub=mmr_objval_ub,
                    mmr_objval_lb=mmr_objval_lb,
                    mmu_prob_random_wins=mmu_random_wins,
                    mmr_prob_random_wins=mmr_random_wins,
                    query_list=[q.to_tuple() for q in query_subset],
                    solve_time=solve_times[k],
                    mmu_objective_eval_time=mmu_objective_eval_time,
                    mmr_objective_eval_time=mmr_objective_eval_time,
                )
            )
        return results

    def get_single_result(
        queries,
        name,
        items,
        num_features,
        method_objval,
        solve_time,
        gamma,
        gamma_normalized,
        k,
        query_seed=None,
        mmu_heuristic_objval_dict={},
        mmr_heuristic_objval_dict={},
    ):
        assert len(queries) == k

        # find the actual obj. val of the first k queries
        query_subset = queries[: (k + 1)]

        if args.evaluate_mmu_objective:
            logger.debug("evaluating query list for MMU")
            t0 = time.time()
            mmu_objval = evaluate_query_list_decomp(
                query_subset,
                items,
                valid_responses,
                args.u0_type,
                gamma_inconsistencies=gamma_normalized,
                problem_type="maximin",
                logger=logger,
            )
            mmu_objective_eval_time = time.time() - t0
            mmu_random_wins = (
                1 if mmu_objval > mmu_heuristic_objval_dict.get(k + 1, -999) else 0
            )

            # find the UB and LB for the MMU objval
            mmu_objval_ub = get_mmu_ub(items, args.u0_type)
            mmu_objval_lb = get_mmu_lb(items, args.u0_type)

        else:
            mmu_objval = None
            mmu_objval_lb = None
            mmu_objval_ub = None
            mmu_objective_eval_time = None
            mmu_random_wins = None

        if args.evaluate_mmr_objective:
            # find the actual obj. val of the first k queries
            query_subset = queries[: (k + 1)]
            t0 = time.time()
            logger.debug("evaluating query list for MMR")
            mmr_objval = evaluate_query_list_decomp(
                query_subset,
                items,
                valid_responses,
                args.u0_type,
                gamma_inconsistencies=gamma_normalized,
                problem_type="mmr",
                logger=logger,
            )
            mmr_objective_eval_time = time.time() - t0
            mmr_random_wins = (
                1 if mmr_objval > mmr_heuristic_objval_dict.get(k + 1, -999) else 0
            )

            # find the UB and LB for the MMR objval
            mmr_objval_ub = get_mmr_ub(items, args.u0_type)
            mmr_objval_lb = get_mmr_lb(items, args.u0_type)

        else:
            mmr_objval = None
            mmr_objval_ub = None
            mmr_objval_lb = None
            mmr_objective_eval_time = None
            mmr_random_wins = None

        result = Result(
            problem_seed=args.problem_seed,
            num_features=num_features,
            num_items=len(items),
            item_sphere_size=item_sphere_size,
            method=name,
            query_seed=query_seed,
            K=k,
            gamma=gamma,
            gamma_normalized=gamma_normalized,
            method_objval=method_objval,
            mmu_objval=mmu_objval,
            mmu_objval_ub=mmu_objval_ub,
            mmu_objval_lb=mmu_objval_lb,
            mmr_objval=mmr_objval,
            mmr_objval_ub=mmr_objval_ub,
            mmr_objval_lb=mmr_objval_lb,
            mmu_prob_random_wins=mmu_random_wins,
            mmr_prob_random_wins=mmr_random_wins,
            query_list=[q.to_tuple() for q in query_subset],
            solve_time=solve_time,
            mmu_objective_eval_time=mmu_objective_eval_time,
            mmr_objective_eval_time=mmr_objective_eval_time,
        )
        return result

    # keep track of all Result objects in a list
    result_list = []

    for num_items, num_features, gamma_unnormalized in param_sets_deduped:

        num_items = int(num_items)
        num_features = int(num_features)

        logger.info(
            "starting parameter set: n-items={}, n-features={}, gamma={}".format(
                num_items, num_features, gamma_unnormalized
            )
        )

        # keep track of all Results from this parameter set
        param_results = []

        logger.info("creating items")
        items = generate_items(
            num_features,
            num_items,
            item_sphere_size=item_sphere_size,
            seed=args.problem_seed,
        )

        # get the max 1-norm of all items
        max_norm = max([np.sum(np.abs(i.features)) for i in items])
        logger.info(f"max item norm = {max_norm}")

        # get the max 1-norm *difference* between items (for mmr)
        max_diff_norm = max(
            [
                np.sum(np.abs(i.features - j.features))
                for i, j in itertools.combinations(items, 2)
            ]
        )
        logger.info(f"max item-diff norm = {max_diff_norm}")

        # alpha for normalizing gamma
        inv_alpha = 0.5

        # ------------------ optimal methods ------------------

        # mmu-opt
        if args.use_mmu_optimal:
            if gamma_unnormalized == 0:
                name = "mmu_opt"
                for k in args.opt_K:
                    logger.info("running MMU-opt (gamma=0, k={})".format(k))
                    t0 = time.time()
                    opt_queries, objval, status = solve_scenario_decomposition(
                        items,
                        k,
                        rs,
                        valid_responses,
                        args.u0_type,
                        max_iter=1000000000,
                        cut_1=True,
                        cut_2=True,
                        start_queries=None,
                        time_limit=args.time_limit,
                        problem_type="maximin",
                        logger=logger,
                    )
                    assert status != "no_incumbent"
                    runtime = time.time() - t0
                    logger.info("finished MMU-opt")
                    logger.info(r"collecting results for MMU-opt (k={})".format(k))
                    assert len(opt_queries) == k
                    result = get_single_result(
                        opt_queries,
                        name,
                        items,
                        num_features,
                        objval,
                        runtime,
                        0.0,
                        0.0,
                        k,
                    )
                    logger.info("writing results to file")
                    with open(output_file, "a") as f:
                        f.write(result_to_str(result))

            else:
                raise NotImplemented

        # mmr-opt
        if args.use_mmr_optimal:
            if gamma_unnormalized == 0:
                name = "mmr_opt"
                for k in args.opt_K:
                    logger.info("running MMR-opt (gamma=0, k={})".format(k))
                    t0 = time.time()
                    opt_queries, objval, status = solve_scenario_decomposition(
                        items,
                        k,
                        rs,
                        valid_responses,
                        args.u0_type,
                        max_iter=1000000000,
                        cut_1=True,
                        cut_2=True,
                        start_queries=None,
                        time_limit=args.time_limit,
                        problem_type="mmr",
                        logger=logger,
                    )
                    assert status != "no_incumbent"
                    runtime = time.time() - t0
                    logger.info("finished MMU-opt")
                    logger.info(r"collecting results for MMR-opt (k={})".format(k))
                    assert len(opt_queries) == k
                    result = get_single_result(
                        opt_queries,
                        name,
                        items,
                        num_features,
                        objval,
                        runtime,
                        0.0,
                        0.0,
                        k,
                    )
                    logger.info("writing results to file")
                    with open(output_file, "a") as f:
                        f.write(result_to_str(result))
            else:
                raise NotImplemented

        # ------------------ maximin utility: warm start heuristic with scenario decomposition ------------------
        mmu_heuristic_objval_dict = {}
        if args.use_mmu_heuristic:
            if gamma_unnormalized == 0:

                # warm start heuristic with scenario decomposition. with gamma = 0, we will use the optimal (heuristic)
                # solution from k-1 to build the heuristic solution to k.
                logger.info("running MMU heuristic (gamma=0)")
                (
                    heuristic_queries,
                    objval_list,
                    _,
                    incremental_times,
                ) = solve_warm_start_decomp_heuristic(
                    items,
                    args.max_K,
                    valid_responses,
                    args.u0_type,
                    logger=logger,
                    time_lim=args.time_limit,
                    time_lim_overall=False,
                    problem_type="maximin",
                    return_incremental_times=True,
                    gamma_inconsistencies=0,
                )
                logger.info("finished MMU heuristic")
                logger.info("collecting results for MMU heuristic")
                heuristic_results = get_results(
                    heuristic_queries,
                    "mmu_heuristic",
                    items,
                    num_features,
                    objval_list,
                    {},
                    {},
                    incremental_times,
                    0.0,
                    0.0,
                )

            else:

                # for gamma > 0, the bound on inconsistencies changes with k, so we need to re-run the warm start heuristic
                # for each k.
                heuristic_results = []
                for k in range(1, args.max_K + 1):
                    # TODO: normalize gamma base on given sigma
                    # gamma = 2.0 * gamma_unnormalized * np.power(k, inv_alpha) * max_norm
                    sigma_hat = np.sqrt(2.0 * k * (gamma_unnormalized ** 2))
                    gamma = (
                        sigma_hat * np.sqrt(2) * erfinv(2.0 * p_confidence - 1.0)
                    )

                    logger.info("running MMU heuristic with k={} (gamma>0)".format(k))
                    t0 = time.time()
                    heuristic_queries, objval, _ = solve_warm_start_decomp_heuristic(
                        items,
                        k,
                        valid_responses,
                        args.u0_type,
                        logger=logger,
                        time_lim=args.time_limit,
                        time_lim_overall=False,
                        gamma_inconsistencies=gamma,
                        problem_type="maximin",
                    )
                    logger.info(
                        "finished MMU heuristic with k={}. collecting results...".format(
                            k
                        )
                    )
                    heuristic_time = time.time() - t0
                    heuristic_results.append(
                        get_single_result(
                            heuristic_queries,
                            "mmu_heuristic",
                            items,
                            num_features,
                            objval,
                            heuristic_time,
                            gamma_unnormalized,
                            gamma,
                            k,
                        )
                    )

            mmu_heuristic_objval_dict = {r.K: r.mmu_objval for r in heuristic_results}

            logger.info("writing MMU heuristic results to file")
            with open(output_file, "a") as f:
                for result in heuristic_results:
                    f.write(result_to_str(result))

        # ------------------ minimax regret: warm start heuristic with scenario decomposition ------------------
        mmr_heuristic_objval_dict = {}
        if args.use_mmr_heuristic:
            if gamma_unnormalized == 0:

                # warm start heuristic with scenario decomposition. with gamma = 0, we will use the optimal (heuristic)
                # solution from k-1 to build the heuristic solution to k.
                logger.info("running MMR heuristic (gamma=0)")
                (
                    heuristic_queries,
                    objval_list,
                    _,
                    incremental_times,
                ) = solve_warm_start_decomp_heuristic(
                    items,
                    args.max_K,
                    valid_responses,
                    args.u0_type,
                    logger=logger,
                    time_lim=args.time_limit,
                    time_lim_overall=False,
                    return_incremental_times=True,
                    problem_type="mmr",
                    gamma_inconsistencies=0,
                )
                logger.info("finished MMR heuristic")
                logger.info("collecting results for MMR heuristic")
                heuristic_results = get_results(
                    heuristic_queries,
                    "mmr_heuristic",
                    items,
                    num_features,
                    objval_list,
                    {},
                    {},
                    incremental_times,
                    0.0,
                    0.0,
                )

            else:

                # for gamma > 0, the bound on inconsistencies changes with k, so we need to re-run the warm start heuristic
                # for each k.
                heuristic_results = []
                for k in range(1, args.max_K + 1):
                    # TODO: normalize gamma base on given sigma
                    # gamma = 2.0 * gamma_unnormalized * np.power(k, inv_alpha) * max_norm
                    sigma_hat = np.sqrt(2.0 * k * (gamma_unnormalized ** 2))
                    gamma = (
                        sigma_hat * np.sqrt(2) * erfinv(2.0 * p_confidence - 1.0)
                    )

                    logger.info("running MMR heuristic with k={} (gamma>0)".format(k))
                    t0 = time.time()
                    (
                        heuristic_queries,
                        objval_list,
                        _,
                    ) = solve_warm_start_decomp_heuristic(
                        items,
                        k,
                        valid_responses,
                        args.u0_type,
                        logger=logger,
                        time_lim=args.time_limit,
                        time_lim_overall=False,
                        gamma_inconsistencies=gamma,
                        problem_type="mmr",
                    )
                    logger.info(
                        "finished MMR heuristic with k={}. collecting results...".format(
                            k
                        )
                    )
                    heuristic_time = time.time() - t0
                    heuristic_results.append(
                        get_single_result(
                            heuristic_queries,
                            "mmr_heuristic",
                            items,
                            num_features,
                            objval_list,
                            heuristic_time,
                            gamma_unnormalized,
                            gamma,
                            k,
                        )
                    )

            mmr_heuristic_objval_dict = {r.K: r.mmr_objval for r in heuristic_results}

            logger.info("writing MMR heuristic results to file")
            with open(output_file, "a") as f:
                for result in heuristic_results:
                    f.write(result_to_str(result))

        # ------------------ random queries ------------------
        if args.num_random_samples > 0:
            logger.info("running random queries")
            random_results = []
            for i in range(args.num_random_samples):
                random_queries, _ = generate_random_queries(
                    items, args.max_K, query_seed
                )

                for k in range(1, args.max_K + 1):
                    # TODO: normalize gamma base on given sigma
                    # gamma = 2.0 * gamma_unnormalized * np.power(k, inv_alpha) * max_norm
                    sigma_hat = np.sqrt(2.0 * k * (gamma_unnormalized ** 2))
                    gamma = (
                        sigma_hat * np.sqrt(2) * erfinv(2.0 * p_confidence - 1.0)
                    )
                    query_list = random_queries[:k]
                    logger.info(
                        "evaluating random queries sample num={}, k={}...".format(
                            i + 1, k
                        )
                    )
                    random_results.append(
                        get_single_result(
                            query_list,
                            "random",
                            items,
                            num_features,
                            0,
                            0,
                            gamma_unnormalized,
                            gamma,
                            k,
                            mmu_heuristic_objval_dict=mmu_heuristic_objval_dict,
                            mmr_heuristic_objval_dict=mmr_heuristic_objval_dict,
                        )
                    )

                query_seed += 1

            logger.info("writing random results results to file")
            with open(output_file, "a") as f:
                for result in random_results:
                    f.write(result_to_str(result))

            logger.info("finished generating random queries")

            # add an aggregate result dict for the random queries
            agg_results = []
            for k in range(args.max_K):
                rand_results = [
                    r
                    for r in random_results
                    if ((r.method == "random") and (r.K == (k + 1)))
                ]
                assert len(rand_results) == args.num_random_samples

                mmu_num_rand_wins = len(
                    [r for r in rand_results if r.mmu_prob_random_wins == 1.0]
                )
                mmu_prob_random_wins = float(mmu_num_rand_wins) / float(
                    args.num_random_samples
                )

                if args.evaluate_mmr_objective:
                    mmr_num_rand_wins = len(
                        [r for r in rand_results if r.mmr_prob_random_wins == 1.0]
                    )
                    mmr_prob_random_wins = float(mmr_num_rand_wins) / float(
                        args.num_random_samples
                    )
                else:
                    mmr_num_rand_wins = 0
                    mmr_prob_random_wins = 0

                agg_results.append(
                    Result(
                        problem_seed=args.problem_seed,
                        num_features=num_features,
                        num_items=len(items),
                        item_sphere_size=item_sphere_size,
                        method="agg_random",
                        query_seed=None,
                        K=k + 1,
                        gamma=gamma_unnormalized,
                        gamma_normalized=gamma,
                        method_objval=0,
                        mmu_objval=0,
                        mmu_objval_ub=0,
                        mmu_objval_lb=0,
                        mmr_objval=0,
                        mmr_objval_ub=0,
                        mmr_objval_lb=0,
                        mmu_prob_random_wins=mmu_prob_random_wins,
                        mmr_prob_random_wins=mmr_prob_random_wins,
                        query_list=[],
                        solve_time=0.0,
                        mmu_objective_eval_time=0.0,
                        mmr_objective_eval_time=0.0,
                    )
                )

            logger.info("writing aggregated random results results to file")
            with open(output_file, "a") as f:
                for result in agg_results:
                    f.write(result_to_str(result))

    logger.info("done.")


def generate_random_queries(items, K, query_seed):
    t0 = time.time()
    rs = np.random.RandomState(query_seed)

    query_list = list(itertools.combinations(items, 2))

    query_inds = rs.choice(len(query_list), K, replace=False)
    queries = [Query(query_list[i][0], query_list[i][1]) for i in query_inds]

    # generate times (just divide the time evenly between each query...)
    times = np.cumsum([(time.time() - t0) / float(K)] * K)

    return queries, times


def main():
    parser = argparse.ArgumentParser(
        description="static experiment comparing optimal heuristic to random "
    )

    parser.add_argument(
        "--max-K", type=int, help="total number of queries to ask", default=3
    )
    parser.add_argument(
        "--opt-K",
        type=int,
        nargs="+",
        help="total number of queries to ask using the optimal methods",
        default=3,
    )
    parser.add_argument(
        "--problem-seed",
        type=int,
        help="random seed for generating the problem instances",
        default=0,
    )
    parser.add_argument(
        "-p",
        "--param-sets",
        action="append",
        type=float,
        nargs="+",
        help="add a parameter set of the form (num_items, num_features, sigma)",
    )
    # parser.add_argument(
    #     "--gamma",
    #     type=float,
    #     nargs="+",
    #     default=0.0,
    #     help="level of agent inconsistencies. if a list is provided, use all of them, while fixing other "
    #     "experiment parameters",
    # )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1e10,
        help="time limit (in seconds) allowed for each stage (k) of the warm-start heuristics, and for "
        "the entire optimal method run",
    )
    # parser.add_argument(
    #     "--num-items",
    #     type=int,
    #     default=20,
    #     nargs="+",
    #     help="number of items in the problem instance. if a list is provided, use all of them, while "
    #     "fixing other experiment parameters",
    # )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=0,
        help="number of sets of random queries to compare to",
    )
    # parser.add_argument(
    #     "--num-features",
    #     type=int,
    #     default=6,
    #     nargs="+",
    #     help="number of features. if a list is provided, use all of them, while fixing other experiment"
    #     "parameters",
    # )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument(
        "--use-mmr-heuristic",
        action="store_true",
        help="if set, run minimax regret heuristic method",
    )
    parser.add_argument(
        "--use-mmu-heuristic",
        action="store_true",
        help="if set, run maximin-utility heuristic method",
    )
    parser.add_argument(
        "--use-mmu-optimal",
        action="store_true",
        help="if set, run maximin-utility optimal method",
    )
    parser.add_argument(
        "--use-mmr-optimal",
        action="store_true",
        help="if set, run minimax-regret optimal method",
    )
    parser.add_argument(
        "--evaluate-mmr-objective",
        action="store_true",
        help="if set, evaluate the minimax-regret objective",
    )
    parser.add_argument(
        "--evaluate-mmu-objective",
        action="store_true",
        help="if set, evaluate the maximin utility objective",
    )
    parser.add_argument(
        "--u0-type",
        type=str,
        default="box",
        help="type of initial uncertainty set to use {'box' | 'positive_normed'}",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string. otherwise, parse args.",
        default=False,
    )

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:
        # arg_str = "--max-K 3"
        arg_str = "--max-K 3"
        arg_str += ' --u0-type box'
        # arg_str += ' --use-mmr-optimal'
        # arg_str += " --use-mmu-heuristic"
        arg_str += " --use-mmr-heuristic"
        # arg_str += ' --time-limit 20'
        # arg_str += " --evaluate-mmu-objective"
        arg_str += " --evaluate-mmr-objective"
        # arg_str += " --problem-seed 101"
        arg_str += " --problem-seed 0"
        arg_str += " --opt-K 0"
        arg_str += " -p 10 5 1.0"
        # arg_str += " --gamma 0.0"
        # arg_str += " --num-items 10"
        arg_str += ' --num-random-samples 5'
        # arg_str += " --num-features 10"
        # arg_str += " --output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        arg_str += " --output-dir DEBUG_folder"
        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    main()
