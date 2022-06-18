import argparse
import time
from collections import namedtuple

import numpy as np
from gurobipy import *
from scipy.special import erfinv
from tqdm import tqdm

from preference_classes import Query, Agent
from scenario_decomposition import evaluate_query_list_decomp
from static_heuristics import solve_warm_start_decomp_heuristic
from recommendation import solve_recommendation_problem
from read_csv_to_items import get_data_items
from utils import generate_filepath, get_logger

# TODO: the objective normalization here is not correct. see static_experiment.py for the correct normalization


def experiment(args):
    """
    similar to static_experiment, but use real data
    """

    # some fixed parameters
    valid_responses = [1, -1]
    query_seed = 0
    p_confidence = 0.9

    rs = np.random.RandomState(0)

    # generate an output file
    output_file = generate_filepath(args.output_dir, f"static_experiment_{args.job_index}", "csv")
    log_file = generate_filepath(args.output_dir, f"static_experiment_LOGS_{args.job_index}", "txt")
    logger = get_logger(logfile=log_file)

    logger.info("generating output file: {}".format(output_file))
    logger.info("generating log file: {}".format(log_file))

    col_list = [
        "num_features",
        "num_items",
        "method",
        "query_seed",
        "K",
        "gamma",
        "gamma_normalized",
        "true_gamma",
        "method_objval",
        "method_objval_normalized",
        "mmu_objval",
        "mmu_normalized_objval",
        "mmr_objval",
        "mmr_normalized_objval",
        "agent_number",
        "rec_item_index",
        "true_u",
        "true_u_normalized",
        "true_regret",
        "true_regret_normalized",
        "true_rank",
        "query_list",
        "solve_time",
        "mmu_objective_eval_time",
        "mmr_objective_eval_time",
    ]

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
            result.num_features,
            result.num_items,
            result.method,
            result.query_seed,
            result.K,
            result.gamma,
            result.gamma_normalized,
            result.true_gamma,
            result.method_objval,
            result.method_objval_normalized,
            result.mmu_objval,
            result.mmu_normalized_objval,
            result.mmr_objval,
            result.mmr_normalized_objval,
            result.agent_number,
            result.rec_item_index,
            result.true_u,
            result.true_u_normalized,
            result.true_regret,
            result.true_regret_normalized,
            result.true_rank,
            result.query_list,
            result.solve_time,
            result.mmu_objective_eval_time,
            result.mmr_objective_eval_time,
        )
        return result_str

    def get_write_results(
        queries,
        name,
        items,
        num_features,
        objval_list,
        mmu_heuristic_objval_dict,
        mmr_heuristic_objval_dict,
        solve_times,
        gamma,
        gamma_normalized,
        query_seed=None,
    ):
        assert len(queries) == args.max_K

        for k in range(args.max_K):
            # find the actual obj. val of the first k queries
            query_subset = queries[: (k + 1)]

            method_objval_normalized = 0
            if name == "mmu_heuristic":
                if args.u0_type == "box":
                    method_objval_normalized = None
                if args.u0_type == "positive_normed":
                    method_objval_normalized = None
            if name == "mmr_heuristic":
                if args.u0_type == "box":
                    method_objval_normalized = None
                if args.u0_type == "positive_normed":
                    method_objval_normalized = None

            if args.evaluate_mmu_objective:
                t0 = time.time()
                logger.debug("evaluating objective list for MMU")
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
                if args.u0_type == "box":
                    mmu_normalized_objval = None
                if args.u0_type == "positive_normed":
                    mmu_normalized_objval = None

            else:
                mmu_objval = None
                mmu_normalized_objval = None
                mmu_objective_eval_time = None

            if args.evaluate_mmr_objective:
                logger.debug("evaluating objective for MMR")
                t0 = time.time()
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
                mmr_normalized_objval = None
                if args.u0_type == "box":
                    mmr_normalized_objval = None
                if args.u0_type == "positive_normed":
                    mmr_normalized_objval = None

            else:
                mmr_objval = None
                mmr_normalized_objval = None
                mmr_objective_eval_time = None

            result = Result(
                num_features=num_features,
                num_items=len(items),
                method=name,
                query_seed=query_seed,
                K=k + 1,
                gamma=gamma,
                gamma_normalized=gamma_normalized,
                true_gamma=args.true_gamma,
                method_objval=objval_list[k],
                method_objval_normalized=method_objval_normalized,
                mmu_objval=mmu_objval,
                mmu_normalized_objval=mmu_normalized_objval,
                mmr_objval=mmr_objval,
                mmr_normalized_objval=mmr_normalized_objval,
                agent_number=None,
                rec_item_index=None,
                true_u=None,
                true_u_normalized=None,
                true_regret=None,
                true_regret_normalized=None,
                true_rank=None,
                query_list=[q.to_tuple() for q in query_subset],
                solve_time=solve_times[k],
                mmu_objective_eval_time=mmu_objective_eval_time,
                mmr_objective_eval_time=mmr_objective_eval_time,
            )
            logger.info(f"writing MMR heuristic result to file for k={k}")
            with open(output_file, "a") as f:
                f.write(result_to_str(result))

    def get_single_result(
        queries,
        name,
        items,
        num_features,
        objval,
        solve_time,
        gamma,
        gamma_normalized,
        k,
        query_seed=None,
    ):
        assert len(queries) == k

        # find the actual obj. val of the first k queries
        query_subset = queries[: (k + 1)]

        method_objval_normalized = 0
        if name == "mmu_heuristic":
            method_objval_normalized = None
        if name == "mmr_heuristic":
            method_objval_normalized = None

        if name == "random":
            # evaluating MMU objective
            logger.info(f"evaluating MMU objval for k={k}")
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
            mmu_normalized_objval = None

            logger.info(f"evaluating MMR objval for k={k}")
            query_subset = queries[: (k + 1)]
            t0 = time.time()
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
            mmr_normalized_objval = None
        else:
            mmu_objval = None
            mmu_normalized_objval = None
            mmu_objective_eval_time = None
            mmr_objval = None
            mmr_normalized_objval = None
            mmr_objective_eval_time = None

        result = Result(
            num_features=num_features,
            num_items=len(items),
            method=name,
            query_seed=query_seed,
            K=k,
            gamma=gamma,
            gamma_normalized=gamma_normalized,
            true_gamma=args.true_gamma,
            method_objval=objval,
            method_objval_normalized=method_objval_normalized,
            mmu_objval=mmu_objval,
            mmr_objval=mmr_objval,
            mmu_normalized_objval=mmu_normalized_objval,
            mmr_normalized_objval=mmr_normalized_objval,
            agent_number=None,
            rec_item_index=None,
            true_u=None,
            true_u_normalized=None,
            true_regret=None,
            true_regret_normalized=None,
            true_rank=None,
            query_list=[q.to_tuple() for q in query_subset],
            solve_time=solve_time,
            mmu_objective_eval_time=mmu_objective_eval_time,
            mmr_objective_eval_time=mmr_objective_eval_time,
        )
        return result

    # read items
    if args.normalize:
        items = get_data_items(
            args.input_csv, max_items=args.max_data_items, standardize_features=False, normalize_features=True, drop_cols=["IsInterpretable_int"]
        )
    else:
        items = get_data_items(
            args.input_csv, max_items=args.max_data_items, standardize_features=False, normalize_features=False, drop_cols=["IsInterpretable_int"]
        )
    num_features = len(items[0].features)

    gamma_unnormalized = args.gamma
    true_gamma_unnormalized = args.true_gamma

    logger.info(f"starting experiment on data from CSV {args.input_csv}")
    # keep track of all Results from this parameter set

    logger.info(f"num items = {len(items)}, num_features = {num_features}")

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

    # if u0-type is normed, calcualte a different normalization constant
    # get the max feature over all items
    max_feat = max([np.max(i.features) for i in items])
    min_feat = min([np.min(i.features) for i in items])
    logger.info(f"max (min) item feature = {max_feat} ({min_feat})")

    # get the max 1-norm *difference* between items (for mmr)
    max_feat_diff = max(
        [
            np.max(np.abs(i.features - j.features))
            for i, j in itertools.combinations(items, 2)
        ]
    )
    min_feat_diff = -max_feat_diff

    logger.info(f"max (min) item feature-diff = {max_feat_diff} ({min_feat_diff})")

    # alpha for normalizing gamma
    inv_alpha = 0.5

    logger.info("creating agents")
    agent_list = []
    for i in range(args.num_agents):
        seed = args.agent_seed + i
        agent_list.append(
            # Agent.random(num_features, id=i, sphere_size=agent_sphere_size, seed=seed,)
            Agent.random_fixed_sum(num_features, id=i, seed=seed)
        )

    # ------------------ maximin utility: warm start heuristic with scenario decomposition ------------------
    mmu_heuristic_objval_dict = {}
    if args.use_mmu_heuristic:
        if gamma_unnormalized == 0:
            gamma = 0

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
                gamma_inconsistencies=gamma,
            )
            # heuristic_queries, objval_list, _, incremental_times = solve_warm_start_heuristic(items, args.max_K,
            #                                                                                valid_responses,
            #                                                                                print_logs=PRINT_LOGS,
            #                                                                                time_lim=args.time_limit,
            #                                                                                time_lim_overall=False,
            #                                                                                problem_type='maximin',
            #                                                                                return_incremental_times=True,
            #                                                                               gamma_inconsistencies=0,
            #                                                                                   )
            logger.info("finished MMU heuristic")
            logger.info("collecting results for MMU heuristic")
            get_write_results(
                heuristic_queries,
                "mmu_heuristic",
                items,
                num_features,
                objval_list,
                {},
                {},
                incremental_times,
                gamma_unnormalized,
                gamma,
            )

        else:
            # raise NotImplemented
            for k in range(1, args.max_K + 1):
                sigma_hat = np.sqrt(2.0 * k * (gamma_unnormalized ** 2))
                gamma = (
                    sigma_hat * np.sqrt(2) * erfinv(2.0 * p_confidence - 1.0)
                )

                logger.info("running MMU heuristic with k={} (gamma>0)".format(k))
                t0 = time.time()
                (
                    heuristic_queries,
                    objval,
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
                    problem_type="maximin",
                )
                logger.info(
                    "finished MMU heuristic with k={}. collecting results...".format(
                        k
                    )
                )
                heuristic_time = time.time() - t0
                logger.info("collecting results for MMU heuristic")
                result = get_single_result(
                    heuristic_queries,
                    "mmu_heuristic",
                    items,
                    num_features,
                    objval[k - 1],
                    heuristic_time,
                    gamma_unnormalized,
                    gamma,
                    k,
                )
                with open(output_file, "a") as f:
                    f.write(result_to_str(result))

    # ------------------ minimax regret: warm start heuristic with scenario decomposition ------------------
    mmr_heuristic_objval_dict = {}
    if args.use_mmr_heuristic:
        if gamma_unnormalized == 0:
            gamma = 0

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
                gamma_inconsistencies=gamma,
            )

            logger.info("collecting results for MMR heuristic")
            get_write_results(
                heuristic_queries,
                "mmr_heuristic",
                items,
                num_features,
                objval_list,
                {},
                {},
                incremental_times,
                gamma_unnormalized,
                gamma,
            )

            # (
            #     heuristic_queries,
            #     objval_list,
            #     _,
            #     incremental_times,
            # ) = solve_warm_start_heuristic(
            #     items,
            #     args.max_K,
            #     valid_responses,
            #     print_logs=PRINT_LOGS,
            #     time_lim=args.time_limit,
            #     time_lim_overall=False,
            #     return_incremental_times=True,
            #     problem_type="mmr",
            # )

            # # run the heuristic manually
            # opt_queries_full = []
            # for k in range(args.max_K):
            #     k_solve = k + 1
            #     logger.info(f"running MMR heuristic, k={k_solve}")
            #     t0 = time.time()
            #     queries_opt, objval, _, _ = static_mip_optimal(
            #         items,
            #         k_solve,
            #         valid_responses,
            #         cut_1=True,
            #         cut_2=False,
            #         fixed_queries=opt_queries_full,
            #         time_lim=args.time_limit,
            #         problem_type="mmr",
            #         gamma_inconsistencies=0,
            #         raise_gurobi_time_limit=False,
            #     )
            #     solve_time = time.time() - t0
            #     opt_queries_full = queries_opt
            #
            #     get_write_single_result(
            #         opt_queries_full,
            #         "mmr_heuristic",
            #         items,
            #         num_features,
            #         objval,
            #         solve_time,
            #         0.0,
            #         0.0,
            #         query_seed=None,
            #     )


        else:
            # raise NotImplemented
            for k in range(1, args.max_K + 1):
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
                logger.info("collecting results for MMR heuristic")
                result = get_single_result(
                    heuristic_queries,
                    "mmr_heuristic",
                    items,
                    num_features,
                    objval_list[k - 1],
                    heuristic_time,
                    gamma_unnormalized,
                    gamma,
                    k,
                )
                with open(output_file, "a") as f:
                    f.write(result_to_str(result))

    # Simulate the agent answers
    if args.num_agents > 0:
        assert len(heuristic_queries) == args.max_K
        for agent in tqdm(agent_list):
            xi = np.random.normal(0.0, true_gamma_unnormalized, args.max_K)
            for k in range(args.max_K):
                agent.answered_queries = []
                query_subset = heuristic_queries[: (k + 1)]
                for q in query_subset:
                    if true_gamma_unnormalized == 0:
                        agent.answer_query(q)
                    else:
                        agent.answer_query(q, error=xi[k])

                if args.use_mmu_heuristic:
                    mmu_objval, rec_item = solve_recommendation_problem(
                        agent.answered_queries,
                        items,
                        "maximin",
                        gamma=gamma,
                        u0_type=args.u0_type,
                        logger=logger,
                    )
                    mmu_objval_normalized = (mmu_objval + max_norm) / (
                        2 * max_norm
                    )
                else:
                    mmu_objval = None
                    mmu_objval_normalized = None

                if args.use_mmr_heuristic:
                    mmr_objval, rec_item = solve_recommendation_problem(
                        agent.answered_queries,
                        items,
                        "mmr",
                        gamma=gamma,
                        u0_type=args.u0_type,
                        logger=logger,
                    )
                    mmr_objval_normalized = (mmr_objval + max_diff_norm) / (
                        2 * max_diff_norm
                    )
                else:
                    mmr_objval = None
                    mmr_objval_normalized = None

                # evaluate true utility, true rank, true regret
                true_rank = agent.true_item_rank(rec_item, items)
                true_u = agent.true_utility(rec_item)
                true_u_normalized = (true_u + max_norm) / (2 * max_norm)
                true_regret = agent.true_item_max_regret(rec_item, items)
                true_regret_normalized = (true_regret + max_diff_norm) / (
                    2 * max_diff_norm
                )

                result = Result(
                    num_features=num_features,
                    num_items=len(items),
                    method="mmr_heuristic",
                    query_seed=query_seed,
                    K=k + 1,
                    gamma=gamma_unnormalized,
                    gamma_normalized=gamma,
                    true_gamma=args.true_gamma,
                    method_objval=None,
                    method_objval_normalized=None,
                    mmu_objval=mmu_objval,
                    mmu_normalized_objval=mmu_objval_normalized,
                    mmr_objval=mmr_objval,
                    mmr_normalized_objval=mmr_objval_normalized,
                    agent_number=agent.id,
                    rec_item_index=rec_item.id,
                    true_u=true_u,
                    true_u_normalized=true_u_normalized,
                    true_regret=true_regret,
                    true_regret_normalized=true_regret_normalized,
                    true_rank=true_rank,
                    query_list=[q.to_tuple() for q in query_subset],
                    solve_time=None,
                    mmu_objective_eval_time=None,
                    mmr_objective_eval_time=None,
                )
                logger.info(f"writing MMR heuristic result to file for k={k}")
                with open(output_file, "a") as f:
                    f.write(result_to_str(result))

    # ------------------ random queries ------------------
    if args.num_random_samples > 0:
        logger.info("running random queries")
        for i in range(args.num_random_samples):
            random_queries, _ = generate_random_queries(items, args.max_K, query_seed)

            for k in range(1, args.max_K + 1):
                sigma_hat = np.sqrt(2.0 * k * (gamma_unnormalized ** 2))
                gamma = (
                    sigma_hat * np.sqrt(2) * erfinv(2.0 * p_confidence - 1.0)
                )
                query_list = random_queries[:k]
                logger.info(
                    "evaluating random queries sample num={}, k={}...".format(i + 1, k)
                )
                result = get_single_result(
                    query_list,
                    "random",
                    items,
                    num_features,
                    0,
                    0,
                    gamma_unnormalized,
                    gamma,
                    k,
                )

                with open(output_file, "a") as f:
                    f.write(result_to_str(result))

            query_seed += 1

        logger.info("finished generating random queries")

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
        "--max-K", type=int, help="total number of queries to ask", default=2
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
        "--gamma",
        type=float,
        # nargs='+',
        default=0.0,
        help="level of agent inconsistencies. if a list is provided, use all of them, while fixing other "
        "experiment parameters",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1e10,
        help="time limit (in seconds) allowed for each stage (k) of the warm-start heuristics, and for "
        "the entire optimal method run",
    )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=1,
        help="number of sets of random queries to compare to",
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--input-csv", type=str, help="csv of item data rto read")
    parser.add_argument(
        "--max-data-items", type=int, help="max number of items to read"
    )
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
        "--standardize-features",
        action="store_true",
        help="if set, standardize the features of the items",
    )
    parser.add_argument(
        "--u0-type",
        type=str,
        default="positive_normed",
        help="type of initial uncertainty set to use {'box' | 'positive_normed'}",
    )
    parser.add_argument(
        "--evaluate-mmu-objective",
        action="store_true",
        help="if set, evaluate the maximin utility objective",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="if set, use normalization",
        default=False,
    )

    parser.add_argument(
        "--agent-seed",
        type=int,
        help="random seed for generating agents instances",
        default=0,
    )
    parser.add_argument(
        "--job-index",
        type=int,
        help="the index of the job to make the filename different",
        default=0,
    )
    parser.add_argument(
        "--true-gamma", type=float, default=0.0, help="level of true agent inconsistencies"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=0,
        help="number of random agents to test elicitation on",
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
        arg_str = "--max-K 2"
        arg_str += " --use-mmr-heuristic"
        arg_str += " --u0-type positive_normed"
        arg_str += " --time-limit 10800"
        # arg_str += " --num-random-samples 2"
        arg_str += " --gamma 0.0"
        arg_str += " --true-gamma 1.0"
        arg_str += " --num-agents 3"
        # arg_str += " --normalize"
        arg_str += " --num-random-samples 0"
        # arg_str += " --output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        arg_str += " --output-dir DEBUG_folder"
        # arg_str += ' --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS_new.csv'
        # arg_str += " --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
        arg_str += " --input-csv test_results/AdultHMIS_20210906_preprocessed_final_Robust_25.csv"
        arg_str += " --max-data-items 50"
        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    main()
