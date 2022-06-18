# compare runtime and objval of the direct MIP approaches with the heuristic. apply a fixed 3hr time limit for each

import argparse
import time
from collections import namedtuple

import numpy as np
from gurobipy import *

from preference_classes import generate_items
from scenario_decomposition import (
    solve_scenario_decomposition,
    evaluate_query_list_decomp,
)
from static_heuristics import solve_warm_start_decomp_heuristic
from utils import generate_filepath, get_logger

# TODO: the objective normalization here is not correct. see static_experiment.py for the correct normalization

def experiment(args):
    # some fixed parameters
    valid_responses = [1, -1]
    item_sphere_size = 10.0
    query_seed = 0

    # 40 items, 10 features, 0 gamma:
    num_items_list = [20]
    num_features = 10
    gamma = 0.0

    k_list = [2, 4, 6, 8, 10]

    # generate an output file
    output_file = generate_filepath(args.output_dir, "timing_experiment", "csv")
    log_file = generate_filepath(args.output_dir, "timing_experiment_LOGS", "txt")
    logger = get_logger(logfile=log_file)

    logger.info("generating output file: {}".format(output_file))
    logger.info("generating log file: {}".format(log_file))

    col_list = [
        "problem_seed",
        "num_features",
        "num_items",
        "item_sphere_size",
        "method",
        "K",
        "mmu_objval",
        "mmu_normalized_objval",
        "query_list",
        "solve_time",
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
            result.problem_seed,
            result.num_features,
            result.num_items,
            result.item_sphere_size,
            result.method,
            result.K,
            result.mmu_objval,
            result.mmu_normalized_objval,
            result.query_list,
            result.solve_time,
        )
        return result_str

    for num_items in num_items_list:

        logger.info("num_items = {}. creating items".format(num_items))
        items = generate_items(
            num_features,
            num_items,
            item_sphere_size=item_sphere_size,
            seed=args.problem_seed,
        )

        # get the max 1-norm of all items
        max_norm = max([np.sum(np.abs(i.features)) for i in items])

        # for k in k_list:
            # # ------------------ MILP: full mip without symmetry breaking ------------------
            # method_name = 'MILP'
            # logger.info('running method ({}), (k={})'.format(method_name, k))
            # t0 = time.time()
            # queries, objval, _, _ = static_mip_optimal(items, k, valid_responses,
            #                                            cut_1=False,
            #                                            cut_2=False,
            #                                            problem_type='maximin',
            #                                            time_lim=args.time_limit,
            #                                            raise_gurobi_time_limit=False
            #                                            )
            # solve_time = time.time() - t0
            #
            # result = Result(
            #     problem_seed=args.problem_seed,
            #     num_features=num_features,
            #     num_items=num_items,
            #     item_sphere_size=item_sphere_size,
            #     method=method_name,
            #     K=k,
            #     mmu_objval=objval,
            #     mmu_normalized_objval=((objval + max_norm) / (2 * max_norm)),
            #     query_list=[q.to_tuple() for q in queries],
            #     solve_time=solve_time,
            # )
            #
            # logger.info('writing ({}) results to file (k={})'.format(method_name, k))
            # with open(output_file, 'a') as f:
            #     f.write(result_to_str(result))
            #
            # # ------------------ MILP + cuts: full mip with symmetry breaking ------------------
            # method_name = 'MILP_cuts'
            # logger.info('running method ({}), (k={})'.format(method_name, k))
            # t0 = time.time()
            # queries, objval, _, _ = static_mip_optimal(items, k, valid_responses,
            #                                            cut_1=True,
            #                                            cut_2=True,
            #                                            problem_type='maximin',
            #                                            time_lim=args.time_limit,
            #                                            raise_gurobi_time_limit=False
            #                                            )
            # solve_time = time.time() - t0
            #
            # result = Result(
            #     problem_seed=args.problem_seed,
            #     num_features=num_features,
            #     num_items=num_items,
            #     item_sphere_size=item_sphere_size,
            #     method=method_name,
            #     K=k,
            #     mmu_objval=objval,
            #     mmu_normalized_objval=((objval + max_norm) / (2 * max_norm)),
            #     query_list=[q.to_tuple() for q in queries],
            #     solve_time=solve_time,
            # )
            #
            # logger.info('writing ({}) results to file (k={})'.format(method_name, k))
            # with open(output_file, 'a') as f:
            #     f.write(result_to_str(result))

            # # ------------------ CCG: scenario decomp without symmetry breaking cuts ------------------
            # method_name = "CCG"
            # logger.info("running method ({}), (k={})".format(method_name, k))
            # rs = np.random.RandomState(args.problem_seed)
            # t0 = time.time()
            # queries, objval, status = solve_scenario_decomposition(
            #     items,
            #     k,
            #     rs,
            #     valid_responses,
            #     args.u0_type,
            #     cut_1=False,
            #     cut_2=False,
            #     max_iter=1000000,
            #     problem_type="maximin",
            #     time_limit=args.time_limit,
            # )
            # assert status != "no_incumbent"
            # solve_time = time.time() - t0
            #
            # result = Result(
            #     problem_seed=args.problem_seed,
            #     num_features=num_features,
            #     num_items=num_items,
            #     item_sphere_size=item_sphere_size,
            #     method=method_name,
            #     K=k,
            #     mmu_objval=objval,
            #     mmu_normalized_objval=((objval + max_norm) / (2 * max_norm)),
            #     query_list=[q.to_tuple() for q in queries],
            #     solve_time=solve_time,
            # )
            #
            # logger.info("writing ({}) results to file (k={})".format(method_name, k))
            # with open(output_file, "a") as f:
            #     f.write(result_to_str(result))

            # # ------------------ CCG_cuts: scenario decomp with symmetry breaking cuts ------------------
            # method_name = 'CCG_cuts'
            # logger.info('running method ({}), (k={})'.format(method_name, k))
            # rs = np.random.RandomState(args.problem_seed)
            # t0 = time.time()
            # queries, objval, _, _ = solve_scenario_decomposition(items, k, rs, valid_responses,
            #                                                      cut_1=True,
            #                                                      cut_2=True,
            #                                                      max_iter=1000000,
            #                                                      problem_type='maximin',
            #                                                      time_limit=args.time_limit,
            #                                                      raise_gurobi_time_limit=False,
            #                                                      )
            # solve_time = time.time() - t0
            #
            # result = Result(
            #     problem_seed=args.problem_seed,
            #     num_features=num_features,
            #     num_items=num_items,
            #     item_sphere_size=item_sphere_size,
            #     method=method_name,
            #     K=k,
            #     mmu_objval=objval,
            #     mmu_normalized_objval=((objval + max_norm) / (2 * max_norm)),
            #     query_list=[q.to_tuple() for q in queries],
            #     solve_time=solve_time,
            # )
            #
            # logger.info('writing ({}) results to file (k={})'.format(method_name, k))
            # with open(output_file, 'a') as f:
            #     f.write(result_to_str(result))

        # ------------------ heuristic: warm start + decomp heuristic approach ------------------
        run_heuristic = True
        if run_heuristic:
            method_name = "heuristic"
            logger.info("running method {})".format(method_name))
            (
                queries,
                objval_list,
                _,
                incremental_times,
            ) = solve_warm_start_decomp_heuristic(
                items,
                max(k_list),
                valid_responses,
                args.u0_type,
                problem_type="maximin",
                time_lim=args.time_limit,
                return_incremental_times=True,
            )

            # get the max k that finished...
            max_k = len(queries)

            k_list_updated = [k for k in k_list if k <= max_k]

            if max_k not in k_list_updated:
                k_list_updated.append(max_k)

            # gather heuristic results
            heuristic_results = []
            for k in k_list_updated:
                query_subset = queries[:k]

                objval = evaluate_query_list_decomp(
                    query_subset,
                    items,
                    valid_responses,
                    args.u0_type,
                    gamma_inconsistencies=gamma,
                    problem_type="maximin",
                )

                heuristic_results.append(
                    Result(
                        problem_seed=args.problem_seed,
                        num_features=num_features,
                        num_items=num_items,
                        item_sphere_size=item_sphere_size,
                        method=method_name,
                        K=k,
                        mmu_objval=objval,
                        mmu_normalized_objval=((objval + max_norm) / (2 * max_norm)),
                        query_list=[q.to_tuple() for q in query_subset],
                        solve_time=incremental_times[k - 1],
                    )
                )

            logger.info("writing ({}) results to file".format(method_name, k))
            with open(output_file, "a") as f:
                for result in heuristic_results:
                    f.write(result_to_str(result))

    logger.info("done.")


def main():
    parser = argparse.ArgumentParser(
        description="timing experiments comparing optimal approaches to the heuristic"
    )

    parser.add_argument(
        "--problem-seed",
        type=int,
        help="random seed for generating the problem instances",
        default=0,
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument(
        "--time-limit",
        type=int,
        help="overall time limit for each approach",
        default=(60 * 60 * 3),
    )
    parser.add_argument(
        "--u0-type",
        type=str,
        help="type of u0",
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
        arg_str = "--output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        arg_str += " --time-limit 10000000"
        arg_str += " --u0-type box"
        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    main()
