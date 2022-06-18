# compare the adaptive elicitation method to both random and AC on a single problem, on many simulated agents.

import argparse
import time
from collections import namedtuple

import numpy as np
from gurobipy import *

from adaptive_elicitation import (
    next_optimal_query_mip,
    get_next_query_ac,
    get_random_query,
)
from preference_classes import Query, Agent
from recommendation import solve_recommendation_problem, recommend_item_ac
from read_csv_to_items import get_data_items
from utils import generate_filepath, get_logger

# TODO: the objective normalization here is not correct. see static_experiment.py for the correct normalization

def experiment(args):
    """
    same as adaptive_experiment_OLD.py but for items read by data
    """

    # some fixed parameters
    item_sphere_size = 10.0
    agent_sphere_size = 1.0
    query_seed = 0

    assert args.obj_type in ["maximin", "mmr"]

    # alpha for normalizing gamma
    inv_alpha = 0.5

    # generate an output file
    output_file = generate_filepath(args.output_dir, "adaptive_experiment", "csv")
    log_file = generate_filepath(args.output_dir, "adaptive_experiment_LOGS", "txt")
    logger = get_logger(logfile=log_file)

    logger.info("generating output file: {}".format(output_file))
    logger.info("generating log file: {}".format(log_file))

    col_list = [
        "num_features",
        "num_items",
        "elicitation_method",
        "recommendation_method",
        "query_seed",
        "K",
        "max_K",
        "gamma",
        "gamma_normalized",
        "agent_number",
        "rec_item_index",
        "mmu_objval",
        "mmu_objval_normalized",
        "mmr_objval",
        "mmr_objval_normalized",
        "true_u",
        "true_u_normalized",
        "true_regret",
        "true_regret_normalized",
        "true_rank",
        "answered_queries",
        "elicitation_time",
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
            result.elicitation_method,
            result.recommendation_method,
            result.query_seed,
            result.K,
            result.max_K,
            result.gamma,
            result.gamma_normalized,
            result.agent_number,
            result.rec_item_index,
            result.mmu_objval,
            result.mmu_objval_normalized,
            result.mmr_objval,
            result.mmr_objval_normalized,
            result.true_u,
            result.true_u_normalized,
            result.true_regret,
            result.true_regret_normalized,
            result.true_rank,
            result.answered_queries,
            result.elicitation_time,
        )
        return result_str

    rec_methods_dict = {
        "maximin": lambda agent, gamma: solve_recommendation_problem(
            agent.answered_queries,
            items,
            "maximin",
            gamma=gamma,
            u0_type=args.u0_type,
            logger=logger,
        )[1],
        "mmr": lambda agent, gamma: solve_recommendation_problem(
            agent.answered_queries,
            items,
            "mmr",
            gamma=gamma,
            u0_type=args.u0_type,
            logger=logger,
        )[1],
        "AC": lambda agent, gamma: recommend_item_ac(
            agent.answered_queries, items, gamma, args.u0_type
        ),
    }

    def get_write_agent_results(
        agent,
        items,
        elicitation_method,
        recommendation_methods,
        gamma_unnormalized,
        rs,
        output_file,
    ):
        """
        simulate elicitation with this agent.
        return a list of results, one for each k = 1, ..., K
        """

        assert elicitation_method in ["maximin", "mmr", "random", "AC"]
        assert set(recommendation_methods).issubset(set(["maximin", "mmr", "AC"]))

        if elicitation_method in ["maximin", "mmr"]:
            next_query = lambda agent, gamma: next_optimal_query_mip(
                agent.answered_queries,
                items,
                elicitation_method,
                gamma,
                time_limit=args.time_limit,
                u0_type=args.u0_type,
                logger=logger,
            )[1]
        if elicitation_method == "AC":
            next_query = lambda agent, gamma: get_next_query_ac(
                agent.answered_queries, items, gamma, args.u0_type
            )
        if elicitation_method == "random":
            next_query = lambda agent, gamma: get_random_query(items, rs)

        # reset the agent's answered queries
        agent.answered_queries = []

        if gamma_unnormalized == 0:
            gamma = 0
            for k in range(args.max_K):
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : solving elicitation step {(k + 1)} of {args.max_K}"
                )
                # answer the next query
                t0 = time.time()
                q = next_query(agent, gamma)
                elicitation_time = time.time() - t0
                assert isinstance(q, Query)
                agent.answer_query(q)

                # make recommendation(s)
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : solving rec."
                )
                rec_items = {
                    rec_method: rec_methods_dict[rec_method](agent, gamma)
                    for rec_method in recommendation_methods
                }

                # evaluate recommendation(s)
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : evaluating rec."
                )
                for rec_method, rec_item in rec_items.items():

                    # evaluate MMU, MMR objval by solving the recommendation problem for this item only
                    if args.obj_type == "maximin":
                        mmu_objval, _ = solve_recommendation_problem(
                            agent.answered_queries,
                            items,
                            "maximin",
                            gamma=gamma,
                            fixed_rec_item=rec_item,
                            u0_type=args.u0_type,
                            logger=logger,
                        )
                        if args.u0_type == "box":
                            mmu_objval_normalized = (mmu_objval + max_norm) / (2 * max_norm)
                        if args.u0_type == "positive_normed":
                            mmu_objval_normalized = (mmu_objval - min_feat) / float(max_feat - min_feat)

                    else:
                        mmu_objval = None
                        mmu_objval_normalized = None

                    if args.obj_type == "mmr":
                        mmr_objval, _ = solve_recommendation_problem(
                            agent.answered_queries,
                            items,
                            "mmr",
                            gamma=gamma,
                            fixed_rec_item=rec_item,
                            u0_type=args.u0_type,
                            logger=logger,
                        )
                        if args.u0_type == "box":
                            mmr_objval_normalized = (mmr_objval + max_diff_norm) / (2 * max_diff_norm)
                        if args.u0_type == "positive_normed":
                            mmr_objval_normalized = (mmr_objval - min_feat_diff) /float(max_feat_diff - min_feat_diff)

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
                        query_seed=query_seed,
                        elicitation_method=elicitation_method,
                        recommendation_method=rec_method,
                        K=len(agent.answered_queries),
                        max_K=args.max_K,
                        gamma=gamma_unnormalized,
                        gamma_normalized=gamma,
                        agent_number=agent.id,
                        rec_item_index=rec_item.id,
                        mmu_objval=mmu_objval,
                        mmu_objval_normalized=mmu_objval_normalized,
                        mmr_objval=mmr_objval,
                        mmr_objval_normalized=mmr_objval_normalized,
                        true_u=true_u,
                        true_u_normalized=true_u_normalized,
                        true_regret=true_regret,
                        true_regret_normalized=true_regret_normalized,
                        true_rank=true_rank,
                        answered_queries=str(
                            [q.to_tuple() for q in agent.answered_queries]
                        ),
                        elicitation_time=elicitation_time,
                    )
                    with open(output_file, "a") as f:
                        f.write(result_to_str(result))
        else:
            raise NotImplemented
            # # gamma > 0: re-run elicitation for each k = 1, ..., K
            # for k_total in range(args.max_K):
            #
            #     num_queries = k_total + 1
            #     gamma_normalized = (
            #         2.0
            #         * gamma_unnormalized
            #         * np.power(num_queries, inv_alpha)
            #         * max_norm
            #     )
            #
            #     # calculate agent error before elicitation
            #     if gamma_normalized > 0:
            #         xi = random_vector_bounded_sum_rejection(args.max_K, gamma_normalized, rs)
            #     else:
            #         xi = np.zeros(args.max_K)
            #
            #     # reset elicitation
            #     agent.answered_queries = []
            #
            #     for k in range(num_queries):
            #
            #         # answer the next query
            #         t0 = time.time()
            #         q = next_query(agent, gamma_normalized)
            #         elicitation_time = time.time() - t0
            #         agent.answer_query(q, error=xi[k])
            #
            #         # make recommendation(s)
            #         rec_items = {
            #             rec_method: rec_methods_dict[rec_method](
            #                 agent, gamma_normalized
            #             )
            #             for rec_method in recommendation_methods
            #         }
            #
            #         # evaluate recommendation(s)
            #         for rec_method, rec_item in rec_items.items():
            #
            #             # evaluate MMU, MMR objval by solving the recommendation problem for this item only
            #             if args.obj_type == "maximin":
            #                 mmu_objval, _ = solve_recommendation_problem(
            #                     agent.answered_queries,
            #                     items,
            #                     "maximin",
            #                     gamma=gamma_normalized,
            #                     fixed_rec_item=rec_item,
            #                 )
            #                 mmu_objval_normalized = (mmu_objval + max_norm) / (
            #                     2 * max_norm
            #                 )
            #             else:
            #                 mmu_objval = None
            #                 mmu_objval_normalized = None
            #
            #             if args.obj_type == "mmr":
            #                 mmr_objval, _ = solve_recommendation_problem(
            #                     agent.answered_queries,
            #                     items,
            #                     "mmr",
            #                     gamma=gamma_normalized,
            #                     fixed_rec_item=rec_item,
            #                 )
            #                 mmr_objval_normalized = (mmr_objval + max_diff_norm) / (
            #                     2 * max_diff_norm
            #                 )
            #             else:
            #                 mmr_objval = None
            #                 mmr_objval_normalized = None
            #
            #             # evaluate true utility, true rank, true regret
            #             true_rank = agent.true_item_rank(rec_item, items)
            #             true_u = agent.true_utility(rec_item)
            #             true_u_normalized = (true_u + max_norm) / (2 * max_norm)
            #             true_regret = agent.true_item_max_regret(rec_item, items)
            #             true_regret_normalized = (true_regret + max_diff_norm) / (
            #                 2 * max_diff_norm
            #             )
            #
            #             agent_results.append(
            #                 Result(
            #                     num_features=num_features,
            #                     num_items=len(items),
            #                     query_seed=query_seed,
            #                     elicitation_method=elicitation_method,
            #                     recommendation_method=rec_method,
            #                     K=len(agent.answered_queries),
            #                     max_K=num_queries,
            #                     gamma=gamma_unnormalized,
            #                     gamma_normalized=gamma_normalized,
            #                     agent_number=agent.id,
            #                     rec_item_index=rec_item.id,
            #                     mmu_objval=mmu_objval,
            #                     mmu_objval_normalized=mmu_objval_normalized,
            #                     mmr_objval=mmr_objval,
            #                     mmr_objval_normalized=mmr_objval_normalized,
            #                     true_u=true_u,
            #                     true_u_normalized=true_u_normalized,
            #                     true_regret=true_regret,
            #                     true_regret_normalized=true_regret_normalized,
            #                     true_rank=true_rank,
            #                     answered_queries=str(
            #                         [q.to_tuple() for q in agent.answered_queries]
            #                     ),
            #                     elicitation_time=elicitation_time,
            #                 )
            #             )

        # return agent_results

    # read items
    items = get_data_items(
        args.input_csv, max_items=args.max_data_items, standardize_features=True, drop_cols=["IsInterpretable_int"]
    )
    num_features = len(items[0].features)

    gamma_unnormalized = args.gamma

    rs = np.random.RandomState()

    logger.info(f"starting experiment on data from CSV {args.input_csv}")

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
    min_feat_diff = - max_feat_diff

    logger.info(f"max (min) item feature-diff = {max_feat_diff} ({min_feat_diff})")

    logger.info("creating agents")
    agent_list = []
    for i in range(args.num_agents):
        seed = args.agent_seed + i
        agent_list.append(
            # Agent.random(num_features, id=i, sphere_size=agent_sphere_size, seed=seed,)
            Agent.random_fixed_sum(num_features, id=i, seed=seed)
        )

    # ------------------ our methods ------------------

    for agent in agent_list:

        # -- MMU elicitation + MMU recommendation
        if args.obj_type == "maximin":
            logger.info("running MMU elicitation")
            get_write_agent_results(
                agent,
                items,
                "maximin",
                ["maximin"],
                gamma_unnormalized,
                rs,
                output_file,
            )

        # -- MMR elicitation + MMR recommendation
        if args.obj_type == "mmr":
            logger.info("running MMR elicitation")
            get_write_agent_results(
                agent, items, "mmr", ["mmr"], gamma_unnormalized, rs, output_file
            )

        # ------------------ other methods ------------------

        # -- AC elicitation + {MMU, MMR, AC} recommendation
        if args.obj_type == "maximin":
            ac_obj_list = ["maximin", "AC"]
        if args.obj_type == "mmr":
            ac_obj_list = ["mmr", "AC"]

        logger.info("running AC elicitation")
        get_write_agent_results(
            agent, items, "AC", ac_obj_list, gamma_unnormalized, rs, output_file
        )

        # -- random elicitation + {MMU, MMR} recommendation
        if args.obj_type == "maximin":
            rand_obj_list = ["maximin"]
        if args.obj_type == "mmr":
            rand_obj_list = ["mmr"]

        logger.info("running random elicitation")
        get_write_agent_results(
            agent, items, "random", rand_obj_list, gamma_unnormalized, rs, output_file
        )

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
        "--agent-seed",
        type=int,
        help="random seed for generating agents instances",
        default=0,
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0, help="level of agent inconsistencies"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=100,
        help="number of random agents to test elicitation on",
    )
    parser.add_argument(
        "--max-data-items", type=int, help="max number of items to read"
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--input-csv", type=str, help="csv of item data rto read")
    parser.add_argument(
        "--obj-type", type=str, help="{mmr | maximin} the problem type to evaluate"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=(60 * 60 * 3),
        help="time limit (in seconds) allowed for each stage (k) of the recommendation problem",
    )
    parser.add_argument(
        "--u0-type",
        type=str,
        default="positive_normed",
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
        arg_str = "--max-K 7"
        arg_str += " --obj-type mmr"
        arg_str += " --u0-type positive_normed"
        arg_str += " --agent-seed 100"
        arg_str += " --gamma 0.0"
        arg_str += " --time-limit 600"
        arg_str += " --num-agents 2"
        # arg_str += " --output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        arg_str += " --output-dir DEBUG_folder"
        # arg_str += " --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
        arg_str += " --input-csv DEBUG_folder/PolicyFeatures_RealData_LAHSA_normalized.csv"
        # arg_str += ' --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS_new.csv'
        arg_str += " --max-data-items 10"
        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    main()
