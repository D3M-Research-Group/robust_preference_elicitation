# compare the adaptive elicitation method to both random and AC on a single problem, on many simulated agents.

import argparse
import time
from collections import namedtuple

import numpy as np
from gurobipy import *
from tqdm import tqdm
from scipy.special import erfinv

from adaptive_elicitation import (
    next_optimal_query_mip,
    get_next_query_ac,
    get_next_query_ac_robust,
    get_random_query,
    get_next_query_polyhedral,
    get_next_query_probpoly,
)
from ellipsoidal import (
    get_next_query_exhaustive,
    initialize_uniform_prior,
    update_bayes_approximation,
)
from preference_classes import Query, Agent, generate_items
from recommendation import (
    solve_recommendation_problem,
    recommend_item_ac,
    recommend_item_ac_robust,
    recommend_item_mean,
)
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
    p_confidence = 0.9

    assert args.obj_type in ["maximin", "mmr"]

    # alpha for normalizing gamma
    inv_alpha = 0.5

    # generate an output file
    output_file = generate_filepath(args.output_dir, f"adaptive_experiment_{args.job_index}", "csv")
    log_file = generate_filepath(args.output_dir, f"adaptive_experiment_LOGS_{args.job_index}", "txt")
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
        "true_gamma",
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
            result.true_gamma,
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
        "robust_AC": lambda agent, gamma: recommend_item_ac_robust(
            agent.answered_queries, items, "mmu", gamma, args.u0_type
        ),
        # "mmr_AC": lambda agent, gamma: recommend_item_ac_robust(
        #     agent.answered_queries, items, "mmr", gamma, args.u0_type
        # ),
        "mmr_AC": lambda agent, gamma: solve_recommendation_problem(
            agent.answered_queries,
            items,
            "mmr",
            gamma=gamma,
            u0_type=args.u0_type,
            logger=logger,
        )[1],
        "mean": lambda agent, gamma: recommend_item_mean(agent.params["mu"], items,),
    }

    def get_write_agent_results(
        agent,
        items,
        elicitation_method,
        recommendation_methods,
        gamma_unnormalized,
        true_gamma_unnormalized,
        rs,
        output_file,
    ):
        """
        simulate elicitation with this agent.
        return a list of results, one for each k = 1, ..., K
        """

        assert elicitation_method in [
            "maximin",
            "mmr",
            "random",
            "AC",
            "robust_AC",
            "ellipsoidal",
            "polyhedral",
            "probpoly",
        ]
        assert set(recommendation_methods).issubset({"maximin", "mmr", "AC", "robust_AC", "mmr_AC", "mean"})

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
        if elicitation_method == "random":
            next_query = lambda agent, gamma: get_random_query(items, rs)
        if elicitation_method == "AC":
            next_query = lambda agent, gamma: get_next_query_ac_robust(
                agent.answered_queries, items, gamma, args.u0_type, rs
            )
        if elicitation_method == "ellipsoidal":
            a = -1.0
            b = 1.0
            agent.params["mu"], agent.params["cov"] = initialize_uniform_prior(
                a, b, num_features
            )
            next_query = lambda agent, gamma: get_next_query_exhaustive(
                agent.params["mu"], agent.params["cov"], items
            )
        if elicitation_method == "polyhedral":
            next_query = lambda agent, gamma: get_next_query_polyhedral(
                agent.answered_queries, items, gamma, args.u0_type, rs
            )
        if elicitation_method == "probpoly":
            next_query = lambda agent, gamma: get_next_query_probpoly(
                agent.answered_queries, items, gamma, args.u0_type
            )

        # reset the agent's answered queries
        agent.answered_queries = []
        np.random.seed(agent.id + 1)

        if gamma_unnormalized == 0:
            gamma = 0
            xi = np.random.normal(0.0, true_gamma_unnormalized, args.max_K)
            for k in range(args.max_K):
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : solving elicitation step {(k + 1)} of {args.max_K}"
                )
                # answer the next query
                t0 = time.time()
                q = next_query(agent, gamma)
                elicitation_time = time.time() - t0
                assert isinstance(q, Query)
                if true_gamma_unnormalized == 0:
                    agent.answer_query(q)
                else:
                    agent.answer_query(q, error=xi[k])

                if elicitation_method == "ellipsoidal":
                    # update agent mu and sigma
                    mu_new, cov_new = update_bayes_approximation(
                        agent.answered_queries[-1],
                        agent.params["mu"],
                        agent.params["cov"],
                    )
                    agent.params["mu"] = mu_new
                    agent.params["cov"] = cov_new

                # make recommendation(s)
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : solving rec."
                )
                # print("making recommendation")
                rec_items = {
                    rec_method: rec_methods_dict[rec_method](agent, gamma)
                    for rec_method in recommendation_methods
                }

                # evaluate recommendation(s)
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : evaluating rec."
                )
                for rec_method, rec_item in rec_items.items():
                    # print("agent:", agent.id, "K:", k, "rec_item:", rec_item.id)
                    # print("evaluating recommendation")

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
                            mmr_objval_normalized = (mmr_objval - min_feat_diff) / float(max_feat_diff - min_feat_diff)

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
                        true_gamma=true_gamma_unnormalized,
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
            # raise NotImplemented
            # gamma > 0: re-run elicitation for each k = 1, ..., K
            xi = np.random.normal(0.0, true_gamma_unnormalized, args.max_K)

            for k in range(args.max_K):
                logger.info(
                    f"agent {(agent.id + 1)} of {args.num_agents} : solving elicitation step {(k + 1)} of {args.max_K}"
                )
                num_queries = k + 1

                sigma_hat = np.sqrt(2.0 * num_queries * (gamma_unnormalized ** 2))
                gamma_normalized = (
                    sigma_hat * np.sqrt(2) * erfinv(2.0 * args.p_confidence - 1.0)
                )
                # print("gamma_normalized:", gamma_normalized)

                # calculate agent error before elicitation
                if elicitation_method in ["probpoly", "AC", "robust_AC"]:
                    alpha = 0
                    for i in range(100):
                        seed = args.agent_seed + i
                        if args.u0_type == "box":
                            agent_tmp = Agent.random(num_features, id=seed, sphere_size=agent_sphere_size, seed=seed,)
                        elif args.u0_type == "positive_normed":
                            agent_tmp = Agent.random_fixed_sum(num_features, id=seed, seed=seed)

                        alpha += agent_tmp.calculate_alpha(gamma_unnormalized, num_features, item_sphere_size=10, seed=seed)

                    gamma_normalized = 1 - alpha / 100

                # # reset elicitation
                # agent.answered_queries = []

                # answer the next query
                t0 = time.time()
                q = next_query(agent, gamma_normalized)
                elicitation_time = time.time() - t0
                if true_gamma_unnormalized == 0:
                    agent.answer_query(q)
                else:
                    agent.answer_query(q, error=xi[k])

                # # update agent u-set if using the ellipsoidal method
                # if elicitation_method == "ellipsoidal":
                #     mu_new, cov_new = update_bayes_approximation(
                #         agent.answered_queries[-1],
                #         agent.params["mu"],
                #         agent.params["cov"],
                #     )
                #     agent.params["mu"] = mu_new
                #     agent.params["cov"] = cov_new
        
                # make recommendation(s)
                # print("making recommendation")
                rec_items = {
                    rec_method: rec_methods_dict[rec_method](
                        agent, gamma_normalized
                    )
                    for rec_method in recommendation_methods
                }
        
                # evaluate recommendation(s)
                # print("evaluating recommendation")
                for rec_method, rec_item in rec_items.items():
                    # print("agent:", agent.id, "K:", k, "rec_item:", rec_item.id)
        
                    # evaluate MMU, MMR objval by solving the recommendation problem for this item only
                    if args.obj_type == "maximin":
                        mmu_objval, _ = solve_recommendation_problem(
                            agent.answered_queries,
                            items,
                            "maximin",
                            gamma=gamma_normalized,
                            fixed_rec_item=rec_item,
                            u0_type=args.u0_type,
                            logger=logger,
                        )
                        mmu_objval_normalized = (mmu_objval + max_norm) / (
                            2 * max_norm
                        )
                    else:
                        mmu_objval = None
                        mmu_objval_normalized = None
        
                    if args.obj_type == "mmr":
                        mmr_objval, _ = solve_recommendation_problem(
                            agent.answered_queries,
                            items,
                            "mmr",
                            gamma=gamma_normalized,
                            fixed_rec_item=rec_item,
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
                    # print("true_rank:", true_rank)
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
                        max_K=num_queries,
                        gamma=gamma_unnormalized,
                        gamma_normalized=gamma_normalized,
                        true_gamma=true_gamma_unnormalized,
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

        # return agent_results

    # read items
    if args.normalize:
        items = get_data_items(
            args.input_csv, max_items=args.max_data_items, standardize_features=False, normalize_features=True, drop_cols=["IsInterpretable_int"]
        )
    else:
        items = get_data_items(
            args.input_csv, max_items=args.max_data_items, standardize_features=False, normalize_features=False, drop_cols=["IsInterpretable_int"]
        )

    # logger.info("creating items")
    # # items = generate_items(
    # #     16,
    # #     30,
    # #     item_sphere_size=10,
    # #     seed=1,
    # #     positive=True
    # # )
    num_features = len(items[0].features)

    gamma_unnormalized = args.gamma
    true_gamma_unnormalized = args.true_gamma

    rs = np.random.RandomState(args.problem_seed)

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
    # for i in range(1):
    #     u_t = np.zeros(num_features)
    #     u_t[[8]] = 1
    #     agent_list.append(Agent(i, num_features, u_true=u_t))

    # ------------------ our methods ------------------

    if args.run_our_methods:

        # -- MMU elicitation + MMU recommendation
        if args.obj_type == "maximin":
            logger.info("running MMU elicitation")
            for agent in tqdm(agent_list):
                get_write_agent_results(
                    agent,
                    items,
                    "maximin",
                    ["maximin"],
                    gamma_unnormalized,
                    true_gamma_unnormalized,
                    rs,
                    output_file,
                )

        # -- MMR elicitation + MMR recommendation
        if args.obj_type == "mmr":
            logger.info("running MMR elicitation")
            for agent in tqdm(agent_list):
                get_write_agent_results(
                    agent, items, "mmr", ["mmr"], gamma_unnormalized, true_gamma_unnormalized, rs, output_file
                )

    # ------------------ other methods ------------------
    if args.run_ac:

        # -- AC elicitation + {MMU, MMR, AC} recommendation
        if args.obj_type == "maximin":
            ac_obj_list = ["robust_AC"]
        if args.obj_type == "mmr":
            ac_obj_list = ["mmr_AC"]

        logger.info("running AC elicitation")
        for agent in tqdm(agent_list):
            get_write_agent_results(
                agent, items, "AC", ac_obj_list, gamma_unnormalized, true_gamma_unnormalized, rs, output_file
            )

    if args.run_random:

        # -- random elicitation + {MMU, MMR} recommendation
        if args.obj_type == "maximin":
            rand_obj_list = ["maximin"]
        if args.obj_type == "mmr":
            rand_obj_list = ["mmr"]

        logger.info("running random elicitation")
        for agent in tqdm(agent_list):
            get_write_agent_results(
                agent, items, "random", rand_obj_list, gamma_unnormalized, true_gamma_unnormalized, rs, output_file
            )

    if args.run_ellipsoidal:

        # -- ellipsoidal elicitation + {MMU, MMR, AC} recommendation
        ellipsoidal_obj_list = ["mean"]

        logger.info("running ellipsoidal elicitation")
        for agent in tqdm(agent_list):
            get_write_agent_results(
                agent, items, "ellipsoidal", ellipsoidal_obj_list, gamma_unnormalized, true_gamma_unnormalized, rs, output_file
            )

    if args.run_polyhedral:

        # -- Polyhedral elicitation + AC recommendation
        poly_obj_list = ["AC"]

        logger.info("running Polyhedral elicitation")
        for agent in tqdm(agent_list):
            get_write_agent_results(
                agent, items, "polyhedral", poly_obj_list, gamma_unnormalized, true_gamma_unnormalized, rs, output_file
            )

    if args.run_probpoly:

        # -- Probabilistic polyhedral elicitation + AC recommendation
        probpoly_obj_list = ["AC"]

        logger.info("running Probabilistic Polyhedral elicitation")
        for agent in tqdm(agent_list):
            get_write_agent_results(
                agent, items, "probpoly", probpoly_obj_list, gamma_unnormalized, true_gamma_unnormalized, rs, output_file
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
        "--problem-seed",
        type=int,
        help="random seed for generating the problem instances",
        default=0,
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
        "--gamma", type=float, default=0.0, help="level of (supposed) agent inconsistencies"
    )
    parser.add_argument(
        "--true-gamma", type=float, default=0.0, help="level of true agent inconsistencies"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=100,
        help="number of random agents to test elicitation on",
    )

    # agent error
    parser.add_argument(
        "--p-confidence",
        type=float,
        default=0.9,
        help="confidence level for robustness to agent errors",
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

    # flags to run different methods
    parser.add_argument(
        "--run-our-methods",
        action="store_true",
        help="if set, use our elicitation methods",
        default=False,
    )
    parser.add_argument(
        "--run-ac",
        action="store_true",
        help="if set, use AC-based method (Toubia et al.)",
        default=False,
    )
    parser.add_argument(
        "--run-random",
        action="store_true",
        help="if set, use random elicitation",
        default=False,
    )
    parser.add_argument(
        "--run-ellipsoidal",
        action="store_true",
        help="if set, use ellipsiodal method (Vielma et al.)",
        default=False,
    )
    parser.add_argument(
        "--run-polyhedral",
        action="store_true",
        help="if set, use polyhedral method (Toubia et al.)",
        default=False,
    )
    parser.add_argument(
        "--run-probpoly",
        action="store_true",
        help="if set, use probabilistic polyhedral method (Toubia et al.)",
        default=False,
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="if set, use normalization",
        default=False,
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
        arg_str = "--max-K 3"
        arg_str += " --obj-type mmr"
        arg_str += " --u0-type positive_normed"
        # arg_str += " --agent-seed 100"
        arg_str += " --agent-seed 0"
        arg_str += " --gamma 2.5"
        arg_str += " --true-gamma 1.5"
        arg_str += " --time-limit 3600"
        arg_str += " --num-agents 3"
        # arg_str += " --run-our-methods --run-ac --run-random --run-ellipsoidal --run-polyhedral --run-probpoly"
        arg_str += " --run-our-methods"
        # arg_str += " --run-random --run-ellipsoidal --run-ac --run-probpoly"
        # arg_str += " --output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        arg_str += " --output-dir DEBUG_folder"
        # arg_str += " --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS-small_new.csv"
        # arg_str += " --input-csv DEBUG_folder/PolicyFeatures_RealData_LAHSA_normalized.csv"
        arg_str += " --input-csv test_results/AdultHMIS_20210906_preprocessed_final_Robust_25.csv"
        # arg_str += ' --input-csv /Users/duncan/research/ActivePreferenceLearning/data/PolicyFeatures_RealData_HMIS_new.csv'
        arg_str += " --max-data-items 50"  # --gamma 1.85 --true-gamma 2.22
        arg_str = "--max-K 5 --obj-type mmr --normalize --u0-type positive_normed --gamma 0.02 --true-gamma 0.02 --time-limit 3600 --max-data-items 50 --num-agents 50 --run-our-methods --input-csv AdultHMIS_20210906_preprocessed_final_Robust_41.csv --output-dir DEBUG_folder"
        arg_str = "--max-K 10 --obj-type mmr --normalize --u0-type positive_normed --gamma 0.00 --true-gamma 0.00 --time-limit 3600 --max-data-items 50 --num-agents 50 --run-polyhedral --input-csv AdultHMIS_20210922_preprocessed_final_Robust_all25.csv --output-dir DEBUG_folder"
        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    main()
