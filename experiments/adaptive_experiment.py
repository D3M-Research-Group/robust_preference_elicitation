# adaptive elicitation experiment, where parameter sets (num items, num features, sigma) are passed as a list

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
from preference_classes import Query, generate_items, Agent
from recommendation import (
    solve_recommendation_problem,
    recommend_item_ac,
    recommend_item_ac_robust,
    recommend_item_mean,
)
from utils import generate_filepath, get_logger

# however the updated normalization should not change the results in previously-run experiments.


def experiment(args):
    """
    compare the adaptive elicitation method to both random and AC on a single problem, on many simulated agents.
    for each agent, simulate the following (elicitation, recommendation pair):

    first, our approaches
    
    1. MMU elicitation + MMU recommendation
    2. MMR elicitation + MMR recommendation

    and the following "other" approaches:

    1. AC elicitation + AC recommendation
    2. AC elicitation + MMU recommendation
    3. AC elicitation + MMR recommendation
    4. Random elicitation + MMU recommendation
    5. Random elicitation + MMR recommendation
    6. Ellipsoidal elicitation + ellipsoidal recommendation
    7. Polyhedral elicitation + AC recommendation
    8. Probabilistic-polyhedral elicitation + AC recommendation

    Where AC elicitation selects the next query closest to the agent's current analytic center (AC),
    and AC recommendation assumes that the agent's true u-vector is the current AC - and selects the highest-utility item.

    After making a recommendation, we record the following:
    - recommendation objval (MMU or MMR, depending on rec. approach)
    - true objval (agent's "true" utility of the rec. for MMU, and "true" regret for MMR)
    - true item rank
    """

    # some fixed parameters
    item_sphere_size = 10.0
    agent_sphere_size = 1.0
    query_seed = 0

    assert args.obj_type in ["maximin", "mmr"]

    if args.u0_type == "positive_normed":
        # have not corrected the objval normalization for this
        raise NotImplemented

    # generate an output file
    output_file = generate_filepath(args.output_dir, "adaptive_experiment", "csv")
    log_file = generate_filepath(args.output_dir, "adaptive_experiment_LOGS", "txt")
    logger = get_logger(logfile=log_file)

    logger.info("generating output file: {}".format(output_file))
    logger.info("generating log file: {}".format(log_file))

    param_sets = (
        args.param_sets
    )  # list of tuples of the form (num_items, num_features, sigma)

    col_list = [
        "problem_seed",
        "num_features",
        "num_items",
        "item_sphere_size",
        "elicitation_method",
        "recommendation_method",
        "query_seed",
        "K",
        "max_K",
        "sigma",
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
            result.problem_seed,
            result.num_features,
            result.num_items,
            result.item_sphere_size,
            result.elicitation_method,
            result.recommendation_method,
            result.query_seed,
            result.K,
            result.max_K,
            result.sigma,
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
            agent.answered_queries, items, "maximin", gamma=gamma, u0_type=args.u0_type,
        )[1],
        "mmr": lambda agent, gamma: solve_recommendation_problem(
            agent.answered_queries, items, "mmr", gamma=gamma, u0_type=args.u0_type,
        )[1],
        "AC": lambda agent, gamma: recommend_item_ac(
            agent.answered_queries, items, gamma, args.u0_type
        ),
        "robust_AC": lambda agent, gamma: recommend_item_ac_robust(
            agent.answered_queries, items, "mmu", gamma, args.u0_type
        ),
        "mmr_AC": lambda agent, gamma: recommend_item_ac_robust(
            agent.answered_queries, items, "mmr", gamma, args.u0_type
        ),
        "mean": lambda agent, gamma: recommend_item_mean(agent.params["mu"], items,),
    }

    def get_agent_results(
        agent, items, elicitation_method, recommendation_methods, sigma, rs
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

        agent_results = []
        num_features = len(items[0].features)

        if elicitation_method in ["maximin", "mmr"]:
            next_query = lambda agent, gamma: next_optimal_query_mip(
                agent.answered_queries,
                items,
                elicitation_method,
                gamma,
                u0_type=args.u0_type,
                logger=logger,
            )[1]
        if elicitation_method == "AC":
            next_query = lambda agent, gamma: get_next_query_ac(
                agent.answered_queries, items, gamma, args.u0_type
            )
        if elicitation_method == "robust_AC":
            next_query = lambda agent, gamma: get_next_query_ac_robust(
                agent.answered_queries, items, gamma, args.u0_type
            )
        if elicitation_method == "random":
            next_query = lambda agent, gamma: get_random_query(items, rs)
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

        if sigma == 0:
            gamma = 0
            for k in range(args.max_K):

                # answer the next query
                t0 = time.time()
                q = next_query(agent, gamma)
                elicitation_time = time.time() - t0
                agent.answer_query(q)

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
                rec_items = {
                    rec_method: rec_methods_dict[rec_method](agent, gamma)
                    for rec_method in recommendation_methods
                }

                # evaluate recommendation(s)
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
                        )
                        mmu_objval_normalized = (mmu_objval + max_norm) / (2 * max_norm)
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

                    agent_results.append(
                        Result(
                            problem_seed=args.problem_seed,
                            num_features=num_features,
                            num_items=len(items),
                            item_sphere_size=item_sphere_size,
                            query_seed=query_seed,
                            elicitation_method=elicitation_method,
                            recommendation_method=rec_method,
                            K=len(agent.answered_queries),
                            max_K=args.max_K,
                            sigma=sigma,
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
                    )
        else:

            xi = np.random.normal(0.0, sigma, args.max_K)

            for k in range(args.max_K):
                num_queries = k + 1

                sigma_hat = np.sqrt(2.0 * num_queries * (sigma ** 2))
                gamma_normalized = (
                    sigma_hat * np.sqrt(2) * erfinv(2.0 * args.p_confidence - 1.0)
                )

                # set gamma = 0 when using polyhedral
                if elicitation_method == "polyhedral":
                    gamma_normalized = 0

                # correct the error term in probpoly and in AC
                if elicitation_method in ["probpoly", "AC", "robust_AC"]:
                    alpha = 0
                    for i in range(100):
                        seed = args.agent_seed + i
                        if args.u0_type == "box":
                            agent_tmp = Agent.random(num_features, id=seed, sphere_size=agent_sphere_size, seed=seed,)
                        elif args.u0_type == "positive_normed":
                            agent_tmp = Agent.random_fixed_sum(num_features, id=seed, seed=seed)

                        alpha += agent_tmp.calculate_alpha(sigma, num_features, item_sphere_size=item_sphere_size, seed=seed)

                    gamma_normalized = 1 - alpha / 100

                # answer the next query
                t0 = time.time()
                q = next_query(agent, gamma_normalized)
                elicitation_time = time.time() - t0
                agent.answer_query(q, error=xi[k])

                # update agent u-set if using the ellipsoidal method
                if elicitation_method == "ellipsoidal":
                    mu_new, cov_new = update_bayes_approximation(
                        agent.answered_queries[-1],
                        agent.params["mu"],
                        agent.params["cov"],
                    )
                    agent.params["mu"] = mu_new
                    agent.params["cov"] = cov_new

                # make recommendation(s)
                rec_items = {
                    rec_method: rec_methods_dict[rec_method](agent, gamma_normalized)
                    for rec_method in recommendation_methods
                }

                # evaluate recommendation(s)
                for rec_method, rec_item in rec_items.items():

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
                        mmu_objval_normalized = (mmu_objval + max_norm) / (2 * max_norm)
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
                    true_u = agent.true_utility(rec_item)
                    true_u_normalized = (true_u + max_norm) / (2 * max_norm)
                    true_regret = agent.true_item_max_regret(rec_item, items)
                    true_regret_normalized = (true_regret + max_diff_norm) / (
                        2 * max_diff_norm
                    )

                    agent_results.append(
                        Result(
                            problem_seed=args.problem_seed,
                            num_features=num_features,
                            num_items=len(items),
                            item_sphere_size=item_sphere_size,
                            query_seed=query_seed,
                            elicitation_method=elicitation_method,
                            recommendation_method=rec_method,
                            K=len(agent.answered_queries),
                            max_K=num_queries,
                            sigma=sigma,
                            gamma_normalized=gamma_normalized,
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
                    )

        return agent_results

    # keep track of all Result objects in a list
    result_list = []

    for num_items, num_features, sigma in param_sets:

        num_items = int(num_items)
        num_features = int(num_features)
        rs = np.random.RandomState(args.problem_seed)

        logger.info(
            "starting parameter set: n-items={}, n-features={}, sigma={}".format(
                num_items, num_features, sigma
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

        # get the max 1-norm *difference* between items (for mmr)
        max_diff_norm = max(
            [
                np.sum(np.abs(i.features - j.features))
                for i, j in itertools.combinations(items, 2)
            ]
        )

        logger.info("creating agents")
        agent_list = []
        for i in range(args.num_agents):
            seed = args.agent_seed + i
            if args.u0_type == "box":
                agent_list.append(
                    Agent.random(
                        num_features, id=seed, sphere_size=agent_sphere_size, seed=seed,
                    )
                )
            elif args.u0_type == "positive_normed":
                agent_list.append(
                    Agent.random_fixed_sum(num_features, id=seed, seed=seed)
                )
            else:
                raise Exception("u0-type not valid")

        # ------------------ our methods ------------------

        if args.run_our_methods:

            # -- MMU elicitation + MMU recommendation
            if args.obj_type == "maximin":
                logger.info("running MMU elicitation")
                for agent in tqdm(agent_list):
                    agent_results = get_agent_results(
                        agent, items, "maximin", ["maximin"], sigma, rs
                    )
                    with open(output_file, "a") as f:
                        for result in agent_results:
                            f.write(result_to_str(result))

            # -- MMR elicitation + MMR recommendation
            if args.obj_type == "mmr":
                logger.info("running MMR elicitation")
                for agent in tqdm(agent_list):
                    agent_results = get_agent_results(
                        agent, items, "mmr", ["mmr"], sigma, rs
                    )
                    with open(output_file, "a") as f:
                        for result in agent_results:
                            f.write(result_to_str(result))

        # ------------------ other methods ------------------

        if args.run_ac:
            # -- AC elicitation + {MMU, MMR, AC} recommendation
            if args.obj_type == "maximin":
                # ac_obj_list = ["maximin", "AC"]
                ac_obj_list = ["robust_AC"]
            if args.obj_type == "mmr":
                # ac_obj_list = ["mmr", "AC"]
                ac_obj_list = ["mmr_AC"]

            logger.info("running AC elicitation")
            tmp_results = []
            for agent in tqdm(agent_list):
                # tmp_results.extend(
                #     get_agent_results(agent, items, "AC", ac_obj_list, sigma, rs)
                # )
                tmp_results.extend(
                    get_agent_results(agent, items, "robust_AC", ac_obj_list, sigma, rs)
                )

            logger.info("writing AC results to file")
            with open(output_file, "a") as f:
                for result in tmp_results:
                    f.write(result_to_str(result))

        if args.run_random:

            # -- random elicitation + {MMU, MMR} recommendation
            if args.obj_type == "maximin":
                rand_obj_list = ["maximin"]
            if args.obj_type == "mmr":
                rand_obj_list = ["mmr"]

            logger.info("running random elicitation")
            tmp_results = []
            for agent in tqdm(agent_list):
                tmp_results.extend(
                    get_agent_results(agent, items, "random", rand_obj_list, sigma, rs)
                )

            logger.info("writing random results to file")
            with open(output_file, "a") as f:
                for result in tmp_results:
                    f.write(result_to_str(result))

        if args.run_ellipsoidal:

            # -- ellipsoidal elicitation + {MMU, MMR, AC} recommendation
            ellipsoidal_obj_list = ["mean"]

            logger.info("running ellipsoidal elicitation")
            tmp_results = []
            for agent in tqdm(agent_list):
                tmp_results.extend(
                    get_agent_results(
                        agent, items, "ellipsoidal", ellipsoidal_obj_list, sigma, rs,
                    )
                )

            logger.info("writing ellipsoidal results to file")
            with open(output_file, "a") as f:
                for result in tmp_results:
                    f.write(result_to_str(result))

        if args.run_polyhedral:

            # -- Polyhedral elicitation + AC recommendation
            poly_obj_list = ["AC"]

            logger.info("running Polyhedral elicitation")
            tmp_results = []
            for agent in tqdm(agent_list):
                tmp_results.extend(
                    get_agent_results(
                        agent, items, "polyhedral", poly_obj_list, sigma, rs
                    )
                )

            logger.info("writing Polyhedral results to file")
            with open(output_file, "a") as f:
                for result in tmp_results:
                    f.write(result_to_str(result))

        if args.run_probpoly:

            # -- Probabilistic polyhedral elicitation + AC recommendation
            probpoly_obj_list = ["AC"]

            logger.info("running Probabilistic Polyhedral elicitation")
            tmp_results = []
            for agent in tqdm(agent_list):
                tmp_results.extend(
                    get_agent_results(
                        agent, items, "probpoly", probpoly_obj_list, sigma, rs
                    )
                )

            logger.info("writing Probabilistic Polyhedral results to file")
            with open(output_file, "a") as f:
                for result in tmp_results:
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
        "--problem-seed",
        type=int,
        help="random seed for generating the problem instances",
        default=0,
    )
    parser.add_argument(
        "--u0-type",
        type=str,
        help='type of agent uncertainty set, either "box", or "positive_normed"',
        default="box",
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        help="random seed for generating agents instances",
        default=0,
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=100,
        help="number of random agents to test elicitation on",
    )
    parser.add_argument(
        "-p",
        "--param-sets",
        action="append",
        type=float,
        nargs="+",
        help="add a parameter set of the form (num_items, num_features, sigma)",
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument(
        "--obj-type", type=str, help="{mmr | maximin} the problem type to evaluate"
    )

    # agent error
    parser.add_argument(
        "--p-confidence",
        type=float,
        default=0.9,
        help="confidence level for robustness to agent errors",
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
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string. otherwise, parse args.",
        default=False,
    )

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:
        # arg_str = "--max-K 10  --obj-type mmr --agent-seed 0  --problem-seed 0 " \
        #           "--num-agents 3 " \
        #           "-p 7 5 0.0 -p 7 5 0.9 " \
        #           "--run-our-methods " \
        #           "--run-polyhedral "
        # "-p 7 5 0.0 -p 10 5 0.0 -p 20 5 0.0 -p 10 3 0.0 -p 10 10 0.0 -p 10 5 0.0 -p 10 5 0.1 -p 10 5 0.2 " \

        arg_str = "--max-K 10 --obj-type maximin --agent-seed 0 --problem-seed 0 --num-agents 50 -p 40 10 2.0 --run-polyhedral"
        # arg_str = "--max-K 10  --obj-type maximin --agent-seed 0  --problem-seed 0 --num-agents 100 -p 30 10 0.0 -p 40 10 0.0 -p 60 10 0.0 -p 40 5 0.0 -p 40 20 0.0  -p 40 10 0.1 -p 40 10 0.2 --run-polyhedral --run-probpoly"
        # arg_str += " --run-polyhedral --run-probpoly"
        # arg_str += " -p 3 10 0.1"
        # arg_str += " -p 3 10 0.2"
        # arg_str += " --sigma 0.1 0.2"
        # arg_str += " --output-dir /Users/duncan/research/ActivePreferenceLearning/RobustActivePreferenceLearning_output/tmp"
        # arg_str += " --output-dir C:\\Users\\yyx_v\\Desktop"
        arg_str += " --output-dir DEBUG_folder"
        # arg_str += " --output-dir D:\Dropbox\Research\Prefercence Elicitation\review_1"
        # arg_str += " --output-dir /Users/yingxiao.ye/Desktop"
        args_fixed = parser.parse_args(arg_str.split())

        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)


if __name__ == "__main__":
    # import cProfile

    # cProfile.run("main()")  #  main() cProfile.run("experiment(args_fixed)")
    main()
