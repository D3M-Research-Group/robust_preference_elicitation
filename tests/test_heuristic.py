# unit tests for the robust rec-learning problem (maximin objective), heuristic

import unittest

import numpy as np

from preference_classes import generate_items, Item
from static_elicitation import static_mip_optimal, solve_warm_start, evaluate_query_list_mip
from scenario_decomposition import solve_scenario_decomposition, solve_warm_start_decomp
from static_heuristics import solve_warm_start_decomp_heuristic

PRINT_LOGS = True


class BaseTestClass:
    class TestStaticHeuristic(unittest.TestCase):
        """contains an evaluation function and test problems to test different static elicitation methods"""

        def evaluate_problem(self, problem_num, elicitation_func):
            """evaluate a test problem using an elicitation function with the signature:
            queries, obj_val = elicitation_func(items, K)
            """
            if problem_num == 1:
                items, K, valid_responses, gamma = self.problem_1()
            else:
                raise Exception("invalid problem number")

            # run the elicitation method
            queries, obj_val = elicitation_func(items, K, valid_responses, gamma)

            # the method should return K queries
            self.assertEqual(len(queries), K)

        def problem_1(self):

            valid_responses = [-1, 1]
            num_features = 3
            num_items = 15
            gamma_inconsistencies = 0
            seed = 0
            item_sphere_size = 10
            K = 5

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            # get the max 1-norm of all items
            max_norm = max([np.sum(np.abs(i.features)) for i in items])

            # normalize gamma
            inv_alpha = 0.5
            gamma = 2.0 * gamma_inconsistencies * np.power(K, inv_alpha) * max_norm

            return items, K, valid_responses, gamma


class TestWarmStartDecompMaximin(BaseTestClass.TestStaticHeuristic):
    """check that the MIP formulation without cuts solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start_decomp_heuristic(items, K,
                                                               valid_responses,
                                                               print_logs=PRINT_LOGS,
                                                               problem_type='maximin',
                                                               gamma_inconsistencies=gamma)
        return queries, objval[-1]

    def test_heuristic_maximin_1(self):
        """check heuristic on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)


class TestWarmStartDecompMMR(BaseTestClass.TestStaticHeuristic):
    """check that the MIP formulation without cuts solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start_decomp_heuristic(items, K,
                                                               valid_responses,
                                                               print_logs=PRINT_LOGS,
                                                               problem_type='mmr',
                                                               gamma_inconsistencies=gamma)
        return queries, objval[-1]

    def test_heuristic_mmr_1(self):
        """check heuristic on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)


if __name__ == '__main__':
    unittest.main()
