# unit tests for the robust rec-learning problem (minimax regret objective), without agent inconsistencies.

import unittest

import numpy as np

from preference_classes import generate_items, Item
from static_elicitation import static_mip_optimal, solve_warm_start, evaluate_query_list_mip
from scenario_decomposition import solve_scenario_decomposition, solve_warm_start_decomp

PRINT_LOGS = True


class BaseTestClass:
    class TestStaticElicitation(unittest.TestCase):
        """contains an evaluation function and test problems to test different static elicitation methods"""

        def evaluate_problem(self, problem_num, elicitation_func):
            """evaluate a test problem using an elicitation function with the signature:
            queries, obj_val = elicitation_func(items, K)
            """
            if problem_num == 1:
                items, K, optimal_objval, valid_responses, gamma = self.problem_1()
            elif problem_num == 2:
                items, K, optimal_objval, valid_responses, gamma = self.problem_2()
            elif problem_num == 3:
                items, K, optimal_objval, valid_responses, gamma = self.problem_3()
            else:
                raise Exception("invalid problem number")

            # run the elicitation method
            queries, obj_val = elicitation_func(items, K, valid_responses, gamma)

            # the method should return K queries
            self.assertEqual(len(queries), K)

            # the method should achieve the optimal objective value
            self.assertAlmostEqual(obj_val, optimal_objval, delta=5e-2)

            # validate the objective value by exhaustively checking the response scenarios & items
            true_objval = evaluate_query_list_mip(queries, items, valid_responses,
                                                  problem_type='mmr',
                                                  gamma_inconsistencies=gamma)
            self.assertAlmostEqual(true_objval, optimal_objval, delta=1e-3)

        def problem_1(self):
            K = 1
            valid_responses = [-1, 1]
            gamma = 0.1

            items = [
                Item([1.0, -0.2, 0.0], 0),
                Item([0.4, 0.0, 0.0], 1),
                Item([0.0, 0.5, -0.001], 2),
                Item([0.1, -1.0, 0.7], 3),
                Item([-0.5, 0.5, 0.5], 4),
            ]

            optimal_objval = 1.3327995386095253

            return items, K, optimal_objval, valid_responses, gamma

        def problem_2(self):
            K = 2
            seed = 1
            num_features = 4
            num_items = 10
            item_sphere_size = 10
            valid_responses = [-1, 1]
            gamma = 0.3

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            optimal_objval = 17.118300965058964

            return items, K, optimal_objval, valid_responses, gamma

        def problem_3(self):
            K = 3
            seed = 1
            num_features = 4
            num_items = 4
            item_sphere_size = 10
            valid_responses = [-1, 1]
            gamma = 1.3

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            optimal_objval = 4.944390596531711

            return items, K, optimal_objval, valid_responses, gamma


class TestMIPCuts(BaseTestClass.TestStaticElicitation):
    """check that the MIP formulation with cuts solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _, _ = static_mip_optimal(items, K, valid_responses,
                                                   cut_1=True,
                                                   cut_2=True,
                                                   problem_type='mmr',
                                                   gamma_inconsistencies=gamma,
                                                   )
        return queries, objval

    def test_mipcuts_1(self):
        """check MIP on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)

    def test_mipcuts_2(self):
        """check MIP on problem 2"""
        self.evaluate_problem(2, self.elicitation_func)

    def test_mipcuts_3(self):
        """check MIP on problem 2"""
        self.evaluate_problem(3, self.elicitation_func)


class TestWarmStart(BaseTestClass.TestStaticElicitation):
    """check that the warm start formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start(items, K, valid_responses,
                                              cut_1=True,
                                              print_logs=PRINT_LOGS,
                                              problem_type='mmr',
                                              gamma_inconsistencies=gamma,
                                              )
        return queries, objval

    def test_warm_1(self):
        """check warm start on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)

    def test_warm_2(self):
        """check warm start on problem 2"""
        self.evaluate_problem(2, self.elicitation_func)

    def test_warm_3(self):
        """check warm start on problem 3"""
        self.evaluate_problem(3, self.elicitation_func)


class TestScenarioDecomp(BaseTestClass.TestStaticElicitation):
    """check that the warm start formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        rs = np.random.RandomState()
        queries, objval, _ = solve_scenario_decomposition(items, K, rs, valid_responses,
                                                             max_iter=100,
                                                             print_logs=PRINT_LOGS,
                                                             problem_type='mmr',
                                                             time_limit=1e10,
                                                             gamma_inconsistencies=gamma,
                                                             )
        return queries, objval

    def test_decomp_1(self):
        """check decomp on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)

    def test_decomp_2(self):
        """check decomp on problem 2"""
        self.evaluate_problem(2, self.elicitation_func)

    def test_decomp_3(self):
        """check decomp on problem 3"""
        self.evaluate_problem(3, self.elicitation_func)


class TestWarmStartScenarioDecomp(BaseTestClass.TestStaticElicitation):
    """check that the warm start + scenario decomp formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start_decomp(items, K, valid_responses,
                                                     print_logs=PRINT_LOGS,
                                                     problem_type='mmr',
                                                     cut_1=True,
                                                     gamma_inconsistencies=gamma,
                                                     )
        return queries, objval

    def test_warm_decomp_1(self):
        """check warm start + decomp on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)

    def test_warm_decomp_2(self):
        """check warm start + decomp on problem 2"""
        self.evaluate_problem(2, self.elicitation_func)

    def test_warm_decomp_3(self):
        """check warm start + decomp on problem 3"""
        self.evaluate_problem(3, self.elicitation_func)


if __name__ == '__main__':
    unittest.main()
