# unit tests for the robust rec-learning problem (maximin objective), with budgeted agent inconsistencies.

import unittest

import numpy as np

from preference_classes import generate_items, Item
from static_elicitation import static_mip_optimal, solve_warm_start, evaluate_query_list_mip
from scenario_decomposition import solve_scenario_decomposition, solve_warm_start_decomp

PRINT_LOGS = False


class BaseTestClass:

    class TestStaticElicitationInconsistencies(unittest.TestCase):
        """contains an evaluation function and test problems to test different static elicitation methods"""

        def evaluate_problem(self, problem_num, elicitation_func):
            """evaluate a test problem using an elicitation function with the signature:
            queries, obj_val = elicitation_func(items, K)
            """
            if problem_num == 1:
                items, gamma, K, optimal_objval, valid_responses = self.problem_1()
            elif problem_num == 2:
                items, gamma, K, optimal_objval, valid_responses = self.problem_2()
            elif problem_num == 3:
                items, gamma, K, optimal_objval, valid_responses = self.problem_3()
            else:
                raise Exception("invalid problem number")

            # run the elicitation method
            queries, obj_val = elicitation_func(items, K, valid_responses, gamma)

            # the method should return K queries
            self.assertEqual(len(queries), K)

            # the method should achieve the optimal objective value
            self.assertAlmostEqual(obj_val, optimal_objval, delta=1e-2)

            # validate the objective value by exhaustively checking the response scenarios & items
            true_objval = evaluate_query_list_mip(queries, items, valid_responses,
                                                  problem_type='maximin',
                                                  gamma_inconsistencies=gamma)
            self.assertAlmostEqual(true_objval, optimal_objval, delta=1e-2)

        def problem_1(self):
            gamma = 0.3
            K = 3
            valid_responses = [-1, 1]

            items = [
                Item([1.0, -0.2, 0.0], 0),
                Item([0.4, 0.0, 0.0], 1),
                Item([0.0, 0.5, -0.001], 2),
                Item([0.1, -1.0, 0.7], 3),
                Item([-0.5, 0.5, 0.5], 4),
            ]

            optimal_objval = -0.4

            return items, gamma, K, optimal_objval, valid_responses

        def problem_2(self):
            K = 3
            gamma = 0.03
            seed = 1
            num_features = 4
            num_items = 6
            item_sphere_size = 10
            valid_responses = [-1, 1]

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            optimal_objval = -1.4646973476123983

            return items, gamma, K, optimal_objval, valid_responses

        def problem_3(self):
            K = 3
            gamma = 0.9
            seed = 1
            num_features = 4
            num_items = 4
            item_sphere_size = 10
            valid_responses = [-1, 1]

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            optimal_objval = -13.898705765106975 # -13.369138793131668

            return items, gamma, K, optimal_objval, valid_responses


class TestMIPCuts(BaseTestClass.TestStaticElicitationInconsistencies):
    """check that the MIP formulation with cuts solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _, _ = static_mip_optimal(items, K, valid_responses,
                                                   cut_1=True,
                                                   cut_2=True,
                                                   problem_type='maximin',
                                                   gamma_inconsistencies=gamma)
        return queries, objval

    def test_mipcuts_1(self):
        """check MIP on problem 1"""
        self.evaluate_problem(1, self.elicitation_func)

    def test_mipcuts_2(self):
        """check MIP on problem 2"""
        self.evaluate_problem(2, self.elicitation_func)

    def test_mipcuts_3(self):
        """check MIP on problem 3"""
        self.evaluate_problem(3, self.elicitation_func)


class TestWarmStart(BaseTestClass.TestStaticElicitationInconsistencies):
    """check that the warm start formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start(items, K, valid_responses,
                                              cut_1=True,
                                              print_logs=PRINT_LOGS,
                                              problem_type='maximin',
                                              gamma_inconsistencies=gamma)
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


class TestScenarioDecomp(BaseTestClass.TestStaticElicitationInconsistencies):
    """check that the warm start formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        rs = np.random.RandomState()
        queries, objval, _ = solve_scenario_decomposition(items, K, rs, valid_responses,
                                                             max_iter=100,
                                                             print_logs=PRINT_LOGS,
                                                             time_limit=1e10,
                                                             problem_type='maximin',
                                                             gamma_inconsistencies=gamma)
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


class TestWarmStartScenarioDecomp(BaseTestClass.TestStaticElicitationInconsistencies):
    """check that the warm start + scenario decomp formulation solves the test problems"""

    def elicitation_func(self, items, K, valid_responses, gamma):
        queries, objval, _ = solve_warm_start_decomp(items, K, valid_responses,
                                                     cut_1=True,
                                                     print_logs=PRINT_LOGS,
                                                     problem_type='maximin',
                                                     gamma_inconsistencies=gamma)
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
