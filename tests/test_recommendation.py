# unit tests for the functions related to recommendation

import unittest

import numpy as np

from preference_classes import generate_items, Agent, Query
from recommendation import solve_recommendation_problem

PRINT_LOGS = True


class BaseTestClass:
    class TestRecommendation(unittest.TestCase):
        """contains an evaluation function and test problems to test different recommendation methods"""

        def evaluate_recommendation(self, problem_num, recommendation_func):
            """evaluate a test problem using an elicitation function with the signature:
            queries, obj_val = elicitation_func(items, K)
            """
            if problem_num == 1:
                (
                    answered_queries,
                    items,
                    gamma,
                    problem_type,
                    objval_opt,
                ) = self.problem_1()
            elif problem_num == 2:
                (
                    answered_queries,
                    items,
                    gamma,
                    problem_type,
                    objval_opt,
                ) = self.problem_2()
            elif problem_num == 3:
                (
                    answered_queries,
                    items,
                    gamma,
                    problem_type,
                    objval_opt,
                ) = self.problem_3()
            elif problem_num == 4:
                (
                    answered_queries,
                    items,
                    gamma,
                    problem_type,
                    objval_opt,
                ) = self.problem_4()
            else:
                raise Exception("invalid problem number")

            # run the elicitation method
            objval, recommended_item = recommendation_func(
                answered_queries, items, problem_type, gamma
            )

            # the method should return K queries
            self.assertAlmostEqual(objval, objval_opt, delta=5e-2)

        def problem_1(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 10
            num_items = 50
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 0
            problem_type = "maximin"

            items = generate_items(
                num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
            )

            agent = Agent.random(
                num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
            )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                q = Query(item_a, item_b)
                agent.answer_query(q)

            objval = -4.254660047880414

            return agent.answered_queries, items, gamma, problem_type, objval

        def problem_2(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 10
            num_items = 50
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 2.4
            problem_type = "maximin"

            items = generate_items(
                num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
            )

            agent = Agent.random(
                num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
            )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                q = Query(item_a, item_b)
                agent.answer_query(q)

            objval = -5.676333538743321

            return agent.answered_queries, items, gamma, problem_type, objval

        def problem_3(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 10
            num_items = 50
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 0
            problem_type = "mmr"

            items = generate_items(
                num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
            )

            agent = Agent.random(
                num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
            )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                q = Query(item_a, item_b)
                agent.answer_query(q)

            objval = 21.356420941306403

            return agent.answered_queries, items, gamma, problem_type, objval

        def problem_4(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 10
            num_items = 50
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 3.0
            problem_type = "mmr"

            items = generate_items(
                num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
            )

            agent = Agent.random(
                num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
            )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                q = Query(item_a, item_b)
                agent.answer_query(q)

            objval = 23.150214834546016

            return agent.answered_queries, items, gamma, problem_type, objval


class TestMMURec(BaseTestClass.TestRecommendation):
    """check that the MIP formulation without cuts solves the test problems"""

    def rec_func(self, answered_queries, items, problem_type, gamma):
        objval, recommended_item = solve_recommendation_problem(
            answered_queries, items, problem_type, gamma=gamma, verbose=False
        )

        return objval, recommended_item

    def test_heuristic_maximin_1(self):
        """check heuristic on problem 1"""
        self.evaluate_recommendation(1, self.rec_func)

    def test_heuristic_maximin_2(self):
        """check heuristic on problem 2"""
        self.evaluate_recommendation(2, self.rec_func)

    def test_heuristic_maximin_3(self):
        """check heuristic on problem 3"""
        self.evaluate_recommendation(3, self.rec_func)

    def test_heuristic_maximin_4(self):
        """check heuristic on problem 4"""
        self.evaluate_recommendation(4, self.rec_func)


if __name__ == "__main__":
    unittest.main()
