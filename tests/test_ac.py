# unit tests for calculating the AC of an uncertainty set

import unittest

import numpy as np

from preference_classes import generate_items, Agent, Query
from utils import find_analytic_center

PRINT_LOGS = True


class TestAC(unittest.TestCase):
    """contains an evaluation function and test problems to test AC calculation"""

    def evaluate_problem(self, problem_num):
        """evaluate a test problem using an elicitation function with the signature:
        queries, obj_val = elicitation_func(items, K)
        """
        if problem_num == 1:
            answered_queries, num_features, gamma, ac, u0_type = self.problem_1()
        elif problem_num == 2:
            answered_queries, num_features, gamma, ac, u0_type = self.problem_2()
        elif problem_num == 3:
            answered_queries, num_features, gamma, ac, u0_type = self.problem_3()
        elif problem_num == 4:
            answered_queries, num_features, gamma, ac, u0_type = self.problem_4()
        elif problem_num == 5:
            answered_queries, num_features, gamma, ac, u0_type = self.problem_5()
        else:
            raise Exception("invalid problem number")

        # calculate the AC
        u_vec = find_analytic_center(
            answered_queries, num_features, gamma, u0_type=u0_type
        )

        self.assertEqual(len(u_vec), num_features)

        for i in range(num_features):
            self.assertAlmostEqual(u_vec[i], ac[i], delta=1e-3)

    def problem_1(self):

        u0_type = "box"
        answered_queries = []
        num_features = 5
        ac = np.zeros(num_features)
        gamma = 0.0

        return answered_queries, num_features, gamma, ac, u0_type

    def problem_2(self):

        u0_type = "box"
        answered_queries = []
        num_features = 5
        ac = np.zeros(num_features)
        gamma = 1.3

        return answered_queries, num_features, gamma, ac, u0_type

    def problem_3(self):

        u0_type = "box"
        num_features = 3
        num_items = 15
        seed = 0
        item_sphere_size = 10
        agent_sphere_size = 1.0
        rs = np.random.RandomState(seed)
        gamma = 0.0
        ac = [0.33085463, 0.44720891, 0.33578232]

        items = generate_items(
            num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
        )

        agent = Agent.random(
            num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
        )

        a, b = rs.choice(num_items, 2, replace=False)
        agent.answer_query(Query(items[a], items[b]))

        return agent.answered_queries, num_features, gamma, ac, u0_type

    def problem_4(self):

        u0_type = "box"
        num_features = 6
        num_items = 20
        seed = 0
        item_sphere_size = 10
        agent_sphere_size = 1.0
        rs = np.random.RandomState(seed)
        gamma = 0.0
        ac = [0.85433585, -0.58266279, 0.72398708, 0.65220436, 0.07951611, -0.84196734]

        items = generate_items(
            num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
        )

        agent = Agent.random(
            num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
        )

        num_queries = 5
        for _ in range(num_queries):
            a, b = rs.choice(num_items, 2, replace=False)
            agent.answer_query(Query(items[a], items[b]))
            agent.answer_query(Query(items[a], items[b]))
            agent.answer_query(Query(items[a], items[b]))

        return agent.answered_queries, num_features, gamma, ac, u0_type

    def problem_5(self):

        u0_type = "positive_normed"
        num_features = 6
        num_items = 20
        seed = 0
        item_sphere_size = 10
        agent_sphere_size = 1.0
        rs = np.random.RandomState(seed)
        gamma = 0.0
        ac = [0.79649247, 0.01445363, 0.04484667, 0.123462, 0.01914137, 0.00160386]

        items = generate_items(
            num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
        )

        agent = Agent.random(
            num_features, id=0, sphere_size=agent_sphere_size, seed=seed,
        )

        num_queries = 5
        for _ in range(num_queries):
            a, b = rs.choice(num_items, 2, replace=False)
            agent.answer_query(Query(items[a], items[b]))
            agent.answer_query(Query(items[a], items[b]))
            agent.answer_query(Query(items[a], items[b]))

        return agent.answered_queries, num_features, gamma, ac, u0_type

    def test_ac_1(self):
        """check ac calculation on problem 1"""
        self.evaluate_problem(1)

    def test_ac_2(self):
        """check ac calculation on problem 2"""
        self.evaluate_problem(2)

    def test_ac_3(self):
        """check ac calculation on problem 3"""
        self.evaluate_problem(3)

    def test_ac_4(self):
        """check ac calculation on problem 4"""
        self.evaluate_problem(4)

    def test_ac_5(self):
        """check ac calculation on problem 5"""
        self.evaluate_problem(5)


if __name__ == "__main__":
    unittest.main()
