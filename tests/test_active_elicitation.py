# unit tests for the functions related to adaptive elicitation
import itertools
import time
import unittest

import numpy as np

from adaptive_elicitation import next_optimal_query_iterative, next_optimal_query_mip
from preference_classes import generate_items, Agent, Query

PRINT_LOGS = True


class BaseTestClass:
    class TestGetNextQuery(unittest.TestCase):
        """contains an evaluation function and test problems to test different recommendation methods"""

        def evaluate_next_query(self, problem_num, query_func):
            """evaluate a test problem using an elicitation function with the signature:
            queries, obj_val = elicitation_func(items, K)
            """
            if problem_num == 1:
                answered_queries, items, query_list, problem_type, gamma, objval_opt = self.problem_1()
            elif problem_num == 2:
                answered_queries, items, query_list, problem_type, gamma, objval_opt = self.problem_2()
            elif problem_num == 3:
                answered_queries, items, query_list, problem_type, gamma, objval_opt = self.problem_3()
            elif problem_num == 4:
                answered_queries, items, query_list, problem_type, gamma, objval_opt = self.problem_4()
            elif problem_num == 5:
                answered_queries, items, query_list, problem_type, gamma, objval_opt = self.problem_5()
            else:
                raise Exception("invalid problem number")

            # run the elicitation method
            t0 = time.time()
            objval, query = query_func(answered_queries, items, problem_type, gamma)
            time_query_func = time.time() - t0

            # make sure that the recommendation objval matches that of the exhaustive search
            t0 = time.time()
            objval_exhaustive, query_exhaustive = next_optimal_query_iterative(answered_queries, items, query_list, problem_type, gamma)
            time_exhaustive = time.time() - t0

            # validate the exhaustive objval
            self.assertAlmostEqual(objval_exhaustive, objval_opt, delta=5e-3)

            # validate the new method
            self.assertAlmostEqual(objval, objval_opt, delta=5e-3)

        def problem_1(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 5
            num_items = 10
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 0
            problem_type = 'maximin'

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            agent = Agent.random(num_features,
                                 id=0,
                                 sphere_size=agent_sphere_size,
                                 seed=seed,
                                 )

            # simulate elicitation
            K = 3
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                if item_a.id < item_b.id:
                    q = Query(item_a, item_b)
                else:
                    q = Query(item_b, item_a)
                agent.answer_query(q)

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            objval = -2.2121378721893716

            return agent.answered_queries, items, query_list, problem_type, gamma, objval

        def problem_2(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 7
            num_items = 12
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 2.4
            problem_type = 'maximin'

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            agent = Agent.random(num_features,
                                 id=0,
                                 sphere_size=agent_sphere_size,
                                 seed=seed,
                                 )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                if item_a.id < item_b.id:
                    q = Query(item_a, item_b)
                else:
                    q = Query(item_b, item_a)
                agent.answer_query(q)

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            objval = -2.735950949506554

            return agent.answered_queries, items, query_list, problem_type, gamma, objval

        def problem_3(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 5
            num_items = 11
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 0
            problem_type = 'mmr'

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            agent = Agent.random(num_features,
                                 id=0,
                                 sphere_size=agent_sphere_size,
                                 seed=seed,
                                 )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                if item_a.id < item_b.id:
                    q = Query(item_a, item_b)
                else:
                    q = Query(item_b, item_a)
                agent.answer_query(q)

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            objval = 9.08093176759

            return agent.answered_queries, items, query_list, problem_type, gamma, objval

        def problem_4(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 4
            num_items = 12
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 3.0
            problem_type = 'mmr'

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            agent = Agent.random(num_features,
                                 id=0,
                                 sphere_size=agent_sphere_size,
                                 seed=seed,
                                 )

            # simulate elicitation
            K = 5
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                if item_a.id < item_b.id:
                    q = Query(item_a, item_b)
                else:
                    q = Query(item_b, item_a)
                agent.answer_query(q)

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            objval = 5.523104957742422

            return agent.answered_queries, items, query_list, problem_type, gamma, objval


        def problem_5(self):

            seed = 0
            rs = np.random.RandomState(seed)
            num_features = 7
            num_items = 20
            agent_sphere_size = 1
            item_sphere_size = 10
            gamma = 2.4
            problem_type = 'maximin'

            items = generate_items(num_features, num_items,
                                   item_sphere_size=item_sphere_size,
                                   seed=seed)

            agent = Agent.random(num_features,
                                 id=0,
                                 sphere_size=agent_sphere_size,
                                 seed=seed,
                                 )

            # simulate elicitation
            K = 9
            for i_q in range(K):
                item_a, item_b = rs.choice(items, 2, replace=False)
                if item_a.id < item_b.id:
                    q = Query(item_a, item_b)
                else:
                    q = Query(item_b, item_a)
                agent.answer_query(q)

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            objval = -1.0472563873475218

            return agent.answered_queries, items, query_list, problem_type, gamma, objval


class TestGetNextQueryMIP(BaseTestClass.TestGetNextQuery):
    """check that the MIP formulation without cuts solves the test problems"""

    def query_func(self, answered_queries, items, problem_type, gamma):
        objval, next_query = next_optimal_query_mip(answered_queries, items, problem_type, gamma)

        return objval, next_query

    def test_get_next_query_mip_1(self):
        """check get next query (MIP) 1"""
        self.evaluate_next_query(1, self.query_func)

    def test_get_next_query_mip_2(self):
        """check get next query (MIP) 2"""
        self.evaluate_next_query(2, self.query_func)

    def test_get_next_query_mip_3(self):
        """check get next query (MIP) 3"""
        self.evaluate_next_query(3, self.query_func)

    def test_get_next_query_mip_4(self):
        """check get next query (MIP) 4"""
        self.evaluate_next_query(4, self.query_func)

    def test_get_next_query_mip_5(self):
        """check get next query (MIP) 5"""
        self.evaluate_next_query(5, self.query_func)


if __name__ == '__main__':
    unittest.main()
