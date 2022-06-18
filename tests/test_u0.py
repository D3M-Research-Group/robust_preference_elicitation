import itertools
import logging
import time
import unittest

import numpy as np

from adaptive_elicitation import next_optimal_query_mip
from preference_classes import generate_items, Agent, Query
from recommendation import solve_recommendation_problem
from static_elicitation import static_mip_optimal
from utils import get_u0

FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)
logger = logging.getLogger()


PRINT_LOGS = True

class BaseTestClass:
    class TestUSet(unittest.TestCase):
        """contains an evaluation function and test problems to test different recommendation methods"""

        def run_test(self, problem_num, u0_type):
            """evaluate a test problem"""
            if problem_num == 1:
                agent, items, query_list, problem_type, gamma = self.problem_1()
            elif problem_num == 2:
                agent, items, query_list, problem_type, gamma = self.problem_2()
            elif problem_num == 3:
                agent, items, query_list, problem_type, gamma = self.problem_3()
            elif problem_num == 4:
                agent, items, query_list, problem_type, gamma = self.problem_4()
            else:
                raise Exception("invalid problem number")

            k_max = 4
            logger.info(f"num items = {len(items)}")
            logger.info(f"num features = {len(items[0].features)}")
            logger.info(f"problem type: {problem_type}")
            logger.info(f"u0 type: {u0_type}")

            # simulate elicitation
            for k in range(k_max):
                _, query = next_optimal_query_mip(agent.answered_queries, items, problem_type, gamma, u0_type=u0_type)
                agent.answer_query(query)

                # run the recommendation method
                objval, recommended_item = solve_recommendation_problem(
                    agent.answered_queries, items, problem_type, gamma=gamma, verbose=False, u0_type=u0_type,
                )
                logger.info(f"rec. objective: {objval}. true objval = {agent.true_utility(recommended_item)}, "
                            f"item id = {recommended_item.id}")

        def problem_1(self):

            seed = 0
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

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            return agent, items, query_list, problem_type, gamma


        def problem_2(self):

            seed = 0
            num_features = 4
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

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            return agent, items, query_list, problem_type, gamma

        def problem_3(self):

            seed = 0
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

            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            return agent, items, query_list, problem_type, gamma

        def problem_4(self):

            seed = 0
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


            query_list = [Query(item_a, item_b) for item_a, item_b in itertools.combinations(items, 2)]

            return agent, items, query_list, problem_type, gamma


class TestBoxU0(BaseTestClass.TestUSet):
    """compare adaptive approaches with [-1, 1] box U^0"""

    def test_box_1(self):
        """box U^0 with problem 1"""
        self.run_test(1, "box")

    def test_box_2(self):
        """box U^0 with problem 2"""
        self.run_test(2, "box")

    def test_box_3(self):
        """box U^0 with problem 3"""
        self.run_test(3, "box")

    def test_box_4(self):
        """box U^0 with problem 4"""
        self.run_test(4, "box")


class TestPosNormedU0(BaseTestClass.TestUSet):
    """compare adaptive approaches with U^0 positive with 1-norm 1"""

    def test_pos_normed_1(self):
        """pos-normed U^0 with problem 1"""
        self.run_test(1, "positive_normed")

    def test_pos_normed_2(self):
        """pos-normed U^0 with problem 2"""
        self.run_test(2, "positive_normed")

    def test_pos_normed_3(self):
        """pos-normed U^0 with problem 3"""
        self.run_test(3, "positive_normed")

    def test_pos_normed_4(self):
        """pos-normed U^0 with problem 4"""
        self.run_test(4, "positive_normed")


if __name__ == '__main__':
    unittest.main()
