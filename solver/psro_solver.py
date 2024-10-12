import numpy as np
import torch
import time
import os
from solver.prd_solver import projected_replicator_dynamics
import copy


class PSRO(object):
    def __init__(self, 
                 env, 
                 args, 
                 evader_runner,
                 pursuer_runner,):
        
        self._num_players = 2
        self.env = env
        self.args = args

        self.evder_runner = copy.deepcopy(evader_runner)
        self.pursuer_runner = copy.deepcopy(pursuer_runner)
        self.pursuer_runners_list = []
        self.evader_runners_list = []

        self.train_pursuer_cnt = 0

        self.meta_solver = projected_replicator_dynamics
        self.save_path = os.path.join(args.save_path, "defender_psro_model")

    def _evaluate(self, evader, defender):
        total_rewards = np.zeros((self._num_players,))

        for _ in range(self.args.eval_episodes):

            evader_actions, _ = evader.get_action()

            terminated = False
            observation, info = self.env.reset()
            t = 0

            while not terminated:
                evader_act = evader_actions[t]
                with torch.no_grad():
                    env_actions = defender.get_env_actions(observation, t)
                    actions = np.array(env_actions)

                actions = np.insert(actions, 0, evader_act)
                observation, reward, terminated, truncated, info = self.env.step(actions)
                t += 1

                if terminated or truncated:
                    total_rewards += np.array(reward)
        
        return total_rewards / self.args.eval_episodes
         

    def _get_runner(self, player_id):

        if player_id == 0:
            return copy.deepcopy(self.evder_runner)
        if player_id == 1:
            return copy.deepcopy(self.pursuer_runner)

    def solve(self, ):
        self.init()
        self.pursuer_runners_list[-1].save(self.save_path, len(self.pursuer_runners_list))

        time_list = []
        worst_case_utility_list = []
        average_worst_case_utility_list = []
        start_time = time.time()
        for i in range(self.args.num_psro_iteration):
            print("add new runners...")
            worst_case_utility, average_worst_case_utility = self.add_new_runner()
            self.pursuer_runners_list[-1].save(self.save_path, len(self.pursuer_runners_list))
            worst_case_utility_list.append(worst_case_utility)
            average_worst_case_utility_list.append(average_worst_case_utility)
            print("worst case utility for pursuer:", worst_case_utility_list, average_worst_case_utility_list)

            print("update meta game...")
            self.update_meta_game()
            print(self.meta_games)

            print("compute meta distribution...")
            self.compute_meta_distribution()
            print(self.meta_strategies)
            time_list.append(time.time() - start_time)
            print(f"Iteration {i+1} done, total cost time is {time_list[-1]}")
            
        print(f"PSRO Done, run time: {time_list}")

    def add_new_runner(self):
        evader_runner = self._get_runner(0)
        pursuer_runner = self._get_runner(1)

        print("computing best response for evader...")
        start_time = time.time()
        defender_rewards = evader_runner.train(self.pursuer_runners_list, self.meta_strategies[1], self.args.train_evader_number)
        best_0 = np.max(evader_runner.q_table)

        average_best_0 = round(np.mean(evader_runner.q_table),4)
        print("run time for computing best response for evader:", round(time.time()-start_time,4))
        print("best response value for evader", best_0, average_best_0)
        
        defender_worst_utility = np.min(np.array(defender_rewards))
        defender_average_utility = round(np.mean(np.array(defender_rewards)), 4)

        print("computing best response for pursuer...")
        start_time = time.time()
        train_infos = pursuer_runner.train(self.evader_runners_list, self.meta_strategies[0])
        print("run time for computing best response for pursuer:", time.time()-start_time)

        self.evader_runners_list.append(evader_runner)
        self.pursuer_runners_list.append(pursuer_runner)

        return defender_worst_utility, defender_average_utility     

    def cal_initial_evader_BR(self):
        self.init()
        evader_runner = self._get_runner(0)
        start_time = time.time()
        defender_rewards = evader_runner.train(self.pursuer_runners_list, self.meta_strategies[1], self.args.train_evader_number)
        best_0 = np.max(evader_runner.q_table)
        print("evader BR computation, run time: {:.6f}, BR value: {:.5f}".format(time.time() - start_time, best_0))      

        return best_0

    def add_pursuer_runner(self):
        start_time = time.time()
        pursuer_agent = self.get_runner(1)
        pursuer_agent.train(self.evader_runners_list, self.meta_strategies[0], self.train_pursuer_episode_number, train_num_per_ite=10)
        print("pursuer BR computation, run time: {:.6f}".format(time.time() - start_time))
        return pursuer_agent     

    def update_meta_game(self):
        r = len(self.evader_runners_list)
        c = len(self.pursuer_runners_list)
        meta_games = [np.full([r, c], fill_value=np.nan),
                      np.full([r, c], fill_value=np.nan)]

        (o_r, o_c) = self.meta_games[0].shape
        for i in [0, 1]:
            for t_r in range(o_r):
                for t_c in range(o_c):
                    meta_games[i][t_r][t_c] = self.meta_games[i][t_r][t_c]

        for t_r in range(r):
            for t_c in range(c):
                if np.isnan(meta_games[0][t_r][t_c]):
                    evaluate_reward = self._evaluate(self.evader_runners_list[t_r], self.pursuer_runners_list[t_c])
                    meta_games[0][t_r][t_c] = evaluate_reward[0]
                    meta_games[1][t_r][t_c] = evaluate_reward[1]

        self.meta_games = meta_games

    def compute_meta_distribution(self):
        self.meta_strategies = self.meta_solver(self.meta_games)            

    def init(self):
        """
        Before psro iteration, generate evader and defender policy, and get self.meta_games and self.meta_strategies
        """
        print("Initializing...")
        evader_runner = self._get_runner(0)
        pursuer_runner = self._get_runner(1)
        evaluate_reward = self._evaluate(evader_runner, pursuer_runner)

        self.pursuer_runners_list.append(pursuer_runner)
        self.evader_runners_list.append(evader_runner)

        r = len(self.evader_runners_list)
        c = len(self.pursuer_runners_list)
        self.meta_games = [np.full([r, c], fill_value=evaluate_reward[0]),
                           np.full([r, c], fill_value=evaluate_reward[1])]
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]