import networkx as nx
import numpy as np
import random
import torch
import time

class PathEvaderRunner(object):
    """
    According evader all possible paths select single path then change trajectory to actions sequence.
    """
    def __init__(self, env, args) -> None:
        f"""
        args:
            - attacker_type: "all_path" select single path from all paths; "exit_node", firstly choose the exit_node, then choose single path from possible paths to the exit  
            - strategy_type: "mix" choose path according each path prob; "greedy" select the max q path
        """       
        self.args = args

        self.env = env
        self.graph = env.nx_graph
        self.colums = env._colums if hasattr(env, '_colums') else None
        self.evader_position = env._initial_location[0] # list
        self.exit_nodes = env._initial_location[2] # list
        self.max_timesteps = env.time_horizon

        self.attacker_type = self.args.attacker_type
        self.strategy_type = self.args.strategy_type

        if self.attacker_type == 'all_path':
            self.path_selections, _ = self.shortest_path()
        elif self.attacker_type == 'exit_node':
            _, self.path_selections = self.shortest_path()
        else:
            raise Exception('Wrong attacker_type')
        
        self.q_table = np.random.uniform(-1, 1, (len(self.path_selections,)))
        # self.q_table = np.array([random.uniform(-1, 1) for _ in range(len(self.path_selections))])

    def get_action(self, strategy=None) -> list:
        '''
        Return:
            path's action, full single path
        '''
        if strategy is None:
            select_strategy = self._get_strategy()
        else:
            select_strategy = strategy

        mask = np.array([0 if len(i) == 0 else 1 for i in self.path_selections.values()]) if self.attacker_type == 'exit_node' else np.array([1] * len(self.path_selections))
        masked_strategy = select_strategy * mask

        total_probability = np.sum(masked_strategy)
        if total_probability == 0:
            raise ValueError("All probabilities are zero. Check your mask or strategy.")
    
        select_strategy = masked_strategy / total_probability

        if self.strategy_type == 'mix':
            path_idx = np.random.choice(np.arange(select_strategy.size), p=select_strategy)
        elif self.strategy_type == 'greedy':
            max_value = np.max(self.q_table)
            max_indices = np.where(self.q_table == max_value)[0] 
            path_idx = random.choice(max_indices)

        if self.attacker_type == 'all_path':
            single_path = self.path_selections[path_idx]
            
        elif self.attacker_type == 'exit_node':        
            exit_node = self.exit_nodes[path_idx]
            single_path = random.choice(self.path_selections[exit_node])
            
            # available_action = [i for i in range(len(self.path_selections[exit_node]))]
            # idx = np.random.choice(available_action)
            # single_path = self.path_selections[exit_node][idx]

        return self._trajectory2actions(single_path), single_path

    def _get_strategy(self,) -> np.ndarray:
        if self.strategy_type == 'mix':
            q_value = np.exp(self.q_table)
            sum_value = np.sum(q_value)
            strategy = q_value / sum_value
        elif self.strategy_type == 'greedy':
            strategy = np.zeros((len(self.path_selections, )))
            strategy[np.argmax(self.q_table)] = 1.
        return strategy

    def shortest_path(self):
        all_paths = []
        exit_paths = {}
        for j in range(len(self.exit_nodes)):
            path_temp = []
            if nx.has_path(self.graph, source=self.evader_position[0], target=self.exit_nodes[j]):

                shortest_path = nx.all_shortest_paths(self.graph, source=self.evader_position[0], target=self.exit_nodes[j])

                for p in shortest_path:
                    if len(p) > self.max_timesteps + 1 or len(path_temp) >= 200:
                        break
                    else:
                        if list(set(p) & set(self.exit_nodes)) == [self.exit_nodes[j]]:
                            path_temp.append(p)

            exit_paths[self.exit_nodes[j]] = path_temp
            all_paths += path_temp

        return all_paths, exit_paths
    
    def _trajectory2actions(self, path: list) -> list:

        if self.colums:
            # Used for gird graph
            directions_to_actions = {0: 0, -self.colums:1, self.colums:2, -1:3, 1:4}

            acts = []
            for idx in range(1, len(path), 1):
                cur_act = path[idx] - path[idx-1]
                acts.append(directions_to_actions[cur_act])

            return acts
        else:
            return None
    
    def train(self, defender_policy_list, meta_probability, sample_number) -> list:
        """ 
        Evaluate every path by simulation sample_number times and update self.q_table
        Return: defender's utility for each evader's selection (if evader run out, defender utility is 0, else 1)
        """
        defender_rds_vs_each_path = []
        for i in range(len(self.path_selections)):
            path_reward = 0
            defender_reward = 0
            for _ in range(sample_number):
                defender_idx = np.random.choice(range(len(defender_policy_list)), p=meta_probability)
                defender_policy = defender_policy_list[defender_idx]

                if self.attacker_type == 'all_path':
                    evader_path = self.path_selections[i]
                elif self.attacker_type == 'exit_node':
                    exit_node = self.exit_nodes[i]
                    if not self.path_selections[exit_node]: # empty list
                        break
                    else:
                        evader_path = random.choice(self.path_selections[exit_node])
                    # available_action = [i for i in range(len(self.path_selections[exit_node]))]
                    # idx = np.random.choice(available_action)
                    # evader_path = self.path_selections[exit_node][idx]
                
                evader_actions = evader_path[1:] if self.env.nextstate_as_action else self._trajectory2actions(evader_path)

                # rollout
                terminated = False
                observation, info = self.env.reset()
                t = 0

                while not terminated:
                    evader_act = evader_actions[t]
                    with torch.no_grad():
                        defender_action = defender_policy.get_env_actions(observation, t)
                    actions = np.array(defender_action) # (n,)

                    if self.env.nextstate_as_action:
                        for i in range(1, len(observation)):
                            actions[i-1] = self.env.graph.change_state[observation[i] - 1][defender_action[i-1]]

                    actions = np.insert(actions, 0, evader_act)
                    observation, reward, terminated, truncated, info = self.env.step(actions)
                    t += 1

                    if terminated or truncated:
                        path_reward += reward[0]
                        if reward[0] == -1:
                            defender_reward += 1                           

            defender_rds_vs_each_path.append(defender_reward/sample_number)
            path_reward /= sample_number
            self.q_table[i] = path_reward
        
        return defender_rds_vs_each_path

    def random_generate_evader_oracle(self, oracle_size) -> list:
        """
        Randomly generate evader strategy oracle, each strategy means the probabilty of choosing each path/exit_node
        """
        policy_list = []
        for _ in range(oracle_size):
            s = np.random.random(size= len(self.path_selections))
            s /= s.sum()
            policy_list.append(s)

        return policy_list
    
    def initialize(self,):
        self.q_table = np.random.uniform(-1, 1, (len(self.path_selections,)))
        return self