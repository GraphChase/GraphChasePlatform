import networkx as nx
import numpy as np
import random
from os.path import join
import os
from agent.nsg_nfsp.nsgnfsp_model import DRRN, AA_MA
from agent.nsg_nfsp.replay_buffer import ReplayBuffer, ReservoirBuffer
from agent.nsg_nfsp.nsgnfsp_defender_policy import AgentAADQN, AgentDRRN, AgentNFSP

class AttackerBandit(object):

    def __init__(self, exits, init_loc, adjlist, time_horizon, capacity=int(1e5), mode='greedy', args=None):
        self.capacity = capacity
        self.actions = []
        self.values = []
        self._next_entry_index = 0
        self.exits = exits
        self.init_loc = init_loc
        self.time_horizon = time_horizon
        self.graph = nx.from_dict_of_lists(adjlist)
        self.selected_exit = None
        assert mode in ['greedy', 'ucb']
        self.mode = mode
        self.T = 0
        self.paths = {}
        time_horizon = self.time_horizon

        for e in self.exits:
            self.paths[e] = list(nx.all_simple_paths(
                self.graph, source=self.init_loc[0], target=e, cutoff=time_horizon))
        
        self.paths = {k: v for k, v in self.paths.items() if v}  # v为空列表时会被过滤掉
        self.exits = [int(key) for key in self.paths.keys()]
        self.num_actions = len(self.exits)
        self.N_a = dict.fromkeys(list(range(self.num_actions)), 1)
        self.estimates = dict.fromkeys(list(range(self.num_actions)), 0)

    def select_action(self, observation=None, legal_actions=None, is_evaluation=True, epsilon=0.1):
        action = self.path[self.t]
        self.t += 1
        return (action,)

    def set_exit(self):
        if self.mode == 'ucb':
            steps = len(self.actions)
            i = max(range(self.num_actions), key=lambda x: self.estimates[x] + np.sqrt(
                2 * np.log(steps) / (1 + self.N_a[x])))
        else:
            i = max(range(self.num_actions), key=lambda x: self.estimates[x])
        self.selected_exit = self.exits[i]
        self.set_path()
        return i

    def update(self, action, value):
        if len(self.actions) < self.capacity:
            self.actions.append(action)
            self.values.append(value)
            self.estimates[action] += 1. / \
                (self.N_a[action] + 1) * (value - self.estimates[action])
            self.N_a[action] += 1
        else:
            assert len(self.actions) == self.capacity
            self.estimates[action] += 1. / \
                (self.N_a[action] + 1) * (value - self.estimates[action])

            del_action = self.actions[self._next_entry_index]
            del_value = self.values[self._next_entry_index]
            # if self.N_a[del_action]<=1:
            #     self.N_a[del_action]==2
            self.estimates[del_action] -= 1. / \
                (self.N_a[del_action] - 1) * \
                (del_value - self.estimates[del_action])
            self.N_a[del_action] -= 1
            self.actions[self._next_entry_index] = action
            self.values[self._next_entry_index] = value
            self.N_a[action] += 1

            self._next_entry_index += 1
            self._next_entry_index %= self.capacity

    def set_path(self):
        self.t = 1
        #paths=list(nx.all_simple_paths(self.graph, source=self.init_loc[0], target=self.selected_exit, cutoff=self.time_horizon))
        self.path = random.choice(self.paths[self.selected_exit])


class NFSPAttackerBandit(object):

    def __init__(self, BrAgent, br_prob=0.1):
        self.br_prob = br_prob
        self.is_br = False
        self.is_expl = False
        self.BrAgent = BrAgent
        self.num_actions = self.BrAgent.num_actions

        self.N_a = np.ones(self.num_actions)
        self.N_a_cache = np.zeros(self.num_actions)

    def sample_mode(self, exlp_prob=0.):
        if np.random.rand() < self.br_prob:
            self.is_br = True
            self.is_expl = False
            action = self.BrAgent.set_exit()
            self.N_a_cache[action] += 1
        else:
            self.is_br = False
            if np.random.rand() < exlp_prob:
                self.is_expl = True
                action = np.random.choice(self.num_actions, 1).item()
            else:
                self.is_expl = False
                prob = self.N_a / self.N_a.sum()
                action = np.random.choice(self.num_actions, 1, p=prob).item()
            self.BrAgent.selected_exit = self.BrAgent.exits[action]
            self.BrAgent.set_path()
        return action

    def select_action(self, observation, legal_actions, is_evaluation=False):
        assert len(observation) == 1
        assert len(legal_actions) == 1
        return self.BrAgent.select_action(observation, legal_actions, is_evaluation)

    def update_N_a(self):
        self.N_a += self.N_a_cache
        self.N_a_cache = np.zeros(self.num_actions)

    def save_model(self, save_folder, prefix=None):
        prop = self.N_a / self.N_a.sum()
        prop = np.around(prop, decimals=4)

        avg_net_name = 'avg_prop.txt'
        br_net_name = 'br_estimates.txt'

        os.makedirs(save_folder, exist_ok=True)
        with open(join(save_folder, avg_net_name), 'a') as f:
            log = 'Episode : {}, Average Probability : {} \n'.format(
                prefix, prop)
            f.write(log)

        with open(join(save_folder, br_net_name), 'a') as f:
            log = 'Episode : {}, Return Estimate : {} \n'.format(
                prefix, self.BrAgent.estimates)
            f.write(log)

    def set_mode(self, mode):
        assert mode in ['avg', 'br']
        if mode == 'avg':
            self.is_br = False
        else:
            self.is_br = True
        self.is_expl = False

    def reset(self):
        if self.is_br:
            self.BrAgent.set_exit()
        else:
            prob = self.N_a / self.N_a.sum()
            action = np.random.choice(self.num_actions, 1, p=prob).item()
            self.BrAgent.selected_exit = self.BrAgent.exits[action]
            self.BrAgent.set_path()

def CreateAttacker(Map, args):
    if args.attacker_mode == 'bandit':
        AttackerBr = AttackerBandit(
            Map.exits, Map.attacker_init, Map.adjlist, Map.time_horizon, capacity=int(1e4),args=args)
        Attacker = NFSPAttackerBandit(AttackerBr, br_prob=args.br_prob)
    else:
        br_buffer = ReplayBuffer(args.br_buffer_capacity)
        avg_buffer = ReservoirBuffer(args.avg_buffer_capacity)
        if args.attacker_mode == 'drrn':
            attacker_br_net = DRRN(Map.num_nodes, Map.time_horizon, args.embedding_size, args.hidden_size,
                                   args.relevant_v_size, naive=args.if_naivedrrn, num_defender=None, out_mode='rl')

            AttackerBr = AgentDRRN(attacker_br_net, br_buffer, epsilon_start=0.05,
                                   epsilon_end=0.001,
                                   epsilon_decay_duration=args.max_episodes*Map.time_horizon*args.br_prob,
                                   lr=args.br_lr, s_q_expl=False, opt_scheduler=False, player_idx=1, Map=Map.adjlist)
            avg_net = DRRN(Map.num_nodes, Map.time_horizon, args.embedding_size, args.hidden_size,
                           args.relevant_v_size,  naive=args.if_naivedrrn, num_defender=None, out_mode='sl')
        elif args.attacker_mode == 'aa':
            attacker_br_net = AA_MA(Map.num_nodes+1, Map.num_nodes, Map.time_horizon,
                                    args.embedding_size, args.hidden_size, args.relevant_v_size, num_defender=None, seq_mode=args.seq_mode, Map=Map)
            AttackerBr = AgentAADQN(attacker_br_net, br_buffer, epsilon_start=0.05,
                                    epsilon_end=0.001,
                                    epsilon_decay_duration=args.max_episodes*Map.time_horizon*args.br_prob,
                                    lr=args.br_lr, opt_scheduler=False, player_idx=1, Map=Map.adjlist)

            avg_net = AA_MA(Map.num_nodes+1, Map.num_nodes, Map.time_horizon,
                            args.embedding_size, args.hidden_size, args.relevant_v_size, num_defender=None, seq_mode=args.seq_mode, Map=Map)
        else:
            ValueError
        Attacker = AgentNFSP(AttackerBr, avg_net, avg_buffer,
                             br_prob=args.br_prob, avg_lr=args.avg_lr, sl_mode=args.attacker_mode)
    return Attacker