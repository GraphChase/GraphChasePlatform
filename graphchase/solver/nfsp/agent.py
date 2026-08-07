import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .model import *
import networkx as nx
import copy
import numpy as np
from os.path import join
import random
import time
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')


class BaseAgent(object):

    def __init__(self, args):
        self.args = args

    def select_action(self, observation, legal_actions, is_evaluation=False):
        NotImplementedError

    def reset(self):
        pass


def query_legal_actions(player_idx, obs, Map, num_defender=None):
    attacker_history, defender_position = obs
    if player_idx == 0:
        if num_defender == 1:
            defender_legal_actions = Map[defender_position]
        else:
            assert num_defender > 1
            before_combination = []
            for i in range(len(defender_position)):
                before_combination.append(Map[defender_position[i]])
            defender_legal_actions = list(product(*before_combination))
        return defender_legal_actions
    else:
        assert num_defender == None
        attacker_legal_actions = Map[attacker_history[-1]]
        return attacker_legal_actions


class AgentDRRN(object):

    def __init__(self,
                 policy_net,
                 buffer,
                 s_q_expl=False,
                 epsilon_start=0.95,
                 epsilon_end=0.05,
                 epsilon_decay_duration=int(1e6),
                 update_target_every=1000,
                 lr=0.0005,
                 opt_scheduler=True,
                 player_idx=None,
                 Map=None):

        self.gamma = 1.0
        self.grad_clip = [-1, 1]
        self._step_counter = 0
        self._learn_step = 0
        self.opt_scheduler = opt_scheduler
        self.s_q_expl = s_q_expl
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration
        self.update_target_every = update_target_every
        self.lr = lr

        self.policy_net = policy_net.to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.buffer = buffer
        # self.optimizer = optim.SGD(
        # self.policy_net.parameters(), lr=self.lr, momentum=0.99,
        # nesterov=True)
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=self.lr)
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.opt_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, [100000,200000], gamma=0.5)
        self.opt_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, int(1e4), gamma=0.95)
        if hasattr(self.policy_net, 'num_defender'):
            self.num_defender = self.policy_net.num_defender
        self.player_idx = player_idx
        self.map = Map

    def select_action(self, observation, legal_actions, is_evaluation=False):
        # observation should be in the form :
        #    defender:  [([1, 2, 3], 2)]
        #    attaker:  [[1, 2, 3]]
        # legal_actions should be in the form : [[1, 2, 3, 4]] / [[3]]
        assert len(observation) == 1
        assert len(legal_actions) == 1
        with torch.no_grad():
            if not self.s_q_expl:
                epsilon = self._get_epsilon(is_evaluation, power=1.0)
                if np.random.rand() < epsilon:
                    action = random.choice(legal_actions[0])  # int / tuple
                    idx = legal_actions[0].index(action)
                else:
                    action, _, _ = self.policy_net(observation, legal_actions)
                    if self.num_defender:
                        if self.num_defender > 1:
                            action = tuple([loc.item() for loc in action[0]])
                        else:
                            action = action.item()
                    else:
                        action = action.item()
                    idx = legal_actions[0].index(action)
            else:
                if is_evaluation:
                    action, _, _ = self.policy_net(observation, legal_actions)
                    action = action.item()
                else:
                    _, _, q_val = self.policy_net(observation, legal_actions)
                    action_prob = F.softmax(q_val, dim=-1)
                    action_idx = torch.multinomial(
                        action_prob, num_samples=1).item()
                    action = legal_actions[0][action_idx]
            if not is_evaluation:
                self._step_counter += 1
            return action, idx

    def _get_epsilon(self, is_evaluation, power=2.0):
        if is_evaluation:
            return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
            self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
            (1 - decay_steps / self._epsilon_decay_duration)**power)
        #decayed_epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(-1.0 * self._step_counter/ self._epsilon_decay_duration)
        return decayed_epsilon

    #############################################
    ########## Learning step for DRRN ###########
    #############################################

    def learning_step(self, transitions):
        obs = [t.obs for t in transitions]
        action = [[t.action[0]] for t in transitions]
        reward = [t.reward for t in transitions]
        next_obs = [t.next_obs for t in transitions]
        #next_legal_action = [t.next_legal_action for t in transitions]
        next_legal_action = [query_legal_actions(
            self.player_idx, t.next_obs, self.map, self.num_defender) for t in transitions]
        is_end = [t.is_end for t in transitions]
        _, s_a_values, _ = self.policy_net(obs, action)  # shape: (batch,)
        _, next_s_a_values, _ = self.target_net(next_obs, next_legal_action)

        target_values = torch.Tensor(
            reward).to(device) + (1 - torch.Tensor(is_end).to(device)) * self.gamma * next_s_a_values
        target_values = target_values.detach()
        target_values.requires_grad = False

        loss = F.smooth_l1_loss(s_a_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(*self.grad_clip)
        # print(param.grad.data)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        if self.opt_scheduler:
            self.opt_scheduler.step()
        self._learn_step += 1

        if self._learn_step % self.update_target_every == 0:
            self.update_target_net()
        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def reset(self):
        pass


class AgentMADQN(object):

    def __init__(self,
                 policy_net,
                 buffer,
                 epsilon_start=0.95,
                 epsilon_end=0.05,
                 epsilon_decay_duration=int(1e6),
                 update_target_every=1000,
                 lr=0.0001,
                 opt_scheduler=True,
                 player_idx=None,
                 Map=None):
        self.gamma = 1.0
        self.grad_clip = [-1, 1]
        self._step_counter = 0
        self._learn_step = 0
        self.opt_scheduler = opt_scheduler
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration
        self.update_target_every = update_target_every
        self.lr = lr

        self.policy_net = policy_net.to(device)
        assert isinstance(self.policy_net, Defender_MA_DQN) or \
            isinstance(self.policy_net, Attacker_MA_DQN) or\
            isinstance(self.policy_net, AA_MA)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.buffer = buffer
        # self.optimizer = optim.SGD(
        # self.policy_net.parameters(), lr=self.lr, momentum=0.99,
        # nesterov=True)
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=self.lr)
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.opt_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, [100000,200000], gamma=0.5)
        self.opt_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, int(1e4), gamma=0.95)

        if hasattr(self.policy_net, 'num_defender'):
            self.num_defender = self.policy_net.num_defender
        self.player_idx = player_idx
        self.map = Map

    def select_action(self, observation, legal_actions, is_evaluation=False):
        # observation should be in the form :
        #    defender:  [([1, 2, 3], 2)]
        #    attaker:  [[1, 2, 3]]
        # legal_actions should be in the form : [[1, 2, 3, 4]] / [[3]]
        assert len(observation) == 1
        assert len(legal_actions) == 1
        with torch.no_grad():
            epsilon = self._get_epsilon(is_evaluation, power=1.0)
            if np.random.rand() < epsilon:
                action = random.choice(legal_actions[0])  # int
                idx = legal_actions[0].index(action)
            else:
                q_val = self.policy_net(observation, legal_actions)
                idx = torch.argmax(q_val[:len(legal_actions[0])])
                action = legal_actions[0][idx]

            if not is_evaluation:
                self._step_counter += 1
            return action, idx

    def _get_epsilon(self, is_evaluation, power=2.0):
        if is_evaluation:
            return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
            self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
            (1 - decay_steps / self._epsilon_decay_duration)**power)
        #decayed_epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(-1.0 * self._step_counter/ self._epsilon_decay_duration)
        return decayed_epsilon

    ##############################################
    ########## Learning step for MADQN ###########
    ##############################################

    def learning_step(self, transitions):
        obs = [t.obs for t in transitions]
        action_idx = [[t.action[1]] for t in transitions]
        reward = [t.reward for t in transitions]
        next_obs = [t.next_obs for t in transitions]
        #next_num_actions = [len(t.next_legal_action) for t in transitions]
        next_num_actions = [len(query_legal_actions(
            self.player_idx, t.next_obs, self.map, self.num_defender)) for t in transitions]
        is_end = [t.is_end for t in transitions]

        q_vals = self.policy_net(obs)  # shape: (batch, max_actions)
        idx = [[k] for k in range(q_vals.size(0))]
        a_values = q_vals[idx, action_idx].flatten()  # shape(batch,)
        next_q_vals = self.target_net(next_obs)
        next_max_q_values = []
        for i in range(len(next_num_actions)):
            next_max_q_values.append(
                torch.max(next_q_vals[i][:next_num_actions[i]]))
        next_max_q_values = torch.stack(next_max_q_values)
        target_values = torch.Tensor(
            reward).to(device) + (1 - torch.Tensor(is_end).to(device)) * self.gamma * next_max_q_values
        target_values = target_values.detach()
        target_values.requires_grad = False

        loss = F.smooth_l1_loss(a_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(*self.grad_clip)
        # print(param.grad.data)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        if self.opt_scheduler:
            self.opt_scheduler.step()
        self._learn_step += 1

        if self._learn_step % self.update_target_every == 0:
            self.update_target_net()
        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def reset(self):
        pass


class AgentAADQN(object):

    def __init__(self,
                 policy_net,
                 buffer,
                 epsilon_start=0.95,
                 epsilon_end=0.05,
                 epsilon_decay_duration=int(1e6),
                 update_target_every=1000,
                 lr=0.0001,
                 opt_scheduler=True,
                 player_idx=None,
                 Map=None):
        self.gamma = 1.0
        self.grad_clip = [-1, 1]
        self._step_counter = 0
        self._learn_step = 0
        self.opt_scheduler = opt_scheduler
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration
        self.update_target_every = update_target_every
        self.lr = lr

        self.policy_net = policy_net.to(device)
        assert isinstance(self.policy_net, Defender_MA_DQN) or \
            isinstance(self.policy_net, Attacker_MA_DQN) or \
            isinstance(self.policy_net, AA_MA)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.buffer = buffer
        # self.optimizer = optim.SGD(
        # self.policy_net.parameters(), lr=self.lr, momentum=0.99,
        # nesterov=True)
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=self.lr)
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # self.opt_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, [100000,200000], gamma=0.5)
        self.opt_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, int(1e4), gamma=0.95)

        if hasattr(self.policy_net, 'num_defender'):
            self.num_defender = self.policy_net.num_defender
        self.player_idx = player_idx
        self.map = Map

    def select_action(self, observation, legal_actions, is_evaluation=False):
        # observation should be in the form :
        #    defender:  [([1, 2, 3], 2)]
        #    attaker:  [[1, 2, 3]]
        # legal_actions should be in the form : [[1, 2, 3, 4]] / [[3]]
        assert len(observation) == 1
        assert len(legal_actions) == 1
        with torch.no_grad():
            epsilon = self._get_epsilon(is_evaluation, power=1.0)
            if np.random.rand() < epsilon:
                action = random.choice(legal_actions[0])  # int
                idx = legal_actions[0].index(action)
            else:
                q_val = self.policy_net(observation, legal_actions)
                legal_actions_idx = legal_actions[0]
                if self.num_defender:
                    if self.num_defender > 1:
                        legal_actions_idx = [self._mutil_loc_to_idx(
                            loc) for loc in legal_actions[0]]
                legal_q_val = q_val[legal_actions_idx]
                idx = torch.argmax(legal_q_val)
                action = legal_actions[0][idx]

            if not is_evaluation:
                self._step_counter += 1
            return action, idx

    def _get_epsilon(self, is_evaluation, power=2.0):
        if is_evaluation:
            return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
            self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
            (1 - decay_steps / self._epsilon_decay_duration)**power)
        #decayed_epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(-1.0 * self._step_counter/ self._epsilon_decay_duration)
        return decayed_epsilon

    ##############################################
    ########## Learning step for AADQN ###########
    ##############################################

    def learning_step(self, transitions):
        obs = [t.obs for t in transitions]
        action = [[t.action[0]] for t in transitions]
        reward = [t.reward for t in transitions]
        next_obs = [t.next_obs for t in transitions]
        #next_legal_actions = [t.next_legal_action for t in transitions]
        next_legal_actions = [query_legal_actions(
            self.player_idx, t.next_obs, self.map, self.num_defender) for t in transitions]
        is_end = [t.is_end for t in transitions]

        q_vals = self.policy_net(obs)  # shape: (batch, max_actions)
        action_idx = action
        if self.num_defender:
            if self.num_defender > 1:
                action_idx = [
                    [self._mutil_loc_to_idx(loc[0])] for loc in action]
        action_idx = torch.tensor(action_idx, dtype=torch.long, device=device)
        a_values = torch.gather(
            q_vals, 1, action_idx).flatten()  # shape(batch,)
        next_q_vals = self.target_net(next_obs)
        next_max_q_values = []
        for i in range(len(next_legal_actions)):
            next_legal_action_idx = next_legal_actions[i]
            if self.num_defender:
                if self.num_defender > 1:
                    next_legal_action_idx = [self._mutil_loc_to_idx(
                        loc) for loc in next_legal_actions[i]]
            next_max_q_values.append(
                torch.max(next_q_vals[i][next_legal_action_idx]))
        next_max_q_values = torch.stack(next_max_q_values)
        target_values = torch.Tensor(
            reward).to(device) + (1 - torch.Tensor(is_end).to(device)) * self.gamma * next_max_q_values
        target_values = target_values.detach()
        target_values.requires_grad = False

        loss = F.smooth_l1_loss(a_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(*self.grad_clip)
        # print(param.grad.data)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        if self.opt_scheduler:
            self.opt_scheduler.step()
        self._learn_step += 1

        if self._learn_step % self.update_target_every == 0:
            self.update_target_net()
        return loss

    def _mutil_loc_to_idx(self, loc):
        # loc: (loc0, loc1,...)
        idx = 0
        for b in range(self.num_defender):
            assert loc[b] <= self.policy_net.num_nodes
            idx += loc[b] * pow(self.policy_net.num_nodes + 1, b)
        return idx

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def reset(self):
        pass


class AgentEval(object):

    def __init__(self, avg_net, sl_mode, num_defender):
        self.avg_net = avg_net.to(device)
        self.sl_mode = sl_mode
        self.num_defender = num_defender

    def select_action(self, observation, legal_actions, is_evaluation=True):
        assert len(observation) == 1
        assert len(legal_actions) == 1
        with torch.no_grad():
            prob = self.action_probs(
                observation, legal_actions, numpy=False)
            action_idx = torch.multinomial(
                prob, num_samples=1).item()
            action = legal_actions[0][action_idx]
            return action, action_idx

    def action_probs(self, observation, legal_actions, numpy=False, temp=1):
        with torch.no_grad():
            if self.sl_mode == 'aa':
                prob = F.softmax(self.avg_net(observation), dim=-1)
                legal_actions_idx = legal_actions[0]
                if self.num_defender:
                    if self.num_defender > 1:
                        legal_actions_idx = [self._mutil_loc_to_idx(
                            loc) for loc in legal_actions[0]]
                prob = prob[legal_actions_idx]
                prob /= prob.sum()
            elif self.sl_mode == 'drrn':
                prob = F.softmax(self.avg_net(
                    observation, legal_actions) / temp, dim=-1)
            elif self.sl_mode == 'ma':
                prob = F.softmax(self.avg_net(observation), dim=-1)[
                    :len(legal_actions[0])]
                prob /= prob.sum()
            else:
                ValueError
            if numpy:
                return prob.cpu().numpy()
            else:
                return prob

    def _mutil_loc_to_idx(self, loc):
        # loc: (loc0, loc1,...)
        idx = 0
        for b in range(self.num_defender):
            assert loc[b] <= self.avg_net.num_nodes
            idx += loc[b] * pow(self.avg_net.num_nodes + 1, b)
        return idx

    def load_model(self, save_folder, prefix=None):
        avg_net_name = 'avg_net.pt'
        if prefix:
            avg_net_name = str(prefix) + 'avg_net.pt'
        self.avg_net.load_state_dict(torch.load(
            join(save_folder, avg_net_name), map_location=device))
        self.avg_net.eval()
        return

    def reset(self):
        pass


class AgentGreedyDefender(object):

    def __init__(self, Map):
        self.adjlist = Map.adjlist
        self.exits = Map.exits
        self.graph = nx.from_dict_of_lists(self.adjlist)

    def select_action(self, observation, legal_actions=None, is_evaluation=True):
        attacker_history, current_positions = observation[0]
        actions = []
        for current_position in current_positions:
            shortest_path_to_attacker = nx.shortest_path(
                self.graph, current_position, attacker_history[-1])
            if len(shortest_path_to_attacker) >= 3:
                action = shortest_path_to_attacker[1]
            else:
                action = shortest_path_to_attacker[0]
                # for ex in self.exits:
                #     if ex in legal_actions[0]:
                #         action = ex
                #         break
            actions.append(action)
        actions = tuple(actions)
        return (actions,)

    def reset(self):
        pass


class AgentRandomDefender(object):

    def __init__(self, Map):
        self.player_idx = 0
        self.map = Map.adjlist
        self.num_defender = Map.num_defender

    def select_action(self, observation, legal_actions=None, is_evaluation=True):
        legal_actions = query_legal_actions(
            self.player_idx, observation[0], self.map, self.num_defender)
        action = random.choice(legal_actions)
        return (action,)

    def reset(self):
        pass


class AgentGreedyAttacker(object):

    def __init__(self, adjlist, exits):
        self.adjlist = adjlist
        self.exits = exits
        self.graph = nx.from_dict_of_lists(adjlist)

    def select_action(self, observation, legal_actions=None, is_evaluation=True, epsilon=0.1):
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions[0])
        else:
            current_position = observation[0][0][-1]
            all_shortest_path = list(nx.all_shortest_paths(
                self.graph, current_position, self.seleted_exit))
            path = random.choice(all_shortest_path)
            action = path[1]
            # action = nx.shortest_path(
            #     self.graph, current_position, self.seleted_exit)[1]
        return (action,)

    def reset(self):
        self.seleted_exit = np.random.choice(self.exits)


class BRAttacker(object):

    def __init__(self, path):
        self.path = path
        self.t = None

    def select_action(self, observation=None, legal_actions=None, is_evaluation=True):
        action = self.path[self.t]
        self.t += 1
        return (action,)

    def reset(self):
        self.t = 1


class AllPathAttacker(object):

    def __init__(self, exits, init_loc, adjlist, time_horizon):
        self.exits = exits
        self.init_loc = init_loc
        self.time_horizon = time_horizon
        self.graph = nx.from_dict_of_lists(adjlist)
        self.T = 0
        self.paths = []
        self.path = None
        for e in self.exits:
            self.paths += list(nx.all_simple_paths(
                self.graph, source=self.init_loc[0], target=e, cutoff=self.time_horizon))

    def select_action(self, observation=None, legal_actions=None, is_evaluation=True):
        action = self.path[self.t]
        self.t += 1
        return (action,)

    def reset(self):
        self.t = 1
        self.path = random.choice(self.paths)


class AttackerBandit(object):

    def __init__(self, exits, init_loc, adjlist, time_horizon, capacity=int(1e5), mode='greedy', args=None):
        self.capacity = capacity
        self.actions = []
        self.values = []
        self._next_entry_index = 0
        self.num_actions = len(exits)
        self.N_a = dict.fromkeys(list(range(self.num_actions)), 1)
        self.estimates = dict.fromkeys(list(range(self.num_actions)), 0)
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
            # self.paths[e] = list(
            #     nx.all_simple_paths(self.graph, source=self.init_loc[0], target=e, cutoff=time_horizon)
            # )
            shortest_paths = list(nx.all_shortest_paths(self.graph, source=self.init_loc[0], target=e))
            self.paths[e] = [path for path in shortest_paths if len(path) - 1 <= time_horizon]
        self.reachable_indices = [idx for idx, exit_node in enumerate(self.exits) if self.paths.get(exit_node)]
        self.reachable_exits = [self.exits[idx] for idx in self.reachable_indices]
        if not self.reachable_indices:
            raise ValueError("Attacker has no feasible paths to any exit.")

    def select_action(self, observation=None, legal_actions=None, is_evaluation=True, epsilon=0.1):
        action = self.path[self.t]
        self.t += 1
        return (action,)

    def set_exit(self):
        if not self.reachable_indices:
            raise ValueError("Attacker has no feasible paths to any exit.")
        if self.mode == 'ucb':
            steps = len(self.actions)
            i = max(self.reachable_indices, key=lambda x: self.estimates[x] + np.sqrt(
                2 * np.log(steps) / (1 + self.N_a[x])))
        else:
            i = max(self.reachable_indices, key=lambda x: self.estimates[x])
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
        reachable_indices = getattr(self.BrAgent, "reachable_indices", list(range(self.num_actions)))
        if not reachable_indices:
            raise ValueError("Attacker has no feasible paths to any exit.")
        if np.random.rand() < self.br_prob:
            self.is_br = True
            self.is_expl = False
            action = self.BrAgent.set_exit()
            self.N_a_cache[action] += 1
        else:
            self.is_br = False
            if np.random.rand() < exlp_prob:
                self.is_expl = True
                action = np.random.choice(reachable_indices, 1).item()
            else:
                self.is_expl = False
                prob = self.N_a / self.N_a.sum()
                reachable_prob = np.asarray([prob[idx] for idx in reachable_indices], dtype=float)
                prob_sum = reachable_prob.sum()
                if not np.isfinite(prob_sum) or prob_sum <= 0:
                    reachable_prob = np.ones(len(reachable_indices), dtype=float)
                reachable_prob = reachable_prob / reachable_prob.sum()
                action = np.random.choice(reachable_indices, 1, p=reachable_prob).item()
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
        # if prefix:
        #     avg_net_name = str(prefix) + 'avg_prop.txt'
        #     br_net_name= str(prefix)+'br_estimates.txt'
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
            reachable_indices = getattr(self.BrAgent, "reachable_indices", list(range(self.num_actions)))
            if not reachable_indices:
                raise ValueError("Attacker has no feasible paths to any exit.")
            prob = self.N_a / self.N_a.sum()
            reachable_prob = np.asarray([prob[idx] for idx in reachable_indices], dtype=float)
            prob_sum = reachable_prob.sum()
            if not np.isfinite(prob_sum) or prob_sum <= 0:
                reachable_prob = np.ones(len(reachable_indices), dtype=float)
            reachable_prob = reachable_prob / reachable_prob.sum()
            action = np.random.choice(reachable_indices, 1, p=reachable_prob).item()
            self.BrAgent.selected_exit = self.BrAgent.exits[action]
            self.BrAgent.set_path()


class AgentNFSP(object):

    def __init__(self, BrAgent, avg_net, avg_buffer, br_prob=0.1, avg_lr=0.01, sl_mode='aa'):
        self.br_prob = br_prob
        self.is_br = False
        self.is_expl = False
        self.avg_lr = avg_lr
        self.BrAgent = BrAgent
        self.br_buffer = self.BrAgent.buffer

        self.avg_net = avg_net.to(device)
        self.avg_buffer = avg_buffer
        # self.avg_optimizer = optim.SGD(
        # self.avg_net.parameters(), lr=self.avg_lr, momentum=0.99,
        # nesterov=True)
        self.avg_optimizer = optim.Adam(
            self.avg_net.parameters(), lr=self.avg_lr)
        self._step_counter = 0
        self._decay_duration = int(1e7)
        if hasattr(self.BrAgent, 'num_defender'):
            self.num_defender = self.BrAgent.num_defender
        assert sl_mode in ['aa', 'drrn', 'ma']
        self.sl_mode = sl_mode

        self.player_idx = self.BrAgent.player_idx
        self.map = self.BrAgent.map

    def sample_mode(self, exlp_prob=0.):
        decay_br_prob = self._br_prob_decay()
        if np.random.rand() < decay_br_prob:
            self.is_br = True
            self.is_expl = False
        else:
            self.is_br = False
            if np.random.rand() < exlp_prob:
                self.is_expl = True
            else:
                self.is_expl = False

    def _br_prob_decay(self, power=3.0):
        # decay_steps = min(self._step_counter, self._decay_duration)
        # decay_br_prob=(self.br_prob*0.1 + (self.br_prob - self.br_prob*0.1) *
        #     (1 - decay_steps / self._decay_duration)**power)
        # return decay_br_prob
        # decay_steps = min(self._step_counter, self._decay_duration)
        # decay_br_prob = (0.1 + (0.3 - 0.1) *
        #                  (decay_steps / self._decay_duration)**power)
        # return decay_br_prob
        return self.br_prob

    def select_action(self, observation, legal_actions, is_evaluation=False):
        assert len(observation) == 1
        assert len(legal_actions) == 1
        self._step_counter += 1

        if self.is_br:
            return self.BrAgent.select_action(observation, legal_actions, is_evaluation)
        else:
            with torch.no_grad():
                if not self.is_expl:
                    prob = self.action_probs(
                        observation, legal_actions, numpy=False)
                    action_idx = torch.multinomial(
                        prob, num_samples=1).item()
                else:
                    action_idx = np.random.choice(
                        len(legal_actions[0]), 1).item()
                action = legal_actions[0][action_idx]
                return action, action_idx

    def action_probs(self, observation, legal_actions, numpy=False):
        with torch.no_grad():
            if self.sl_mode == 'aa':
                prob = F.softmax(self.avg_net(observation), dim=-1)
                legal_actions_idx = legal_actions[0]
                if self.num_defender:
                    if self.num_defender > 1:
                        legal_actions_idx = [self._mutil_loc_to_idx(
                            loc) for loc in legal_actions[0]]
                prob = prob[legal_actions_idx]
                prob /= prob.sum()
            elif self.sl_mode == 'drrn':
                prob = F.softmax(self.avg_net(observation, legal_actions), dim=-1)
            elif self.sl_mode == 'ma':
                prob = F.softmax(self.avg_net(observation), dim=-1)[
                    :len(legal_actions[0])]
                prob /= prob.sum()
            else:
                ValueError
            if numpy:
                return prob.cpu().numpy()
            else:
                return prob

    def _mutil_loc_to_idx(self, loc):
        # loc: (loc0, loc1,...)
        idx = 0
        for b in range(self.num_defender):
            assert loc[b] <= self.avg_net.num_nodes
            idx += loc[b] * pow(self.avg_net.num_nodes + 1, b)
        return idx

    def learning_br_net(self, transitions):
        loss = self.BrAgent.learning_step(transitions)
        return loss

    def learning_avg_net(self, samples):
        if self.sl_mode == 'aa':
            obs = [s.obs for s in samples]
            action = [s.action[0] for s in samples]
            action_idx = action
            if self.num_defender:
                if self.num_defender > 1:
                    action_idx = [
                        self._mutil_loc_to_idx(loc) for loc in action]
            labels = torch.tensor(action_idx, dtype=torch.long, device=device)
            logis = self.avg_net(obs)
        elif self.sl_mode == 'ma':
            obs = [s.obs for s in samples]
            action_idx = [s.action[1] for s in samples]
            labels = torch.tensor(action_idx, dtype=torch.long, device=device)
            logis = self.avg_net(obs)
        elif self.sl_mode == 'drrn':
            obs = [s.obs for s in samples]
            #legal_actions = [s.legal_actions for s in samples]
            legal_actions = [query_legal_actions(
                self.player_idx, s.obs, self.map, self.num_defender) for s in samples]
            action_idx = [s.action[1] for s in samples]
            labels = torch.tensor(action_idx, dtype=torch.long, device=device)
            logis = self.avg_net(obs, legal_actions)
        else:
            ValueError
        loss = F.cross_entropy(logis, labels)
        self.avg_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_net.parameters(), 1)
        self.avg_optimizer.step()
        return loss

    def save_model(self, save_folder, prefix=None):
        avg_net_name = 'avg_net.pt'
        br_net_name = 'br_net.pt'
        if prefix:
            avg_net_name = str(prefix) + 'avg_net.pt'
            br_net_name = str(prefix) + 'br_net.pt'
        torch.save(self.avg_net.state_dict(), join(save_folder, avg_net_name))
        torch.save(self.BrAgent.policy_net.state_dict(),
                   join(save_folder, br_net_name))

    def load_model(self, save_folder, prefix=None):
        avg_net_name = 'avg_net.pt'
        br_net_name = 'br_net.pt'
        if prefix:
            avg_net_name = str(prefix) + 'avg_net.pt'
            br_net_name = str(prefix) + 'br_net.pt'
        self.avg_net.load_state_dict(torch.load(
            join(save_folder, avg_net_name), map_location=device))
        self.BrAgent.policy_net.load_state_dict(torch.load(
            join(save_folder, br_net_name), map_location=device))

    def set_mode(self, mode):
        assert mode in ['avg', 'br']
        if mode == 'avg':
            self.is_br = False
        else:
            self.is_br = True
        self.is_expl = False

    def reset(self):
        pass
