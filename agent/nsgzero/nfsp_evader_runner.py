import random
import numpy as np
import networkx as nx
import math
from env.base_env import BaseGame

class NFSPAttacker:
    def __init__(self, game : BaseGame, args):
        self.game = game
        self.args = args
        self.nx_graph : nx.graph = game.graph.graph
        self.attacker_init = game._evader_initial_locations
        self.exits = list(game._exit_locations)
        self.time_horizon = game.time_horizon
        self.paths = {}
        self.path = None
        self.require_update=True

        # if args.graph_id >=2:
        #     for i,e in enumerate(self.exits):
        #         self.paths[e]=list(nx.all_simple_paths(self.nx_graph,
        #                  self.attacker_init[0], e, cutoff=self.game.graph.t_cutoff[i]-3))
        # else:
        for e in self.exits:
            self.paths[e] = list(nx.all_simple_paths(self.nx_graph,
                                                     self.attacker_init[0], e, cutoff=self.time_horizon))
        self.paths = {k: v for k, v in self.paths.items() if v}  # v为空列表时会被过滤掉
        self.exits = [int(key) for key in self.paths.keys()]

        self.ban_capacity=args.ban_capacity # bandit capacity
        self._next_idx=0
        self.acts=[]
        self.act_values=[]
        self.n_acts = dict.fromkeys(self.exits, 1) # track the number of actions being chosen recently
        self.act_est = dict.fromkeys(self.exits, 0)

        self.cache_capacity=args.cache_capacity
        self.N_acts=np.ones(len(self.exits))
        self.cache=np.zeros(len(self.exits))

        self.br_rate=args.br_rate

    def update(self, act, val):
        if len(self.acts) < self.ban_capacity:
            self.acts.append(act)
            self.act_values.append(val)
            self.act_est[act] += (val-self.act_est[act])/(self.n_acts[act]+1)
            self.n_acts[act]+=1
        else:
            assert len(self.acts) == self.ban_capacity
            self.act_est[act] += (val-self.act_est[act])/(self.n_acts[act]+1)

            del_act=self.acts[self._next_idx]
            del_val=self.act_values[self._next_idx]
            self.act_est[del_act] -= (del_val-self.act_est[del_act])/(self.n_acts[del_act]-1)
            self.n_acts[del_act]-=1

            self.acts[self._next_idx]=act
            self.act_values[self._next_idx]=val
            self.n_acts[act]+=1

            self._next_idx += 1
            self._next_idx %= self.ban_capacity
        
        if self.cache.sum() >= self.cache_capacity:
            self.N_acts+=self.cache
            self.cache = np.zeros(len(self.exits))

    def select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        act = self.path[self.t]
        self.t += 1
        return act        
    
    def train_select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        legal_act = self.game.graph.adjlist[self.path[self.t-1]]
        act = self.path[self.t]
        act_idx = legal_act.index(act)
        self.t += 1
        return act, act_idx

    def reset(self, train=True):
        self.is_br=True
        value = []
        for key in self.exits:
            value.append(self.act_est[key])
        q_value = [math.exp(i) for i in value]
        sum_value = sum(q_value)
        strategy = [q / sum_value for q in q_value]
        selected_exit=np.random.choice(self.exits, 1, p=np.array(strategy)).item()
        self.path = random.choice(self.paths[selected_exit])
        if train:
            idx = self.exits.index(selected_exit)
            self.cache[idx]+=1

        self.t=1
        self.selected_exit=selected_exit

    def synch(self, act_est, N_acts):
        self.act_est=act_est
        self.N_acts=N_acts    