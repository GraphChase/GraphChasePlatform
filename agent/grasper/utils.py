import copy
import numpy as np
from graph.grid_graph import GridGraph
from graph.any_graph import AnyGraph
from env.grid_env import GridEnv
from env.any_graph_env import AnyGraphEnv
import random
import time
import torch.nn as nn
import torch
import math
import dgl
import os

def sample_env(args, default_game=None, env_str_list=None):
    if default_game is None:
        while True:
            gen_graph, evader_path = generate_graph(args, compute_path=True)

            # check whether the game is valid, e.g., there is at least one path that the attacker can evade
            path_length = np.array([len(i[0]) if len(i) > 0 else 0 for i in evader_path.values()])

            if env_str_list is None:
                if sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= args.min_evader_pth_len:
                    if args.graph_type == 'Grid_Graph':
                        env = GridEnv(gen_graph, render_mode="rgb_array")
                    elif args.graph_type == 'SG_Graph':
                        env = AnyGraphEnv(gen_graph, render_mode="rgb_array")
                    break
            else:
                if args.graph_type == 'Grid_Graph':
                    env = GridEnv(gen_graph, render_mode="rgb_array")
                elif args.graph_type == 'SG_Graph':
                    env = AnyGraphEnv(gen_graph, render_mode="rgb_array")
                if sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= args.min_evader_pth_len and env.condition_to_str() not in env_str_list:
                    break
    return env

def generate_graph(args, compute_path=True):
    graph_type = args.graph_type
    row = args.row
    column = args.column
    edge_probability = args.edge_probability
    assert args.min_time_horizon <= args.max_time_horizon
    if args.min_time_horizon < args.max_time_horizon:
        time_horizon = np.random.randint(args.min_time_horizon, args.max_time_horizon)
    else:
        time_horizon = args.max_time_horizon

    if graph_type == 'Grid_Graph':
        exit_node_candidates = [i + 1 for i in range(column)] + [(row - 1) * column + i + 1 for i in range(column)] + \
                               [i * column + 1 for i in range(1, row - 1)] + [i * column + column for i in range(1, row - 1)]
        exit_node = list(np.random.choice(exit_node_candidates, args.num_exit, replace=False))
        exit_node = sorted(exit_node)
        feasible_locations = np.random.permutation([i for i in range(1, row * column + 1) if i not in exit_node])
        initial_locations = [feasible_locations[0], list(np.random.choice(list(feasible_locations[1:]), args.num_defender))]

        graph = GridGraph(initial_locations[1], [initial_locations[0]], exit_node, time_horizon, 
                          row, column, edge_probability)
        
        if compute_path:
            return graph, graph.get_shortest_path(time_horizon)[1] # if args.attacker_type == 'exit_node' else graph.get_shortest_path(time_horizon)[0] 
        else:
            return graph
        
    elif graph_type == 'SG_Graph':
        # max_node_num = 372
        max_node_num = 620
        node_list = list(np.arange(max_node_num) + 1)
        exit_node = list(np.random.choice(node_list, args.num_exit, replace=False))
        exit_node = sorted(exit_node)
        feasible_locations = [node for node in node_list if node not in exit_node]
        initial_locations = [feasible_locations[0], list(np.random.choice(list(feasible_locations[1:]), args.num_defender))]

        # graph = AnyGraph(initial_locations[1], [initial_locations[0]], exit_node, time_horizon, 
        #                  f"/home/shuxin_zhuang/workspace/GraphChase/graph/graph_files/sg.gpickle",
        #                  edge_probability)
        graph = AnyGraph(initial_locations[1], [initial_locations[0]], exit_node, time_horizon, 
                         f"/home/shuxin_zhuang/workspace/GraphChase/graph/graph_files/manhattan.gpickle",
                         edge_probability)
        
        if compute_path:
            return graph, graph.get_shortest_path(time_horizon)[1] # if args.attacker_type == 'exit_node' else graph.get_shortest_path(time_horizon)[0] 
        else:
            return graph        
        
    elif graph_type == 'SF_Graph':
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        max_node_num = args.sf_sw_node_num
        node_list = list(np.arange(max_node_num) + 1)
        rnd_node_list = np.random.permutation(node_list)
        exit_node = rnd_node_list[:args.num_exit]
        exit_node = sorted(exit_node)
        feasible_locations = [node for node in node_list if node not in exit_node]
        rnd_feasible_locations = np.random.permutation(feasible_locations)
        initial_locations = [rnd_feasible_locations[0], list(rnd_feasible_locations[1:args.num_defender + 1])]
    elif graph_type == 'SY_Graph':
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        max_node_num = 200
        candidate_start_nodes = [103, 112, 34, 155, 94, 117, 132, 53, 174, 198, 50, 91, 26, 29, 141, 13, 138, 197]
        node_list = [i+1 for i in range(max_node_num) if i+1 not in candidate_start_nodes]
        rnd_node_list = np.random.permutation(node_list)
        exit_node = rnd_node_list[:args.num_exit]
        exit_node = sorted(exit_node)
        rnd_feasible_locations = np.random.permutation(candidate_start_nodes)
        initial_locations = [rnd_feasible_locations[0], list(rnd_feasible_locations[1:args.num_defender + 1])]
    else:
        raise ValueError(f"Unsupported graph type {graph_type}.")
    
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out
    
def get_dgl_graph(graph:GridGraph, node_feat_dim):
    feat, adj = graph.get_graph_info(node_feat_dim, return_adj=True, normalize_adj=False)
    num_nodes = adj.shape[0]
    node_type_dict = {'default': num_nodes}
    edge_type_dict = {('default', 'default', 'default'): (adj.nonzero()[0], adj.nonzero()[1])}
    hg = dgl.heterograph(edge_type_dict, node_type_dict)
    if "attr" not in hg.ndata:
        hg.ndata["attr"] = torch.FloatTensor(feat)
    return hg

def set_pretrain_model_path(args, iteration):
    save_path = ''
    if args.graph_type == "Grid_Graph":
        save_path = 'data/pretrain_models/grasper_mappo/grid_{}_probability_{}/pretrain_model'\
            .format(args.row * args.column, args.edge_probability)
    elif args.graph_type == 'SG_Graph':
        save_path = f'data/pretrain_models/grasper_mappo/sg_graph_probability_{args.edge_probability}/pretrain_model'
    elif args.graph_type == 'SY_Graph':
        save_path = f'data/pretrain_models/grasper_mappo/sy_graph/pretrain_model'
    elif args.graph_type == 'SF_Graph':
        save_path = f'data/pretrain_models/grasper_mappo/sf_graph_{args.sf_sw_node_num}/pretrain_model'
    if args.use_end_to_end:
        save_path += "/use_e2e"
    else:
        save_path += "/not_e2e"
    save_path = os.path.join(args.save_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    location = "num_gts{}_{}_{}_iter{}_bsize{}_node_feat{}_gnn{}_{}_{}_dnum{}_enum{}_T{}_{}_mep{}"\
        .format(args.num_games, args.num_task, args.num_sample, iteration, args.batch_size, args.node_feat_dim,
                args.gnn_num_layer, args.gnn_hidden_dim, args.gnn_output_dim, args.num_defender, args.num_exit,
                args.min_time_horizon, args.max_time_horizon, args.min_evader_pth_len)

    if args.use_end_to_end:
        if args.use_emb_layer:
            location += "_use_el"
        if args.use_augmentation:
            location += "_aug"
        location += f"_gp{args.pool_size}"
    else:
        if args.use_emb_layer:
            location += "_use_el"
            if args.use_augmentation:
                location += "_aug"
            if args.load_graph_emb_model:
                location += "_load_gem"
            location += f"_gp{args.pool_size}"

    if args.use_act_supervisor:
        location += "_as1_{}_{}_{}".format(args.act_sup_coef_max, args.act_sup_coef_min, args.act_sup_coef_decay)

    return os.path.join(save_path, location)