import sys
import torch
import os
import copy
import time
import re
from itertools import product
from common_utils import set_seeds, directory_config
import networkx as nx
import numpy as np

from env.grid_env import GridEnv
from graph.graph_files.custom_graph import CustomGraph
from agent.pretrain_psro.path_evader_runner import PathEvaderRunner

def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate_worst_case_utility.py <test_alg> <pursuer_model_path> <evader_model_path>")
        sys.exit(1)

    evaluate_alg = sys.argv[1]
    pursuer_model_path = sys.argv[2]
    evader_model_path = sys.argv[3]
    sys.argv = sys.argv[0:1] + sys.argv[4:]

    if evaluate_alg == 'nsgzero':
        from configs.nsgzero_configs import parse_args
    elif evaluate_alg == 'nsgnfsp':
        from configs.nsgnfsp_configs import parse_args
    elif evaluate_alg == 'pretrainpsro':
        from configs.pretrain_psro_configs import parse_args
    elif evaluate_alg == 'grasper':
        from configs.grasper_configs import parse_args, get_mappo_config
        parser = get_mappo_config()
        mappo_args = parser.parse_known_args(sys.argv[1:])[0] 

        if torch.cuda.is_available() and args.use_cuda:
            torch.cuda.set_device(args.device_id)        
    else:
        print(f"The tested algorithm has not been implemented in the evaluate code")
        sys.exit(1)

    args = parse_args()
    args.save_path = f"./experiments/evaluate/{evaluate_alg}"
    args.save_path = directory_config(args.save_path)

    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    set_seeds(args.seed)

    if evaluate_alg == 'nsgnfsp' or evaluate_alg == 'nsgzero':
        graph = CustomGraph(args.graph_id)
        env = GridEnv(graph, return_reward_mode='defender', return_legal_action=True, nextstate_as_action=True, render_mode="rgb_array")
    elif evaluate_alg == 'pretrainpsro' or evaluate_alg == 'grasper':
        graph = CustomGraph(args.graph_id)
        env = GridEnv(graph, render_mode="rgb_array")

    # Custom evader path
    if evader_model_path == 'exit_node':
        args.attacker_type = 'exit_node'
        args.strategy_type = 'mix'
        evader_runner = PathEvaderRunner(env, args)
        evader_runner.path_selections.clear()
        for e in evader_runner.exit_nodes:
            evader_runner.path_selections[e] = list(nx.all_simple_paths(evader_runner.graph, 
                                                                        source=evader_runner.evader_position[0], 
                                                                        target=e, 
                                                                        cutoff=graph.time_horizon))
    
    if evaluate_alg == 'nsgnfsp':
        from agent.nsg_nfsp.nsgnfsp_defender_policy import CreateDefender
        from agent.nsg_nfsp.nsgnfsp_attacker_policy import CreateAttacker
        from agent.nsg_nfsp.nsgnfsp_defender_runner import NsgNfspDefenderRunner
        from solver.nsgnfsp_solver import NsgNfspSolver

        defender_policy = CreateDefender(graph, args)
        defender_policy.load_model(os.path.dirname(pursuer_model_path), os.path.basename(pursuer_model_path))
        defender_runner = NsgNfspDefenderRunner(env, defender_policy, args)

        evader_runner_load = copy.deepcopy(evader_runner)
        evader_runner.train([defender_runner], (1.0, ), 1000)
        evader_br = max(evader_runner.q_table)

        defender_policy = CreateDefender(graph, args)
        Attacker = CreateAttacker(graph, args)
        solver = NsgNfspSolver(env, defender_runner, Attacker)
        solver.solve()

        from agent.nsg_nfsp.utils import evaluate
        defender_br = evaluate(env, defender_runner.policy, Attacker, 0, 1000)

        nashconv = evader_br + defender_br
        print(f"NashConv: {nashconv}")
    elif evaluate_alg == 'pretrainpsro':
        from agent.pretrain_psro.node_embedding import NodeEmbedding
        from agent.pretrain_psro.ppo_defender_runner import PretrainPsroDefenderRunner
        from agent.pretrain_psro.ppo_defender_policy import PPOAgent

        prefix_number = []
        for filename in os.listdir(pursuer_model_path):
            # 匹配文件名中带有 'actor.pt' 的文件，并提取前面的数字
            match = re.match(r'(\d+)_actor\.pt', filename)
            if match:
                prefix_number.append(int(match.group(1)))
        prefix_number.sort()

        n2v = NodeEmbedding(graph, args)
        defender_policy = PPOAgent(env, args)

        defender_runner_list = []
        for prefix in prefix_number:
            args.load_defender_model = pursuer_model_path
            args.pretrain_model_checkpoint = prefix
            defender_runner = PretrainPsroDefenderRunner(env, defender_policy, args, n2v)
            defender_runner_list.append(copy.deepcopy(defender_runner))

        meta_strategy_path = os.path.join(pursuer_model_path, f"{len(prefix_number)}_defender_meta_strategy.npy")
        meta_strategy = np.load(meta_strategy_path)
        meta_strategy = meta_strategy / np.sum(meta_strategy)

        # calculate_worst_case_utility(env, defender_runner_list, evader_runner, evaluate_alg, next_state_as_action=False, meta_probability=meta_strategy)

        print(f"Calculate worst case utility Done!")
    elif evaluate_alg == 'grasper':
        from agent.grasper.grasper_mappo_policy import RHMAPPO
        from agent.grasper.mappo_policy import RMAPPO
        from agent.grasper.grasper_mappo_defender_runner import GrasperDefenderRunner

        pretrain_policy = RHMAPPO(env, mappo_args, args)
        finetune_policy = RMAPPO(env, mappo_args, args)        
        defender_runner = GrasperDefenderRunner(pretrain_policy, finetune_policy, args, env)
        defender_runner.load_pre_pretrain_model(env)

        prefix_number = []
        for filename in os.listdir(pursuer_model_path):
            # 匹配文件名中带有 'actor.pt' 的文件，并提取前面的数字
            match = re.match(r'(\d+)_actor\.pt', filename)
            if match:
                prefix_number.append(int(match.group(1)))
        prefix_number.sort()

        defender_runner_list = []
        for prefix in prefix_number:
            defender_runner.load_ft_model(pursuer_model_path, prefix)
            defender_runner_list.append(copy.deepcopy(defender_runner))

        meta_strategy_path = os.path.join(pursuer_model_path, f"{len(prefix_number)}_defender_meta_strategy.npy")
        meta_strategy = np.load(meta_strategy_path)
        meta_strategy = meta_strategy / np.sum(meta_strategy)

        # calculate_worst_case_utility(env, defender_runner_list, evader_runner, evaluate_alg, next_state_as_action=False, meta_probability=meta_strategy)

        print(f"Calculate worst case utility Done!")

if __name__ == "__main__":
    main()