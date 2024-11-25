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
from agent.cfr_mix.evaluation.mix_deep_worst_case_utility import mixed_traverse_tree

def calculate_worst_case_utility(env, 
                          defender_runner_list: list, 
                          evader_runner:PathEvaderRunner, 
                          alg,
                          next_state_as_action,
                          meta_probability=None, 
                          sample_number=500,
                          custom_horizon=None
                          ) -> list:
    """ 
    Evaluate every path by simulation sample_number times and update self.q_table
    Return: defender's utility for each evader's selection (if evader run out, defender utility is 0, else 1)
    """
    defender_rds_vs_each_path = []
    tmp_res = 2
    for exit_node in evader_runner.path_selections.keys():

        for evader_path in evader_runner.path_selections[exit_node]:
            defender_reward = 0

            for _ in range(sample_number):

                if meta_probability is None:
                    defender_runner = defender_runner_list[0]
                else:
                    defender_idx = np.random.choice(range(len(defender_runner_list)), p=meta_probability)
                    defender_runner = defender_runner_list[defender_idx]

                if next_state_as_action == False:
                    evader_actions = evader_runner._trajectory2actions(evader_path)
                else:
                    evader_actions = evader_path

                if alg == 'cfrmix':                  
                    agent_init_location = []
                    agent_init_location = env.initial_locations[0] + [tuple(env.initial_locations[1])]
                    if custom_horizon is not None:
                        defender_reward += mixed_traverse_tree(env, defender_runner, agent_init_location, evader_actions, custom_horizon)

                else:
                    # rollout
                    terminated = False
                    s_t = time.time()
                    observation, info = env.reset()
                    t = 0

                    while not terminated:
                        evader_act = evader_actions[t+1] if next_state_as_action else evader_actions[t]

                        if alg == 'nsgnfsp':
                            with torch.no_grad():
                                defender_obs = [copy.deepcopy(info["evader_history"]), 
                                                copy.deepcopy(tuple(info["defender_history"][-1]))]
                                
                                def_current_legal_action = list(product(*info["defender_legal_actions"]))

                                defender_a = defender_runner.policy.select_action(
                                    [defender_obs], [def_current_legal_action], is_evaluation=True)

                                actions = np.insert(np.array(defender_a[0]), 0, evader_act)

                        elif alg == "nsgzero":
                            with torch.no_grad():                         
                                defender_obs = (info["evader_history"], info["defender_history"][-1])
                                # defender action
                                defender_act, _ = defender_runner.policy.train_select_act(
                                    defender_obs, info["defender_legal_actions"], prior=False)
                                
                                actions = np.array((evader_act,) + defender_act, dtype=int)
                        
                        elif alg == "pretrainpsro" or alg == 'grasper':
                            with torch.no_grad():
                                defender_action = defender_runner.get_env_actions(observation, t)
                                actions = np.array(defender_action)
                                actions = np.insert(actions, 0, evader_act)

                        observation, reward, terminated, truncated, info = env.step(actions)
                        t += 1

                        if terminated or truncated:
                            if alg == 'nsgnfsp' or alg == 'nsgzero':
                                reward = max(reward, 0.)
                                defender_reward += reward
                            elif alg == 'pretrainpsro' or alg == 'grasper':
                                reward = max(reward[1], 0.)
                                defender_reward += reward                        
                            else:
                                print("To be done")
                                sys.exit(1)
                            s_t = time.time()

            if defender_reward/sample_number < tmp_res:
                tmp_res = defender_reward/sample_number
                print(f"Path: {evader_path}, defender worst case utility:{tmp_res}")
            
            defender_rds_vs_each_path.append(defender_reward/sample_number)
    
    return defender_rds_vs_each_path

def main():
    if len(sys.argv) < 4:
        print("Usage: python evaluate_worst_case_utility.py <test_alg> <custom_time_horizon> <model_path>")
        sys.exit(1)

    evaluate_alg = sys.argv[1]
    custom_horizon = int(sys.argv[2])
    model_path = sys.argv[3]
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
    elif evaluate_alg == 'cfrmix':
        from configs.cfr_mix_configs import parse_args
        args = parse_args()    

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
        graph = CustomGraph(args.graph_id, custom_horizon)
        env = GridEnv(graph, return_reward_mode='defender', return_legal_action=True, nextstate_as_action=True, render_mode="rgb_array")
    elif evaluate_alg == 'pretrainpsro' or evaluate_alg == 'grasper':
        graph = CustomGraph(args.graph_id)
        env = GridEnv(graph, render_mode="rgb_array")
    elif evaluate_alg == 'cfrmix':
        from graph.cfr_graph_wrapper import CfrmixGraph
        
        graph = CustomGraph(args.graph_id)
        game_graph = CfrmixGraph(row=graph.row, column=graph.column, initial_locations=graph.initial_locations, time_horizon=graph.time_horizon)
        env = GridEnv(graph, render_mode="rgb_array")

    # Custom evader path
    args.attacker_type = 'exit_node'
    args.strategy_type = 'mix'
    evader_runner = PathEvaderRunner(env, args)
    evader_runner.path_selections.clear()
    for e in evader_runner.exit_nodes:
        evader_runner.path_selections[e] = list(nx.all_simple_paths(evader_runner.graph, 
                                                                    source=evader_runner.evader_position[0], 
                                                                    target=e, 
                                                                    cutoff=custom_horizon))
    
    if evaluate_alg == 'nsgzero':
        from agent.nsgzero.mcts_defender_policy import NsgzeroDefenderPolicy
        from agent.nsgzero.mcts_defender_runner import NsgzeroDefenderRunner
        
        policy_setup_env = copy.deepcopy(env)
        if args.graph_id == 0 or args.graph_id == 1:
            policy_setup_env.time_horizon = 6
        elif args.graph_id == 2 or args.graph_id == 3:
            policy_setup_env.time_horizon = 4
        else:
            print("To be done")
            sys.exit(1)
        defender_policy = NsgzeroDefenderPolicy(env, args, policy_setup_env)
        defender_runner = NsgzeroDefenderRunner(env, defender_policy, args)
        defender_runner.load_models(model_path)
        calculate_worst_case_utility(env, [defender_runner], evader_runner, evaluate_alg, next_state_as_action=True)
    
        print(f"Calculate worst case utility Done!")
    elif evaluate_alg == 'nsgnfsp':
        from agent.nsg_nfsp.nsgnfsp_defender_policy import CreateDefender
        from agent.nsg_nfsp.nsgnfsp_defender_runner import NsgNfspDefenderRunner

        defender_policy = CreateDefender(graph, args)
        defender_policy.load_model(os.path.dirname(model_path), os.path.basename(model_path))
        defender_runner = NsgNfspDefenderRunner(env, defender_policy, args)
        calculate_worst_case_utility(env, [defender_runner], evader_runner, evaluate_alg, next_state_as_action=True)
    
        print(f"Calculate worst case utility Done!")
    elif evaluate_alg == 'pretrainpsro':
        from agent.pretrain_psro.node_embedding import NodeEmbedding
        from agent.pretrain_psro.ppo_defender_runner import PretrainPsroDefenderRunner
        from agent.pretrain_psro.ppo_defender_policy import PPOAgent

        prefix_number = []
        for filename in os.listdir(model_path):
            # 匹配文件名中带有 'actor.pt' 的文件，并提取前面的数字
            match = re.match(r'(\d+)_actor\.pt', filename)
            if match:
                prefix_number.append(int(match.group(1)))
        prefix_number.sort()

        n2v = NodeEmbedding(graph, args)
        defender_policy = PPOAgent(env, args)

        defender_runner_list = []
        for prefix in prefix_number:
            args.load_defender_model = model_path
            args.pretrain_model_checkpoint = prefix
            defender_runner = PretrainPsroDefenderRunner(env, defender_policy, args, n2v)
            defender_runner_list.append(copy.deepcopy(defender_runner))

        meta_strategy_path = os.path.join(model_path, f"{len(prefix_number)}_defender_meta_strategy.npy")
        meta_strategy = np.load(meta_strategy_path)
        meta_strategy = meta_strategy / np.sum(meta_strategy)

        calculate_worst_case_utility(env, defender_runner_list, evader_runner, evaluate_alg, next_state_as_action=False, meta_probability=meta_strategy)

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
        for filename in os.listdir(model_path):
            # 匹配文件名中带有 'actor.pt' 的文件，并提取前面的数字
            match = re.match(r'(\d+)_actor\.pt', filename)
            if match:
                prefix_number.append(int(match.group(1)))
        prefix_number.sort()

        defender_runner_list = []
        for prefix in prefix_number:
            defender_runner.load_ft_model(model_path, prefix)
            defender_runner_list.append(copy.deepcopy(defender_runner))

        meta_strategy_path = os.path.join(model_path, f"{len(prefix_number)}_defender_meta_strategy.npy")
        meta_strategy = np.load(meta_strategy_path)
        meta_strategy = meta_strategy / np.sum(meta_strategy)

        calculate_worst_case_utility(env, defender_runner_list, evader_runner, evaluate_alg, next_state_as_action=False, meta_probability=meta_strategy)

        print(f"Calculate worst case utility Done!")
    elif evaluate_alg == 'cfrmix':
        from agent.cfr_mix.player_class.deep_team_class import DefenderGroup

        defender_runner = DefenderGroup(time_horizon=env.time_horizon, player_number=env.defender_num, hidden_dim=args.network_dim)

        # Set defender policy
        pattern = r'5DeepMCFR4_mix_probe_cfr_defender_strategy_model_(\d+)\.dat'
        max_number = -1
        latest_model = None
        for filename in os.listdir(model_path):
            match = re.match(pattern, filename)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
                    latest_model = filename    
        model_path = os.path.join(model_path, latest_model)    
        defender_runner.strategy_model = torch.load(model_path)

        calculate_worst_case_utility(game_graph, [defender_runner], evader_runner, evaluate_alg, next_state_as_action=True, custom_horizon=custom_horizon)

        # path_set = graph.get_path(history[0], time_horizon, False)
        # print('path', len(path_set))
        # utility = np.zeros(len(path_set))

        # return min(utility), utility               

if __name__ == "__main__":
    main()