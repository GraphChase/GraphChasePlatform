import random
from agent.cfr_mix.utils import run_action, is_terminal, terminal_util, get_mix_information_set
import numpy as np
import logging
import torch
from agent.cfr_mix.player_class.deep_team_class import DefenderGroup


def mixed_traverse_tree(graph, opponent_agent, history, path, time_horizon):
    if is_terminal(graph, history, time_horizon):
        return terminal_util(graph, history, 1)

    if len(history) % 2 == 0:
        time = int(len(history) / 2)
        action = path[time]
        next_history = history[:]
        next_history.append(action)
        reward = mixed_traverse_tree(graph, opponent_agent, next_history, path, time_horizon)
        return reward
    else:
        info = get_mix_information_set(graph, history, 1)
        strategy = opponent_agent.get_average_strategy(info, history[-2], info.action_set)
        action_sampled_index = []
        for i, location in enumerate(history[-2]):
            agent_action_index, _ = sample_action(info.action_set[i], strategy[i])
            action_sampled_index.append(agent_action_index)

        action_sampled = run_action(action_sampled_index, history[-2], graph)
        next_history = history[:]
        next_history.append(action_sampled)
        action_utils = mixed_traverse_tree(graph, opponent_agent, next_history, path, time_horizon)
        return action_utils
 

def mixed_best_response_value(graph, opponent, history, time_horizon):
    path_set = graph.get_path(history[0], time_horizon, False)
    print('path', len(path_set))
    # path_set = [path_set[87], path_set[96], path_set[103]]
    # path_set = [path_set[25], path_set[439], path_set[435], path_set[430], path_set[410], path_set[401], path_set[426], path_set[406], path_set[374], path_set[243], path_set[45]]
    utility = np.zeros(len(path_set))
    for i, p in enumerate(path_set):
        for j in range(1000):
            utility[i] += mixed_traverse_tree(graph, opponent, history, p, time_horizon)
        utility[i] = utility[i] / 1000
        print("path", i + 1, utility[i])
    return min(utility), utility


def sample_action(action_set, sample_probability):
    action, action_probability = 0, 0.0
    temp = random.randint(1, 100000) / 100000.
    strategy_sum = 0
    for i in range(0, len(action_set)):
        strategy_sum += sample_probability[i]
        if temp <= strategy_sum:
            action = action_set[i]
            action_probability = sample_probability[i]
            break
        elif i == len(action_set) - 1:
            action = action_set[i]
            action_probability = sample_probability[i]
            break
    return action, action_probability


def evaluation_mix_cfr(time_horizon, game_graph, init_location, defender_hidden_dim, strategy_model_file, number):
    exploitability = []
    for i in number:
        print(i)
        Defender = DefenderGroup(time_horizon=time_horizon, player_number=len(init_location[1]), hidden_dim=defender_hidden_dim)
        Defender.strategy_model = torch.load(strategy_model_file.format(i))
        deep_u0, _ = mixed_best_response_value(game_graph, Defender, init_location, time_horizon)
        exploitability.append(deep_u0)
        logging.info("Iteration time:{}, worse case utility:{}".format(i, exploitability))
    return exploitability