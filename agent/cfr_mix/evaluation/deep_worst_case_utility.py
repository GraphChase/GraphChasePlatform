import random
from agent.cfr_mix.utils import run_action, is_terminal, terminal_util, get_deep_information_set
import numpy as np
import logging
import torch
from player_class.deep_defender_class import Defender

def traverse_tree(graph, opponent_agent, history, path, time_horizon):
    if is_terminal(graph, history, time_horizon):
        return terminal_util(graph, history, 1)

    if len(history) % 2 == 0:
        time = int(len(history) / 2)
        action = path[time]
        next_history = history[:]
        next_history.append(action)
        reward = traverse_tree(graph, opponent_agent, next_history, path, time_horizon)
    else:
        info = get_deep_information_set(graph, history, 1)
        strategy = opponent_agent.get_average_strategy(info)
        action_index, _ = sample_action(info.available_actions, strategy)
        action = run_action(action_index, history[-2], graph)
        next_history = history[:]
        next_history.append(action)
        reward = traverse_tree(graph, opponent_agent, next_history, path, time_horizon)
    return reward


def best_response_value(graph, opponent, history, time_horizon):
    path_set = graph.get_path(history[0], time_horizon, False)
    print('path', len(path_set))
    utility = np.zeros(len(path_set))
    for i, p in enumerate(path_set):
        for j in range(1000):
            utility[i] += traverse_tree(graph, opponent, history, p, time_horizon)
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


def evaluation_deep_cfr(time_horizon, game_graph, init_location, defender_hidden_dim, strategy_model_file, number):
    exploitability = []
    for i in number:
        print(i)
        Defender_player = Defender(time_horizon=time_horizon, player_number=len(init_location[1]), hidden_dim=defender_hidden_dim)
        Defender_player.strategy_model = torch.load(strategy_model_file.format(i))
        deep_u0, _ = best_response_value(game_graph, Defender_player, init_location, time_horizon)
        exploitability.append(deep_u0)
        logging.info("Iteration time:{}, worse case utility:{}".format(i, exploitability))
    return exploitability