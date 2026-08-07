import logging
import random

import numpy as np
import torch

from graphchase.solver.cfrmix.game_setting import information_set as game
from graphchase.solver.cfrmix.game_setting.information_set import run_action
from graphchase.solver.cfrmix.player_class.deep_team_class import DefenderGroup


def _attacker_escaped(graph, history):
    defender_location = history[-1]
    attacker_location = history[-2]
    return attacker_location in graph.exit_node and attacker_location not in defender_location


def _simulate_escape(graph, opponent_agent, history, path, time_horizon):
    if game.is_terminal(graph, history, time_horizon):
        return _attacker_escaped(graph, history)

    if len(history) % 2 == 0:
        time = int(len(history) / 2)
        action = path[time]
        next_history = history[:]
        next_history.append(action)
        return _simulate_escape(graph, opponent_agent, next_history, path, time_horizon)

    info = game.get_mix_information_set(graph, history, 1)
    strategy = opponent_agent.get_average_strategy(info, history[-2], info.action_set)
    action_sampled_index = []
    for i, location in enumerate(history[-2]):
        agent_action_index, _ = sample_action(info.action_set[i], strategy[i])
        action_sampled_index.append(agent_action_index)

    action_sampled = run_action(action_sampled_index, history[-2], graph)
    next_history = history[:]
    next_history.append(action_sampled)
    return _simulate_escape(graph, opponent_agent, next_history, path, time_horizon)


def _group_paths_by_exit(graph, start_node, time_horizon):
    path_set = graph.get_path(start_node, time_horizon, False)
    paths_by_exit = {}
    for path in path_set:
        if not path:
            continue
        exit_node = path[-1]
        paths_by_exit.setdefault(exit_node, []).append(path)
    return paths_by_exit


def _simulate_exit_escape_rate(graph, opponent, history, time_horizon, paths, num_trials):
    if not paths:
        return float("nan")
    successes = 0
    for _ in range(num_trials):
        path = random.choice(paths)
        if _simulate_escape(graph, opponent, history, path, time_horizon):
            successes += 1
    return successes / num_trials


def sample_action(action_set, sample_probability):
    action, action_probability = 0, 0.0
    temp = random.randint(1, 100000) / 100000.0
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
    eval_episodes = 1000
    for i in number:
        Defender = DefenderGroup(
            time_horizon=time_horizon, player_number=len(init_location[1]), hidden_dim=defender_hidden_dim
        )
        Defender.strategy_model = torch.load(strategy_model_file.format(i))
        paths_by_exit = _group_paths_by_exit(game_graph, init_location[0], time_horizon)
        exit_values = []
        for exit_node, paths in paths_by_exit.items():
            escape_rate = _simulate_exit_escape_rate(
                game_graph, Defender, init_location, time_horizon, paths, eval_episodes
            )
            exit_values.append((exit_node, escape_rate))

        if exit_values:
            worst_exit, max_escape_rate = max(exit_values, key=lambda item: item[1])
            defender_wcu = 1 - max_escape_rate
        else:
            worst_exit = -1
            max_escape_rate = float("nan")
            defender_wcu = float("nan")

        log = "Attacker exit escape rates:\n"
        for exit_node, value in exit_values:
            log += f"Exit {exit_node} escape rate : {value}\n"
        log += f"Worst exit : {worst_exit}, Attacker escape rate : {max_escape_rate}\n"
        log += f"Worst-case defender utility : {defender_wcu}"
        logging.info(log)

        exploitability.append(defender_wcu)
        logging.info("Iteration time:%s, worse case utility:%s", i, exploitability)
    return exploitability
