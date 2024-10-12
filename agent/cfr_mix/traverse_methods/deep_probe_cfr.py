from agent.cfr_mix.utils import is_terminal, terminal_util, get_deep_information_set, run_action
import numpy as np
import random


def probe_traverse_tree(graph, one_history, player_id, player_list, time_horizon, sample_number, itr_round):
    terminal_flag = is_terminal(graph, one_history, time_horizon)
    if terminal_flag:
        return terminal_util(graph, one_history, player_id)

    if len(one_history) % 2 == player_id:
        info = get_deep_information_set(graph, one_history, player_id)
        strategy = player_list[player_id].get_strategy(info, itr_round)
        if info.action_number < sample_number:
            action_set = info.available_actions
        else:
            action_set = random.sample(info.available_actions, sample_number)

        sampled_action = random.sample(action_set, 1)
        util_for_action = np.zeros(len(action_set))
        probability_for_action = np.zeros(len(action_set))
        for i, action_index in enumerate(action_set):
            action = run_action(action_index, one_history[-2], graph)
            next_history = one_history[:]
            next_history.append(action)
            if action_index in sampled_action:
                util_for_action[i] = probe_traverse_tree(graph, next_history, player_id, player_list,
                                                        time_horizon, sample_number, itr_round)
            else:
                util_for_action[i] = probe_cfr(graph, next_history, player_id, player_list, time_horizon, itr_round)
            index = info.available_actions.index(action_index)
            probability_for_action[i] = strategy[index]

        util_for_info = np.sum(util_for_action * probability_for_action)
        regret = (util_for_action - util_for_info) * itr_round
        player_list[player_id].regret_memory_add(info, action_set, regret)
        return util_for_info
    else:
        info = get_deep_information_set(graph, one_history, 1-player_id)
        strategy = player_list[1-player_id].get_strategy(info, itr_round)
        action_sampled, probability = sample_action(info.available_actions, strategy)
        player_list[1-player_id].strategy_memory_add(info, info.available_actions, strategy * itr_round)

        action_sampled = run_action(action_sampled, one_history[-2], graph)
        next_history = one_history[:]
        next_history.append(action_sampled)
        util_for_action = probe_traverse_tree(graph, next_history, player_id, player_list, time_horizon, sample_number, itr_round)
        return util_for_action


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


def probe_cfr(graph, one_history, player_id, player_list, time_horizon, itr_round):
    terminal_flag = is_terminal(graph, one_history, time_horizon)
    if terminal_flag:
        return terminal_util(graph, one_history, player_id)

    info = get_deep_information_set(graph, one_history, len(one_history) % 2)
    strategy = player_list[len(one_history) % 2].get_strategy(info, itr_round)
    action_sampled_index, probability_for_sampled_action = sample_action(info.available_actions, strategy)

    action = run_action(action_sampled_index, one_history[-2], graph)
    next_history = one_history[:]
    next_history.append(action)
    util_for_action = probe_cfr(graph, next_history, player_id, player_list, time_horizon, itr_round)
    return util_for_action
