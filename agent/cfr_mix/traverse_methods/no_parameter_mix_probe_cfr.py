import itertools
from agent.cfr_mix.utils import is_terminal, terminal_util, get_mix_information_set, run_action
import numpy as np
import random


def probe_traverse_tree(graph, one_history, player_id, player_list, time_horizon, sample_number, itr_round):
    terminal_flag = is_terminal(graph, one_history, time_horizon)
    if terminal_flag:
        return terminal_util(graph, one_history, player_id)

    if len(one_history) % 2 == player_id:
        if player_id == 0:
            info = get_mix_information_set(graph, one_history, player_id)
            strategy = player_list[player_id].get_strategy(info, itr_round)
            if info.action_number > sample_number:
                action_set = random.sample(info.available_actions, sample_number)
            else:
                action_set = info.available_actions

            action_sampled_index = random.sample(action_set, 1)
            util_for_action = np.zeros(len(action_set))
            probability_for_action = np.zeros(len(action_set))
            for i, action_index in enumerate(action_set):
                index = info.available_actions.index(action_index)
                probability_for_action[i] = strategy[index]

                action = run_action(action_index, one_history[-2], graph)
                next_history = one_history[:]
                next_history.append(action)
                if action_index in action_sampled_index:
                    util_for_action[i] = probe_traverse_tree(graph, next_history, player_id, player_list,
                                                            time_horizon, sample_number, itr_round)
                else:
                    util_for_action[i] = probe(graph, next_history, player_id, player_list, time_horizon, itr_round)
        else:
            info = get_mix_information_set(graph, one_history, player_id)
            strategy_list = player_list[player_id].get_strategy(info, one_history[-2], info.action_set, itr_round)
            if info.action_number < sample_number:
                action_set = list(itertools.product(*info.action_set))
            else:
                # all_action = list(itertools.product(*info.action_set))
                # action_set = random.sample(all_action, sample_number)
                action_set = []
                for _ in range(sample_number):
                    action_sampled = []
                    for i in range(len(one_history[-2])):
                        agent_action_index = random.sample(info.action_set[i], 1)[0]
                        action_sampled.append(agent_action_index)
                    if tuple(action_sampled) not in action_set:
                        action_set.append(tuple(action_sampled))

            action_sampled_index = random.sample(action_set, 1)
            util_for_action = np.zeros(len(action_set))
            probability_for_action = np.zeros(len(action_set))
            for i, action_index in enumerate(action_set):
                probability = 1
                for idx, a in enumerate(action_index):
                    index = info.action_set[idx].index(a)
                    probability = probability * strategy_list[idx][index]
                probability_for_action[i] = probability

                action = run_action(action_index, one_history[-2], graph)
                next_history = one_history[:]
                next_history.append(action)
                if action_index in action_sampled_index:
                    util_for_action[i] = probe_traverse_tree(graph, next_history, player_id, player_list, time_horizon, sample_number, itr_round)
                else:
                    util_for_action[i] = probe(graph, next_history, player_id, player_list, time_horizon, itr_round)

        util_for_info = np.sum(util_for_action * probability_for_action)
        regret = (util_for_action - util_for_info) * itr_round
        player_list[player_id].regret_memory_add(info, action_set, regret)
        return util_for_info
    else:
        if 1 - player_id == 0:
            info = get_mix_information_set(graph, one_history, 1 - player_id)
            strategy = player_list[1-player_id].get_strategy(info, itr_round)
            action_sampled, action_probability = sample_action(info.available_actions, strategy)
            player_list[1-player_id].strategy_memory_add(info, info.available_actions, strategy * itr_round)
        else:
            info = get_mix_information_set(graph, one_history, 1 - player_id)
            strategy_list = player_list[1-player_id].get_strategy(info, one_history[-2], info.action_set, itr_round)
            action_sampled = []
            for i, location in enumerate(one_history[-2]):
                agent_action, _ = sample_action(info.action_set[i], strategy_list[i])
                action_sampled.append(agent_action)
                player_list[1-player_id].strategy_memory_add(info, i, location, info.action_set[i], strategy_list[i]* itr_round)
            action_sampled = tuple(action_sampled)

        action = run_action(action_sampled, one_history[-2], graph)
        next_history = one_history[:]
        next_history.append(action)
        util_for_action = probe_traverse_tree(graph, next_history, player_id, player_list, time_horizon, sample_number, itr_round)
        return util_for_action


def probe(graph, one_history, player_id, player_list, time_horizon, itr_round):
    if is_terminal(graph, one_history, time_horizon):
        return terminal_util(graph, one_history, player_id)
    else:
        if len(one_history) % 2 == 0:
            info = get_mix_information_set(graph, one_history, 0)
            strategy = player_list[0].get_strategy(info, itr_round)
            action_sampled, action_probability = sample_action(info.available_actions, strategy)
        else:
            info = get_mix_information_set(graph, one_history, 1)
            strategy_list = player_list[1].get_strategy(info, one_history[-2], info.action_set, itr_round)
            action_sampled = []
            for i in range(len(one_history[-2])):
                agent_action, _ = sample_action(info.action_set[i], strategy_list[i])
                action_sampled.append(agent_action)
            action_sampled = tuple(action_sampled)

        action = run_action(action_sampled, one_history[-2], graph)
        next_history = one_history[:]
        next_history.append(action)
        return probe(graph, next_history, player_id, player_list, time_horizon, itr_round)


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
