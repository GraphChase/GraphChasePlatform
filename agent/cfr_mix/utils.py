import itertools
EPS = 0.0001


class DeepInformationSet(object):
    def __init__(self, available_actions, history, player_id):
        self.available_actions = available_actions
        self.history = history
        self.action_number = len(self.available_actions)

        if player_id == 0:
            self.key = history[0::2]
        else:
            self.key = [history[0:-1:2], history[1:-1:2]]


def get_deep_information_set(graph, history, player_id):
    if player_id == 1:
        action_set = []
        for l in history[-2]:
            action_set.append([i + 1 for i in range(len(graph.adjacent[l - 1]))])
        action = list(itertools.product(*action_set))
    else:
        action = [i + 1 for i in range(len(graph.adjacent_not_i[history[-2] - 1]))]

    info = DeepInformationSet(action, history, player_id)
    return info


class MixInformationSet(object):
    def __init__(self, action_set, history, player_id):
        if player_id == 0:
            self.key = history[0::2]
            self.available_actions = action_set
            self.action_number = len(self.available_actions)
        else:
            self.key = [history[0:-1:2], history[1:-1:2]]
            self.action_set = action_set
            self.action_number = 1
            for action in self.action_set:
                self.action_number = self.action_number * len(action)


def get_mix_information_set(graph, history, player_id):
    if player_id == 1:
        action_set = []
        for l in history[-2]:
            action_set.append([i + 1 for i in range(len(graph.adjacent[l - 1]))])
    else:
        action_set = [i + 1 for i in range(len(graph.adjacent_not_i[history[-2] - 1]))]

    info = MixInformationSet(action_set, history, player_id)
    return info


def run_action(action_index, current_location, graph):
    if type(action_index) == int:
        return graph.adjacent_not_i[current_location - 1][action_index - 1]
    else:
        action = []
        for i, l in enumerate(current_location):
            action.append(graph.adjacent[current_location[i] - 1][action_index[i] - 1])
        return tuple(action)


def is_terminal(graph, history, time_horizon):
    terminal_flag = False
    if len(history) % 2 == 0:
        if len(history) == (time_horizon + 1) * 2:
            terminal_flag = True
        else:
            defender_location = history[-1]
            attacker_location = history[-2]
            if (attacker_location in graph.exit_node) or (attacker_location in defender_location):
                terminal_flag = True
    return terminal_flag


def terminal_util(graph, history, player_id):
    defender_location = history[-1]
    attacker_location = history[-2]

    if (attacker_location in graph.exit_node) and (attacker_location not in defender_location):
        attacker_util = 0
    else:
        attacker_util = -1

    defender_util = - attacker_util
    return attacker_util if player_id == 0 else defender_util
