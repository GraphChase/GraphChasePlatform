from __future__ import annotations

import networkx as nx

from graphchase.graph.game_settings import GameSettings


class Maps:
    def __init__(self, settings: GameSettings):
        if len(settings.attacker_init) != 1:
            raise ValueError("NFSP currently supports a single attacker start node.")

        self.graph = settings.graph
        self.exits = list(settings.exit_nodes)
        self.num_defender = len(settings.defender_init)

        attacker_node = int(settings.attacker_init[0])
        self.attacker_init = [attacker_node]

        if self.num_defender == 1:
            self.defender_init = [int(settings.defender_init[0])]
        else:
            self.defender_init = [tuple(int(node) for node in settings.defender_init)]

        adjlist = nx.to_dict_of_lists(self.graph)
        max_actions = 0
        for node, neighbors in adjlist.items():
            actions = list(neighbors)
            if node not in actions:
                actions.append(node)
            actions = sorted(actions)
            adjlist[node] = actions
            max_actions = max(max_actions, len(actions))

        node_ids = sorted(adjlist.keys())
        if not node_ids:
            raise ValueError("Graph contains no nodes.")
        if min(node_ids) <= 0:
            raise ValueError("NFSP expects node indices to start from 1.")
        if max(node_ids) != len(node_ids):
            raise ValueError("NFSP expects contiguous node indices from 1..N.")

        self.num_nodes = len(adjlist)
        self.adjlist = adjlist
        self.max_actions = pow(max_actions, self.num_defender)
