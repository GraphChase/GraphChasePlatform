from __future__ import annotations

from graphchase.graph.game_settings import GameSettings


class Graph:
    def __init__(self, settings: GameSettings) -> None:
        self.settings = settings
        self.graph = settings.graph
        node_ids = sorted(self.graph.nodes())
        if not node_ids:
            raise ValueError("Graph must contain at least one node")
        self.node_id_map = {node: idx + 1 for idx, node in enumerate(node_ids)}
        self.reverse_node_map = {idx + 1: node for idx, node in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        self.defender_init = [tuple(self.node_id_map[node] for node in settings.defender_init)]
        self.attacker_init = [self.node_id_map[node] for node in settings.attacker_init]
        self.exits = [self.node_id_map[node] for node in settings.exit_nodes]
        self.num_defender = len(settings.defender_init)
        self.time_horizon = int(settings.time_horizon)
        self.adjlist = self._build_adjlist()
        max_actions = max((len(actions) for actions in self.adjlist.values()), default=1)
        self.degree = max_actions
        self.max_actions = max_actions ** max(1, self.num_defender)

    def _build_adjlist(self) -> dict[int, list[int]]:
        adjlist: dict[int, list[int]] = {}
        for node in self.graph.nodes():
            neighbors = self.settings.neighbor_map.get(node, set()) if self.settings.neighbor_map else set(self.graph.neighbors(node))
            actions = sorted(set(neighbors) | {node})
            adjlist[self.node_id_map[node]] = [self.node_id_map[action] for action in actions]
        return adjlist
