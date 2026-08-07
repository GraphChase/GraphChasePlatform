from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from graphchase.graph.game_settings import GameSettings


MAX_DEGREES = 30


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj = adj + np.eye(adj.shape[0])
    degrees = np.array(adj.sum(axis=1))
    degrees = np.diag(np.power(degrees, -0.5).flatten())
    return degrees.dot(adj).dot(degrees)


@dataclass
class AttackerPaths:
    all_paths: list[list[int]]
    paths_by_exit: dict[int, list[list[int]]]


class GrasperGraph:
    def __init__(self, settings: GameSettings, graph_type: str, edge_probability: float) -> None:
        self.graph: nx.Graph = settings.graph
        self.exit_node = list(settings.exit_nodes)
        self.type = graph_type
        self.edge_probability = float(edge_probability)
        self.total_node_number = int(self.graph.number_of_nodes())
        self.use_node_embedding = False
        self.embedding_size = 0
        self.node_embedding_method = "none"
        self.embedding_order = "none"
        self.node_information_type = "all"
        self.node_information_normalize = True
        self.similarity = "cosine"

        self._neighbors: dict[int, list[int]] = {
            node: sorted(list(self.graph.neighbors(node))) for node in self.graph.nodes()
        }
        self.max_branch = max((len(neighbors) for neighbors in self._neighbors.values()), default=0) + 1

    def legal_actions(self, node: int) -> list[int]:
        return [0] + self._neighbors.get(node, [])

    def get_legal_action(self, node: int) -> list[int]:
        return list(range(len(self.legal_actions(node))))


class GrasperGame:
    def __init__(self, settings: GameSettings, args, action_type: str = "exit_node", compute_path: bool = True) -> None:
        self.settings = settings
        graph_type = getattr(args, "graph_type", settings.get_metadata("graph_type", "Custom_Graph"))
        edge_probability = float(getattr(args, "edge_probability", settings.get_metadata("edge_probability", 1.0)))
        self._graph = GrasperGraph(settings, graph_type=graph_type, edge_probability=edge_probability)
        self._time_horizon = int(settings.time_horizon)
        self._max_time_horizon = int(getattr(args, "max_time_horizon", self._time_horizon))
        self._initial_location = [settings.attacker_init[0], list(settings.defender_init)]
        self._defender_num = len(settings.defender_init)
        self.agent_num = self._defender_num + 1
        self.use_mix = bool(getattr(args, "use_mix", True))
        self.use_past_history = bool(getattr(args, "use_past_history", False))
        self.use_equal_action = bool(getattr(args, "use_equal_action", True))
        self.prob_of_obs_attacker = float(getattr(args, "prob_of_obs_attacker", 1.0))
        self.defender_mix_action = self._graph.max_branch
        self.defender_action_num = pow(self.defender_mix_action, self._defender_num)
        self.node_feat_dim = int(getattr(args, "node_feat_dim", 3))
        self.node_list = list(sorted(self._graph.graph.nodes()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}

        self.action_type = action_type
        attacker_paths = self._build_attacker_paths(compute_path)
        if action_type == "exit_node":
            self.attacker_path = attacker_paths.paths_by_exit
        else:
            self.attacker_path = attacker_paths.all_paths

        if self.use_past_history:
            self.defender_state_representation_size = self._time_horizon + 1 + self._defender_num
            self.observation_size = self._time_horizon + 1 + 1
        else:
            self.defender_state_representation_size = 1 + self._defender_num + 1
            self.observation_size = 1 + 1 + 1 + 1

    def _build_attacker_paths(self, compute_path: bool) -> AttackerPaths:
        if not compute_path:
            return AttackerPaths(all_paths=[], paths_by_exit={})
        if not self.settings.attacker_init:
            return AttackerPaths(all_paths=[], paths_by_exit={})
        start_node = self.settings.attacker_init[0]
        all_paths: list[list[int]] = []
        paths_by_exit: dict[int, list[list[int]]] = {exit_node: [] for exit_node in self.settings.exit_nodes}
        for exit_node in self.settings.exit_nodes:
            try:
                shortest_paths = nx.all_shortest_paths(self._graph.graph, source=start_node, target=exit_node)
                for path in shortest_paths:
                    if len(path) <= self._time_horizon + 1:
                        path_list = list(path)
                        all_paths.append(path_list)
                        paths_by_exit[exit_node].append(path_list)
            except nx.NetworkXNoPath:
                continue
        return AttackerPaths(all_paths=all_paths, paths_by_exit=paths_by_exit)

    def condition_to_str(self) -> str:
        return f"T{self._time_horizon}_loc{self._initial_location}_exit{self._graph.exit_node}"

    def get_graph_info(self, ret_adj: bool = False, normalize_adj: bool = True):
        nodes = self.node_list
        num_nodes = len(nodes)
        feat = np.zeros((num_nodes, self.node_feat_dim), dtype=float)
        if self.node_feat_dim >= 1:
            for node in self.settings.exit_nodes:
                if node in self.node_to_idx:
                    feat[self.node_to_idx[node], 0] = 1
        if self.node_feat_dim >= 2 and self.settings.attacker_init:
            attacker_node = self.settings.attacker_init[0]
            if attacker_node in self.node_to_idx:
                feat[self.node_to_idx[attacker_node], 1] = 1
        if self.node_feat_dim >= 3:
            for node in self.settings.defender_init:
                if node in self.node_to_idx:
                    feat[self.node_to_idx[node], 2] += 1
        degrees = np.array([deg for _, deg in self._graph.graph.degree()], dtype=int)
        degrees = np.minimum(degrees, MAX_DEGREES)
        degree_one_hot = np.eye(MAX_DEGREES + 1)[degrees]
        feat = np.concatenate((feat, degree_one_hot), axis=1)
        if not ret_adj:
            return feat
        adj = nx.to_numpy_array(self._graph.graph, nodelist=nodes, dtype=float)
        if normalize_adj:
            adj = _normalize_adj(adj)
        return feat, adj

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_type": self._graph.type,
            "edge_probability": self._graph.edge_probability,
            "attacker_init": list(self.settings.attacker_init),
            "defender_init": list(self.settings.defender_init),
            "exit_nodes": list(self.settings.exit_nodes),
            "time_horizon": self._time_horizon,
        }
