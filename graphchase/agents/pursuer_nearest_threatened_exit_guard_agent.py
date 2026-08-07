from __future__ import annotations

from typing import Any

import networkx as nx

from graphchase.interfaces.agent_base import AgentBase


class PursuerNearestThreatenedExitGuardAgent(AgentBase):
    def __init__(self, graph: nx.Graph, exit_nodes: list[int]) -> None:
        super().__init__()
        self.graph = graph
        self.exit_nodes = [int(node) for node in exit_nodes]
        self._node_to_node_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="weight"))
        self._node_to_exit_dist = {
            int(exit_node): nx.single_source_dijkstra_path_length(self.graph, int(exit_node), weight="weight")
            for exit_node in self.exit_nodes
        }

    def _edge_weight(self, u: int, v: int) -> float:
        return float(self.graph[u][v]["weight"])

    def _node_distance(self, source: int, target: int) -> float:
        return float(self._node_to_node_dist[int(source)].get(int(target), float("inf")))

    def _node_distance_to_exit(self, node: int, exit_node: int) -> float:
        return float(self._node_to_exit_dist[exit_node].get(int(node), float("inf")))

    def _position_distance_to_node(self, position: tuple[int, int, float], node: int) -> float:
        start, end, dist_to_end = int(position[0]), int(position[1]), float(position[2])
        if start == end:
            return self._node_distance(start, node)
        edge_weight = self._edge_weight(start, end)
        dist_to_start = max(edge_weight - dist_to_end, 0.0)
        via_start = dist_to_start + self._node_distance(start, node)
        via_end = dist_to_end + self._node_distance(end, node)
        return min(via_start, via_end)

    def _position_distance_to_exit(self, position: tuple[int, int, float], exit_node: int) -> float:
        start, end, dist_to_end = int(position[0]), int(position[1]), float(position[2])
        if start == end:
            return self._node_distance_to_exit(start, exit_node)
        edge_weight = self._edge_weight(start, end)
        dist_to_start = max(edge_weight - dist_to_end, 0.0)
        via_start = dist_to_start + self._node_distance_to_exit(start, exit_node)
        via_end = dist_to_end + self._node_distance_to_exit(end, exit_node)
        return min(via_start, via_end)

    def _action_distance_to_target(
        self,
        position: tuple[int, int, float],
        action: int,
        target_node: int,
        remaining_time: float,
    ) -> float:
        start, end, dist_to_end = int(position[0]), int(position[1]), float(position[2])
        if action == 0:
            if start == end and start == target_node:
                return 0.0
            return self._position_distance_to_node(position, target_node) + float(remaining_time)
        if start == end:
            return self._edge_weight(start, int(action)) + self._node_distance(int(action), target_node)
        edge_weight = self._edge_weight(start, end)
        if int(action) == end:
            return dist_to_end + self._node_distance(end, target_node)
        if int(action) == start:
            return max(edge_weight - dist_to_end, 0.0) + self._node_distance(start, target_node)
        raise ValueError(f"Illegal action {action} for defender position {position}")

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        attacker_state = [tuple(position) for position in obs.get("attacker_state", [])]
        defender_state = [tuple(position) for position in obs.get("defender_state", [])]
        legal_actions = info.get("defender_legal_action", [])
        remaining_time = float(info.get("remaining_time", 1.0))
        if len(defender_state) != len(legal_actions):
            raise ValueError("Defender state and legal action counts do not match")
        if not attacker_state:
            return [0 for _ in defender_state]

        threat_by_exit: dict[int, float] = {}
        for exit_node in self.exit_nodes:
            attacker_best_dist = min(
                self._position_distance_to_exit(attacker_position, exit_node)
                for attacker_position in attacker_state
            )
            defender_best_dist = min(
                self._position_distance_to_exit(defender_position, exit_node)
                for defender_position in defender_state
            )
            threat_by_exit[exit_node] = defender_best_dist - attacker_best_dist

        threatened_exits = [exit_node for exit_node, threat in threat_by_exit.items() if threat > 0.0]
        if threatened_exits:
            candidate_exits = threatened_exits
        else:
            max_threat = max(threat_by_exit.values())
            candidate_exits = [
                exit_node for exit_node, threat in threat_by_exit.items() if abs(threat - max_threat) <= 1e-9
            ]

        actions: list[int] = []
        for defender_position, action_space in zip(defender_state, legal_actions):
            assigned_exit = min(
                candidate_exits,
                key=lambda exit_node: (
                    self._position_distance_to_exit(defender_position, exit_node),
                    -threat_by_exit[exit_node],
                    exit_node,
                ),
            )
            best_action = min(
                [int(action) for action in action_space],
                key=lambda action: (
                    self._action_distance_to_target(defender_position, action, assigned_exit, remaining_time),
                    action,
                ),
            )
            actions.append(best_action)
        return actions

    def act(self, observation: Any, state: Any = None, **kwargs) -> tuple[list[int], Any]:
        info = kwargs.get("info")
        if info is None:
            raise ValueError("PursuerNearestThreatenedExitGuardAgent.act requires info")
        return self.policy_action(observation, info), state

    def value(self, observation: Any, state: Any = None, **kwargs) -> float:
        return 0.0

    def learnable_params(self) -> dict[str, Any]:
        return {}

    def reset_state(self) -> None:
        return None

    def reset(self) -> None:
        self.reset_state()
