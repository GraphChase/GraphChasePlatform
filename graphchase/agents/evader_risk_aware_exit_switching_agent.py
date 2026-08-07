from __future__ import annotations

from typing import Any

import networkx as nx

from graphchase.interfaces.agent_base import AgentBase


class EvaderRiskAwareExitSwitchingAgent(AgentBase):
    def __init__(
        self,
        graph: nx.Graph,
        exit_nodes: list[int],
        risk_weight: float = 1.0,
        safety_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.exit_nodes = [int(node) for node in exit_nodes]
        self.risk_weight = float(risk_weight)
        self.safety_weight = float(safety_weight)
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

    def _action_anchor_node(self, position: tuple[int, int, float], action: int) -> int:
        start, end, dist_to_end = int(position[0]), int(position[1]), float(position[2])
        if action == 0:
            if start == end:
                return start
            edge_weight = self._edge_weight(start, end)
            dist_to_start = max(edge_weight - dist_to_end, 0.0)
            return end if dist_to_end <= dist_to_start else start
        if start == end:
            return int(action)
        if int(action) == start:
            return start
        if int(action) == end:
            return end
        raise ValueError(f"Illegal action {action} for attacker position {position}")

    def _action_distance_to_exit(
        self,
        position: tuple[int, int, float],
        action: int,
        exit_node: int,
        remaining_time: float,
    ) -> float:
        start, end, dist_to_end = int(position[0]), int(position[1]), float(position[2])
        if action == 0:
            return self._position_distance_to_exit(position, exit_node) + float(remaining_time)
        if start == end:
            return self._edge_weight(start, int(action)) + self._node_distance_to_exit(int(action), exit_node)
        edge_weight = self._edge_weight(start, end)
        if int(action) == end:
            return dist_to_end + self._node_distance_to_exit(end, exit_node)
        if int(action) == start:
            return max(edge_weight - dist_to_end, 0.0) + self._node_distance_to_exit(start, exit_node)
        raise ValueError(f"Illegal action {action} for attacker position {position}")

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        attacker_state = obs.get("attacker_state", [])
        defender_state = obs.get("defender_state", [])
        legal_actions = info.get("attacker_legal_action", [])
        remaining_time = float(info.get("remaining_time", 1.0))
        if len(attacker_state) != len(legal_actions):
            raise ValueError("Attacker state and legal action counts do not match")

        defender_positions = [tuple(position) for position in defender_state]
        actions: list[int] = []
        for position, action_space in zip(attacker_state, legal_actions):
            best_action = None
            best_score = None
            best_escape_cost = None
            for action in [int(candidate) for candidate in action_space]:
                anchor_node = self._action_anchor_node(tuple(position), action)
                if defender_positions:
                    min_defender_anchor_dist = min(
                        self._position_distance_to_node(defender_position, anchor_node)
                        for defender_position in defender_positions
                    )
                else:
                    min_defender_anchor_dist = float("inf")

                for exit_node in self.exit_nodes:
                    escape_cost = self._action_distance_to_exit(tuple(position), action, exit_node, remaining_time)
                    if defender_positions:
                        min_defender_exit_dist = min(
                            self._position_distance_to_exit(defender_position, exit_node)
                            for defender_position in defender_positions
                        )
                    else:
                        min_defender_exit_dist = float("inf")
                    threat_penalty = max(0.0, escape_cost - min_defender_exit_dist)
                    score = escape_cost + self.risk_weight * threat_penalty - self.safety_weight * min_defender_anchor_dist
                    candidate_key = (score, escape_cost, -min_defender_anchor_dist, action, exit_node)
                    if best_score is None or candidate_key < best_score:
                        best_score = candidate_key
                        best_escape_cost = escape_cost
                        best_action = action

            if best_action is None or best_escape_cost is None:
                raise RuntimeError("Failed to choose attacker heuristic action")
            actions.append(int(best_action))
        return actions

    def act(self, observation: Any, state: Any = None, **kwargs) -> tuple[list[int], Any]:
        info = kwargs.get("info")
        if info is None:
            raise ValueError("EvaderRiskAwareExitSwitchingAgent.act requires info")
        return self.policy_action(observation, info), state

    def value(self, observation: Any, state: Any = None, **kwargs) -> float:
        return 0.0

    def learnable_params(self) -> dict[str, Any]:
        return {}

    def reset_state(self) -> None:
        return None

    def reset(self) -> None:
        self.reset_state()
