from __future__ import annotations

from collections.abc import Iterable

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import GameSettings


class graph:
    def __init__(self, settings: GameSettings, env: UNSGEnv | None = None):
        self.settings = settings
        self.graph = settings.graph
        self.exit_node = list(settings.exit_nodes)
        self.env = env
        if settings.neighbor_map is not None:
            neighbor_map = settings.neighbor_map
        else:
            neighbor_map = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}
        self.neighbor_map = neighbor_map
        self._sorted_neighbors = {node: sorted(neighbors) for node, neighbors in neighbor_map.items()}

    def _legal_actions(self, node_id: int, include_stay: bool) -> list[int]:
        if self.env is not None:
            actions = list(self.env._legal_action_list((node_id, node_id, 0.0)))
            if include_stay:
                return actions
            return [action for action in actions if action != 0]
        if node_id not in self._sorted_neighbors:
            raise ValueError(f"Node {node_id} not present in graph")
        neighbors = self._sorted_neighbors[node_id]
        if include_stay:
            return [0] + neighbors
        return list(neighbors)

    def attacker_action_indices(self, node_id: int) -> list[int]:
        return list(range(len(self._legal_actions(node_id, include_stay=False))))

    def defender_action_indices(self, node_id: int) -> list[int]:
        return list(range(len(self._legal_actions(node_id, include_stay=True))))

    def _resolve_action(self, current_node: int, action_value: int) -> int:
        return current_node if action_value == 0 else action_value

    def _map_action_index(self, node_id: int, action_index: int, include_stay: bool) -> int:
        actions = self._legal_actions(node_id, include_stay=include_stay)
        if action_index < 0 or action_index >= len(actions):
            raise ValueError(f"Invalid action index {action_index} for node {node_id} with {len(actions)} actions")
        return self._resolve_action(node_id, actions[action_index])

    def map_attacker_action(self, node_id: int, action_index: int) -> int:
        return self._map_action_index(node_id, action_index, include_stay=False)

    def map_defender_action(self, node_id: int, action_index: int) -> int:
        return self._map_action_index(node_id, action_index, include_stay=True)

    def map_defender_joint_action(self, current_locations: Iterable[int], action_indices: Iterable[int]) -> tuple[int, ...]:
        locations = list(current_locations)
        indices = list(action_indices)
        if len(locations) != len(indices):
            raise ValueError("Defender action index length does not match number of defender locations")
        node_actions = []
        for loc, action_index in zip(locations, indices):
            node_actions.append(self.map_defender_action(int(loc), int(action_index)))
        return tuple(node_actions)

    def get_neighbor_node(self, node_number: int) -> list[int]:
        if node_number not in self._sorted_neighbors:
            raise ValueError(f"Node {node_number} not present in graph")
        return list(self._sorted_neighbors[node_number])

    def get_path(self, node_number_start: int, length: int, flag: bool = True) -> list[list[int]]:
        paths: list[list[int]] = []

        def dfs(path: list[int]):
            current = path[-1]
            if current in self.exit_node and len(path) <= length + 1:
                paths.append(path[:])
            if len(path) >= length + 1:
                return
            for neighbor in self.get_neighbor_node(current):
                if flag and neighbor in path:
                    continue
                path.append(neighbor)
                dfs(path)
                path.pop()

        dfs([node_number_start])
        return paths
