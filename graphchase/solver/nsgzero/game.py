from __future__ import annotations

import copy
from itertools import product
import networkx as nx

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import GameSettings
from graphchase.solver.nsgzero.graph import Graph


class Game:
    def __init__(self, settings: GameSettings) -> None:
        self.settings = settings
        self.graph = Graph(settings)
        self.nx_graph = settings.graph
        self.adjlist = self.graph.adjlist
        self.time_horizon = self.graph.time_horizon
        self.defender_init = self.graph.defender_init
        self.attacker_init = self.graph.attacker_init
        self.exits = self.graph.exits
        self.num_defender = self.graph.num_defender
        self.num_nodes = self.graph.num_nodes
        self.node_id_map = self.graph.node_id_map
        self.reverse_node_map = self.graph.reverse_node_map
        self.env = UNSGEnv(settings)
        self.last_reward = {"defender": 0.0, "attacker": 0.0}
        self.terminated = False
        self.truncated = False
        self.reset()

    def reset(self):
        self.env.reset()
        self.last_reward = {"defender": 0.0, "attacker": 0.0}
        self.terminated = False
        self.truncated = False
        attacker_nodes, defender_nodes = self._current_internal_positions()
        if len(attacker_nodes) != 1:
            raise ValueError("NSGZero currently supports a single attacker")
        attacker_his = [attacker_nodes[0]]
        defender_his = [tuple(defender_nodes)]
        self.current_state = GameState(self, copy.deepcopy(defender_his), copy.deepcopy(attacker_his))
        return self.current_state

    def step(self, defender_act, attacker_act):
        if not isinstance(defender_act, tuple) or not isinstance(attacker_act, int):
            raise ValueError("Defender act must be tuple and attacker act must be int")
        attacker_pos = self.env.positions[0]
        defender_positions = self.env.positions[self.env.num_attackers :]
        env_attacker_act = self._map_internal_action_to_env(attacker_act, attacker_pos)
        env_defender_act = [
            self._map_internal_action_to_env(action, pos) for action, pos in zip(defender_act, defender_positions)
        ]
        _, reward, terminated, truncated, _ = self.env.step(
            {"attacker_action": [env_attacker_act], "defender_action": env_defender_act}
        )
        self.last_reward = reward
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        attacker_nodes, defender_nodes = self._current_internal_positions()
        defender_his = copy.deepcopy(self.current_state.defender_his)
        attacker_his = copy.deepcopy(self.current_state.attacker_his)
        if len(attacker_nodes) != 1:
            raise ValueError("NSGZero currently supports a single attacker")
        defender_his.append(tuple(defender_nodes))
        attacker_his.append(attacker_nodes[0])
        self.current_state = GameState(self, defender_his, attacker_his)
        return self.current_state

    def _current_internal_positions(self) -> tuple[list[int], list[int]]:
        attacker_positions = self.env.positions[: self.env.num_attackers]
        defender_positions = self.env.positions[self.env.num_attackers :]
        attacker_nodes = [self._position_to_internal(pos) for pos in attacker_positions]
        defender_nodes = [self._position_to_internal(pos) for pos in defender_positions]
        return attacker_nodes, defender_nodes

    def _position_to_internal(self, pos: tuple[int, int, float]) -> int:
        node = int(pos[0])
        if node not in self.node_id_map:
            raise ValueError(f"Node id {node} not found in node mapping")
        return self.node_id_map[node]

    def _map_internal_action_to_env(self, internal_action: int, pos: tuple[int, int, float]) -> int:
        if internal_action not in self.reverse_node_map:
            raise ValueError(f"Invalid internal action {internal_action}")
        actual_action = int(self.reverse_node_map[internal_action])
        start, end, _ = pos
        if start == end and actual_action == start:
            return 0
        return actual_action

    def build_paths_by_exit(self, cutoff: int | None = None) -> tuple[dict[int, list[list[int]]], list[list[int]]]:
        if len(self.attacker_init) != 1:
            raise ValueError("NSGZero currently supports a single attacker for path generation")
        start_actual = self.reverse_node_map[self.attacker_init[0]]
        paths_by_exit: dict[int, list[list[int]]] = {}
        all_paths: list[list[int]] = []
        for exit_internal in self.exits:
            exit_actual = self.reverse_node_map[exit_internal]
            paths: list[list[int]] = []
            try:
                for path in nx.all_shortest_paths(self.nx_graph, source=start_actual, target=exit_actual):
                    if cutoff is not None and len(path) - 1 > cutoff:
                        continue
                    paths.append([self.node_id_map[node] for node in path])
            except nx.NetworkXNoPath:
                paths = []
            paths_by_exit[exit_internal] = paths
            all_paths.extend(paths)
        if not all_paths:
            all_paths = [[self.attacker_init[0]]]
        return paths_by_exit, all_paths


class GameState:
    def __init__(self, game: Game, defender_his: list[tuple[int, ...]], attacker_his: list[int]) -> None:
        self.game = game
        self.defender_his = defender_his
        self.attacker_his = attacker_his

        self.adjlist = game.adjlist
        self.time_horizon = game.time_horizon
        self.defender_init = game.defender_init
        self.attacker_init = game.attacker_init
        self.exits = game.exits
        self.num_defender = game.num_defender

        if len(defender_his) != len(attacker_his):
            raise ValueError("Defender and attacker history must be aligned")

    def is_end(self, attacker_his=None, defender_position=None):
        if attacker_his is None:
            attacker_his = self.attacker_his
        if defender_position is None:
            defender_position = self.defender_his[-1]
        if not attacker_his:
            return True
        attacker_pos = attacker_his[-1]
        if attacker_pos in defender_position:
            return True
        if attacker_pos in self.exits:
            return True
        if len(attacker_his) >= self.time_horizon + 1:
            return True
        return False

    def obs(self):
        defender_obs = (self.attacker_his, self.defender_his[-1])
        attacker_obs = (self.attacker_his, self.defender_his[0])
        return defender_obs, attacker_obs

    def reward(self, attacker_his=None, defender_position=None):
        if attacker_his is None:
            attacker_his = self.attacker_his
        if defender_position is None:
            defender_position = self.defender_his[-1]
        if not self.is_end(attacker_his, defender_position):
            return 0, 0
        attacker_pos = attacker_his[-1] if attacker_his else None
        if attacker_pos is not None and attacker_pos in defender_position:
            defender_reward = 1
        elif attacker_pos is not None and attacker_pos in self.exits:
            defender_reward = -1
        else:
            defender_reward = 1
        attacker_reward = -defender_reward
        return defender_reward, attacker_reward

    def legal_action(self, combinational=False, attacker_his=None, defender_position=None):
        if attacker_his is None:
            attacker_his = self.attacker_his
        if defender_position is None:
            defender_position = self.defender_his[-1]
        if not attacker_his:
            attacker_his = [self.attacker_init[0]]
        if defender_position is None:
            defender_position = self.defender_init[0]
        attacker_pos = attacker_his[-1]

        if self.is_end(attacker_his, defender_position):
            attacker_legal_act = [attacker_pos]
            defender_legal_act = [[pos] for pos in defender_position]
            if combinational:
                defender_legal_act = [tuple(defender_position)]
            return defender_legal_act, attacker_legal_act

        attacker_legal_act = self.adjlist[attacker_pos]
        defender_legal_act = [self.adjlist[pos] for pos in defender_position]
        if combinational:
            defender_legal_act = self._query_legal_defender_actions(defender_legal_act)
        return defender_legal_act, attacker_legal_act

    def _map_env_actions(self, legal_actions: list[list[int]], positions: list[tuple[int, int, float]]):
        mapped: list[list[int]] = []
        for actions, pos in zip(legal_actions, positions):
            start = int(pos[0])
            mapped_actions: list[int] = []
            for action in actions:
                actual = start if action == 0 else int(action)
                mapped_actions.append(self.game.node_id_map[actual])
            seen: set[int] = set()
            unique_actions: list[int] = []
            for action in mapped_actions:
                if action in seen:
                    continue
                seen.add(action)
                unique_actions.append(action)
            mapped.append(unique_actions)
        return mapped

    def _query_legal_defender_actions(self, defender_actions: list[list[int]]):
        return list(product(*defender_actions))
