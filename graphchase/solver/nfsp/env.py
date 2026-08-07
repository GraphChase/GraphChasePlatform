from __future__ import annotations

import copy
from itertools import product
from typing import Iterable

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import GameSettings


class Env:
    def __init__(self, settings: GameSettings, step_penalty: float = 0.0):
        self.settings = settings
        self.env = UNSGEnv(settings)
        self.time_horizon = int(settings.time_horizon)
        self.exits = list(settings.exit_nodes)
        self.step_penalty = float(step_penalty)
        self.num_attackers = len(settings.attacker_init)
        self.num_defenders = len(settings.defender_init)
        if self.num_attackers != 1:
            raise ValueError("NFSP wrapper currently supports a single attacker.")
        self.multi_defender = self.num_defenders > 1
        self.current_state: GameState | None = None

    def reset(self, defender_init=None, attacker_init=None):
        if defender_init is not None or attacker_init is not None:
            raise ValueError("NFSP wrapper does not support overriding initial positions.")
        obs, info = self.env.reset()
        attacker_nodes = self._positions_to_nodes(obs["attacker_state"])
        defender_nodes = self._positions_to_nodes(obs["defender_state"])
        defender_history = [tuple(defender_nodes)] if self.multi_defender else [defender_nodes[0]]
        attacker_history = [attacker_nodes[0]]
        self.current_state = GameState(self, defender_history, attacker_history, info, defender_nodes, attacker_nodes)
        return self.current_state

    def simu_step(self, defender_a, attacker_a):
        if self.current_state is None:
            raise ValueError("Environment must be reset before stepping.")
        if self.current_state.is_end():
            raise ValueError("Episode has ended; reset the environment before stepping again.")

        defender_action = self._resolve_action(defender_a, is_defender=True)
        attacker_action = self._resolve_action(attacker_a, is_defender=False)

        defender_actions = self._to_env_action_list(defender_action, self.current_state.defender_nodes)
        attacker_actions = self._to_env_action_list(attacker_action, self.current_state.attacker_nodes)

        obs, _, _, _, info = self.env.step(
            {"attacker_action": attacker_actions, "defender_action": defender_actions}
        )

        attacker_nodes = self._positions_to_nodes(obs["attacker_state"])
        defender_nodes = self._positions_to_nodes(obs["defender_state"])

        defender_history = copy.deepcopy(self.current_state.defender_history)
        attacker_history = copy.deepcopy(self.current_state.attacker_history)
        if self.multi_defender:
            defender_history.append(tuple(defender_nodes))
        else:
            defender_history.append(defender_nodes[0])
        attacker_history.append(attacker_nodes[0])

        self.current_state = GameState(self, defender_history, attacker_history, info, defender_nodes, attacker_nodes)
        return self.current_state

    def close(self):
        self.env.close()

    def _positions_to_nodes(self, positions: Iterable[Iterable[float]]) -> list[int]:
        nodes: list[int] = []
        for pos in positions:
            start, end, dist = int(pos[0]), int(pos[1]), float(pos[2])
            if start == end and dist == 0.0:
                nodes.append(start)
                continue
            edge_weight = self.settings.edge_weight(start, end)
            dist_to_end = dist
            dist_to_start = max(edge_weight - dist_to_end, 0.0)
            if dist_to_end <= dist_to_start:
                nodes.append(end)
            else:
                nodes.append(start)
        return nodes

    def _resolve_action(self, action, is_defender: bool):
        action_value, action_idx = self._split_action(action)
        if self.current_state is None:
            raise ValueError("No current state available.")
        legal_actions, per_agent_legal = self.current_state.legal_actions_for_mapping(is_defender)

        resolved = action_value
        if action_idx is not None:
            idx = int(action_idx)
            if idx < 0 or idx >= len(legal_actions):
                raise ValueError(f"Action index {idx} out of range for legal actions.")
            resolved = legal_actions[idx]

        if isinstance(resolved, int) and resolved not in legal_actions and 0 <= resolved < len(legal_actions):
            resolved = legal_actions[resolved]

        if is_defender and self.multi_defender:
            if isinstance(resolved, (list, tuple)):
                if not self._joint_action_is_legal(resolved, per_agent_legal):
                    resolved = self._map_branch_to_node_actions(per_agent_legal, resolved)
                if isinstance(resolved, list):
                    resolved = tuple(resolved)
            else:
                raise ValueError("Defender joint action must be a list or tuple.")
        else:
            if isinstance(resolved, (list, tuple)) and len(resolved) == 1:
                resolved = resolved[0]

        if resolved not in legal_actions:
            raise ValueError(f"Resolved action {resolved} is not legal for this state.")
        return resolved

    @staticmethod
    def _split_action(action):
        if isinstance(action, tuple):
            if len(action) == 0:
                raise ValueError("Empty action tuple is invalid.")
            if len(action) >= 2:
                return action[0], action[1]
            return action[0], None
        return action, None

    @staticmethod
    def _joint_action_is_legal(action: Iterable[int], per_agent_legal: list[list[int]]) -> bool:
        if len(per_agent_legal) != len(list(action)):
            return False
        return all(int(a) in legal for a, legal in zip(action, per_agent_legal))

    @staticmethod
    def _map_branch_to_node_actions(legal_actions: list[list[int]], branch_actions: Iterable[int]) -> list[int]:
        branch_list = list(branch_actions)
        if len(branch_list) != len(legal_actions):
            raise ValueError(
                f"Branch action length {len(branch_list)} does not match defender count {len(legal_actions)}"
            )
        node_actions: list[int] = []
        for idx, (branch_idx, acts) in enumerate(zip(branch_list, legal_actions)):
            if branch_idx < 0 or branch_idx >= len(acts):
                raise ValueError(f"Invalid branch index {branch_idx} for defender {idx} with {len(acts)} actions")
            node_actions.append(int(acts[branch_idx]))
        return node_actions

    @staticmethod
    def _to_env_action_list(action, current_nodes: list[int]) -> list[int]:
        if isinstance(action, (list, tuple)):
            actions = list(action)
        else:
            actions = [action]
        if len(actions) != len(current_nodes):
            raise ValueError("Action list length does not match number of agents.")
        env_actions: list[int] = []
        for act, current_node in zip(actions, current_nodes):
            act_val = int(act)
            if act_val == int(current_node):
                env_actions.append(0)
            else:
                env_actions.append(act_val)
        return env_actions


class GameState:
    def __init__(
        self,
        env: Env,
        defender_history: list,
        attacker_history: list[int],
        info: dict,
        defender_nodes: list[int],
        attacker_nodes: list[int],
    ):
        self.env = env
        self.time_horizon = env.time_horizon
        self.exits = env.exits
        self.step_penalty = env.step_penalty
        self.defender_history = defender_history
        self.attacker_history = attacker_history
        self.multi_defender = env.multi_defender
        if self.multi_defender:
            self.num_defender = env.num_defenders
        self._info = info
        self.defender_nodes = defender_nodes
        self.attacker_nodes = attacker_nodes
        assert len(self.defender_history) == len(self.attacker_history)
        assert len(self.defender_history) >= 1

    def is_end(self):
        if not self.multi_defender:
            if (len(self.defender_history) == self.time_horizon + 1) or (
                self.defender_history[-1] == self.attacker_history[-1]
            ) or (self.attacker_history[-1] in self.exits):
                return True
            return False
        if (len(self.defender_history) == self.time_horizon + 1) or (
            self.attacker_history[-1] in self.defender_history[-1]
        ) or (self.attacker_history[-1] in self.exits):
            return True
        return False

    def obs(self, play_id=None):
        defender_obs = (self.attacker_history, self.defender_history[-1])
        attacker_obs = (self.attacker_history, self.defender_history[0])
        if play_id is None:
            return defender_obs, attacker_obs
        if play_id == 0:
            return defender_obs
        if play_id == 1:
            return attacker_obs
        raise ValueError("invalid player_id.")

    def reward(self, play_id=None, is_evaluation=False):
        if is_evaluation:
            defender_reward = 0
            if not self.multi_defender:
                if self.defender_history[-1] == self.attacker_history[-1]:
                    defender_reward += 1
                elif self.attacker_history[-1] in self.exits:
                    defender_reward -= 1
                elif len(self.attacker_history) == self.time_horizon + 1:
                    defender_reward += 1
            else:
                if self.attacker_history[-1] in self.defender_history[-1]:
                    defender_reward += 1
                elif self.attacker_history[-1] in self.exits:
                    defender_reward -= 1
                elif len(self.attacker_history) == self.time_horizon + 1:
                    defender_reward += 1
            attacker_reward = -defender_reward
        else:
            attacker_reward = 0.0
            defender_reward = -self.step_penalty
            if not self.multi_defender:
                if self.defender_history[-1] == self.attacker_history[-1]:
                    defender_reward += 1
                    attacker_reward -= 1
                elif self.attacker_history[-1] in self.exits:
                    defender_reward -= 1
                    attacker_reward += 1
                elif len(self.attacker_history) == self.time_horizon + 1:
                    defender_reward += 1
                    attacker_reward -= 1
            else:
                if self.attacker_history[-1] in self.defender_history[-1]:
                    defender_reward += 1
                    attacker_reward -= 1
                elif self.attacker_history[-1] in self.exits:
                    defender_reward -= 1
                    attacker_reward += 1
                elif len(self.attacker_history) == self.time_horizon + 1:
                    defender_reward += 1
                    attacker_reward -= 1
                elif self.attacker_history[-1] in self.attacker_history[:-1]:
                    attacker_reward -= 0.5
        if play_id is None:
            return defender_reward, attacker_reward
        if play_id == 0:
            return defender_reward
        if play_id == 1:
            return attacker_reward
        raise ValueError("invalid player_id.")

    def legal_actions_for_mapping(self, is_defender: bool) -> tuple[list, list[list[int]]]:
        per_agent = self._per_agent_legal_actions(is_defender)
        if is_defender and self.multi_defender:
            joint_actions = list(product(*per_agent))
            return joint_actions, per_agent
        return per_agent[0], per_agent

    def legal_action(self, play_id=None):
        if play_id is None:
            defender_actions = self._defender_legal_action()
            attacker_actions = self._attacker_legal_action()
            return defender_actions, attacker_actions
        if play_id == 0:
            return self._defender_legal_action()
        if play_id == 1:
            return self._attacker_legal_action()
        raise ValueError("invalid player_id.")

    def _per_agent_legal_actions(self, is_defender: bool) -> list[list[int]]:
        if self.is_end():
            if is_defender:
                if self.multi_defender:
                    return [[0] for _ in range(self.num_defender)]
                return [[0]]
            return [[0]]
        raw_key = "defender_legal_action" if is_defender else "attacker_legal_action"
        raw_actions = self._info.get(raw_key, [])
        current_nodes = self.defender_nodes if is_defender else self.attacker_nodes
        mapped: list[list[int]] = []
        for acts, current_node in zip(raw_actions, current_nodes):
            action_list = []
            for act in acts:
                act_val = int(act)
                if act_val == 0:
                    action_list.append(int(current_node))
                else:
                    action_list.append(act_val)
            # Keep legal action ordering consistent with Maps.adjlist:
            # sorted(neighbors + [current_node]) with duplicates removed.
            mapped.append(sorted(set(action_list)))
        return mapped

    def _defender_legal_action(self):
        if self.is_end():
            if self.multi_defender:
                return [(0,) * self.num_defender]
            return [0]
        per_agent = self._per_agent_legal_actions(True)
        if self.multi_defender:
            return list(product(*per_agent))
        return per_agent[0]

    def _attacker_legal_action(self):
        if self.is_end():
            return [0]
        per_agent = self._per_agent_legal_actions(False)
        return per_agent[0]
