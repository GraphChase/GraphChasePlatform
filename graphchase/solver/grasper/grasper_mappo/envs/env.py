from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.solver.grasper.grasper_game import GrasperGame


@dataclass
class GrasperTimeStep:
    rewards: list[float]
    observations: dict[str, Any]
    done: bool

    def last(self) -> bool:
        return self.done


class RL_Env:
    def __init__(
        self,
        game: GrasperGame,
        attacker_strategy=None,
        attacker_strategy_type: str = "mix",
        action_type: str = "all_path",
    ) -> None:
        self.game = game
        self.defender_num = self.game._defender_num
        self.agent_num = self.game.agent_num
        self.obs_dim = self.game.observation_size
        self.share_obs_dim = self.game.defender_state_representation_size
        self.action_dim = self.game.defender_mix_action
        self.exit_node = list(self.game._graph.exit_node)
        self.time_horizon = self.game._time_horizon
        self.attacker_path = None
        self.attacker_strategy_type = attacker_strategy_type
        self.action_type = action_type
        self.attacker_strategy = attacker_strategy
        self.attacker_runner_pool = None
        self.attacker_meta_strategy = None
        self.attacker_runner = None
        self._attacker_cursor = 0
        self._should_reset = True
        self._last_obs: dict[str, Any] = {}
        self._last_info: dict[str, Any] = {}

        self._env = UNSGEnv(self.game.settings)

        self._paths_by_exit, self._all_paths = self._prepare_attacker_paths()
        self.action_number = len(self._all_paths) if action_type == "all_path" else len(self.exit_node)

        if self.attacker_strategy is None:
            self.initialize_attacker_strategy()

    def close(self) -> None:
        self._env.close()

    def reset_by_game(self, game: GrasperGame) -> None:
        self.game = game
        self.defender_num = self.game._defender_num
        self.agent_num = self.game.agent_num
        self.obs_dim = self.game.observation_size
        self.share_obs_dim = self.game.defender_state_representation_size
        self.action_dim = self.game.defender_mix_action
        self.exit_node = list(self.game._graph.exit_node)
        self.time_horizon = self.game._time_horizon
        self._env.close()
        self._env = UNSGEnv(self.game.settings)
        self._paths_by_exit, self._all_paths = self._prepare_attacker_paths()
        self.action_number = len(self._all_paths) if self.action_type == "all_path" else len(self.exit_node)
        self._should_reset = True

    def initialize_attacker_strategy(self, attacker_strategy=None) -> None:
        if attacker_strategy is not None:
            self.attacker_strategy = attacker_strategy
            return
        if self.attacker_strategy_type == "mix":
            if self.action_type == "all_path":
                scores = np.array([random.uniform(1, 10) for _ in range(self.action_number)])
            else:
                scores = np.array(
                    [
                        random.uniform(1, 10) if len(self._paths_by_exit.get(node, [])) > 0 else 0
                        for node in self.exit_node
                    ]
                )
            self.attacker_strategy = scores / scores.sum() if scores.sum() > 0 else np.ones(self.action_number) / self.action_number
        else:
            scores = np.zeros(self.action_number)
            if self.action_type == "all_path":
                idx = np.random.randint(0, self.action_number)
            else:
                candidate_idx = [i for i, node in enumerate(self.exit_node) if len(self._paths_by_exit.get(node, [])) > 0]
                idx = int(np.random.choice(candidate_idx)) if candidate_idx else 0
            scores[idx] = 1.0
            self.attacker_strategy = scores

    def set_attacker_runner_pool(self, attacker_runners, meta_strategy=None) -> None:
        self.attacker_runner_pool = attacker_runners
        self.attacker_meta_strategy = meta_strategy

    def _prepare_attacker_paths(self) -> tuple[dict[int, list[list[int]]], list[list[int]]]:
        paths_by_exit = {node: [] for node in self.exit_node}
        all_paths = []
        attacker_paths = self.game.attacker_path
        if isinstance(attacker_paths, dict):
            for exit_node, paths in attacker_paths.items():
                paths_by_exit[exit_node] = [list(path) for path in paths]
                all_paths.extend(paths_by_exit[exit_node])
        else:
            all_paths = [list(path) for path in attacker_paths]
            for path in all_paths:
                if path:
                    paths_by_exit.setdefault(path[-1], []).append(path)
        return paths_by_exit, all_paths

    def _choose_attacker_runner(self):
        if not self.attacker_runner_pool:
            return None
        runners = list(self.attacker_runner_pool)
        if not runners:
            return None
        if self.attacker_meta_strategy is None:
            probs = np.ones(len(runners), dtype=float) / float(len(runners))
        else:
            probs = np.asarray(self.attacker_meta_strategy, dtype=float)
            if probs.sum() <= 0:
                probs = np.ones(len(runners), dtype=float) / float(len(runners))
            else:
                probs = probs / probs.sum()
        idx = int(np.random.choice(len(runners), p=probs))
        runner = runners[idx]
        if hasattr(runner, "clone"):
            runner = runner.clone()
        if hasattr(runner, "reset"):
            runner.reset()
        return runner

    def _select_attacker_path(self) -> None:
        if self.action_type == "all_path":
            if not self._all_paths:
                self.attacker_path = []
                return
            action_index = int(np.random.choice(range(len(self._all_paths)), p=self.attacker_strategy))
            self.attacker_path = self._all_paths[action_index]
        else:
            action_index = int(np.random.choice(range(self.action_number), p=self.attacker_strategy))
            exit_node = self.exit_node[action_index]
            candidate_paths = self._paths_by_exit.get(exit_node, [])
            if not candidate_paths:
                self.attacker_path = []
                return
            idx = int(np.random.choice(range(len(candidate_paths))))
            self.attacker_path = candidate_paths[idx]
        self._attacker_cursor = 0

    def _position_to_node(self, position: tuple[int, int, float]) -> int:
        start, end, dist = position
        if start == end or dist <= 0:
            return int(end)
        return int(start)

    def _build_obs(self, obs: dict, info: dict) -> tuple[np.ndarray, np.ndarray]:
        attacker_states = obs.get("attacker_state", [])
        defender_states = obs.get("defender_state", [])
        attacker_node = 0
        if attacker_states:
            attacker_node = self._position_to_node(tuple(attacker_states[0]))
        defender_nodes = [self._position_to_node(tuple(pos)) for pos in defender_states]
        cur_time = int(info.get("cur_time", 0))
        shared_state = [attacker_node] + defender_nodes + [cur_time]
        shared_obs = np.array([shared_state for _ in range(self.defender_num)])
        obs_list = []
        for idx, node in enumerate(defender_nodes):
            obs_list.append([attacker_node, node, cur_time, idx])
        return shared_obs, np.array(obs_list)

    def _map_actions(self, action_indices: list[int], legal_actions: list[list[int]]) -> list[int]:
        mapped = []
        for idx, legal in zip(action_indices, legal_actions):
            if idx < 0 or idx >= len(legal):
                mapped.append(0)
            else:
                mapped.append(int(legal[idx]))
        return mapped

    def _attacker_actions(self, obs: dict, info: dict) -> list[int]:
        if self.attacker_runner is not None:
            actions = self.attacker_runner.policy_action(obs, info)
            if not isinstance(actions, list):
                return [int(actions)]
            return [int(a) for a in actions]
        if not self.attacker_path:
            return [0 for _ in range(self._env.num_attackers)]
        next_idx = self._attacker_cursor + 1
        if next_idx < len(self.attacker_path):
            next_node = int(self.attacker_path[next_idx])
            self._attacker_cursor = next_idx
            return [next_node] + [0 for _ in range(self._env.num_attackers - 1)]
        return [0 for _ in range(self._env.num_attackers)]

    def step(self, action):
        if self._should_reset:
            done = True
            reward = [0.0]
            state = [0 for _ in range(self.share_obs_dim)]
            sub_agent_obs = [[0 for _ in range(self.obs_dim)] for _ in range(self.defender_num)]
            time_step = GrasperTimeStep(rewards=[0.0, 0.0], observations={}, done=True)
            return [
                np.array(state).repeat(self.defender_num).reshape(self.defender_num, -1),
                np.array(sub_agent_obs),
                np.array([reward for _ in range(self.defender_num)]),
                np.array([[done] for _ in range(self.defender_num)]),
                time_step,
            ]

        defender_actions = [int(a) for a in action]
        legal_actions = self._last_info.get("defender_legal_action", [])
        if not legal_actions:
            legal_actions = self._env._legal_actions(is_attacker=False)
        defender_node_actions = self._map_actions(defender_actions, legal_actions)
        attacker_actions = self._attacker_actions(self._last_obs, self._last_info)

        obs, reward, terminated, truncated, info = self._env.step(
            {"attacker_action": attacker_actions, "defender_action": defender_node_actions}
        )
        done = bool(terminated or truncated)
        shared_obs, sub_agent_obs = self._build_obs(obs, info)
        self._last_obs = obs
        self._last_info = info
        self._should_reset = done

        time_step = GrasperTimeStep(rewards=[reward["attacker"], reward["defender"]], observations=obs, done=done)
        sub_agent_reward = [[reward["defender"]] for _ in range(self.defender_num)]
        sub_agent_done = [[done] for _ in range(self.defender_num)]
        return [
            shared_obs,
            sub_agent_obs,
            np.array(sub_agent_reward),
            np.array(sub_agent_done),
            time_step,
        ]

    def reset(self):
        if self.attacker_runner_pool is not None:
            self.attacker_runner = self._choose_attacker_runner()
        else:
            self.attacker_runner = None
            self._select_attacker_path()

        obs, info = self._env.reset()
        self._last_obs = obs
        self._last_info = info
        self._should_reset = False
        shared_obs, sub_agent_obs = self._build_obs(obs, info)
        time_step = GrasperTimeStep(rewards=[0.0, 0.0], observations=obs, done=False)
        return [shared_obs, sub_agent_obs, time_step]
