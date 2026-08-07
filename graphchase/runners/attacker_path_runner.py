from __future__ import annotations

from typing import Any
import os
import pickle
import networkx as nx
import numpy as np
import copy
import logging

from graphchase.interfaces.runner_base import RunnerBase
from graphchase.interfaces.agent_base import AgentBase
from graphchase.algorithms.null_algorithm import NullAlgorithm
from graphchase.agents.attacker_path_agent import PathAgent
from graphchase.envs.vec_rollout_pool import VecRolloutPool

logger = logging.getLogger(__name__)

class AttackerPathRunner(RunnerBase):
    """
    Runner that searches attacker paths as best responses against a defender runner.
    """

    def __init__(
        self,
        env_builder,
        action_type: str = "exit_node",
        strategy_type: str = "mix",
        algorithm=None,
        max_path_length: int | None = None,
        attacker_path_type: str = "shortest",
        metrics_logger=None,
    ) -> None:
        if action_type not in {"exit_node", "all_path"}:
            raise ValueError("action_type must be 'exit_node' or 'all_path'")
        if strategy_type not in {"mix", "greedy"}:
            raise ValueError("strategy_type must be 'mix' or 'greedy'")
        if attacker_path_type not in {"shortest", "simple"}:
            raise ValueError("attacker_path_type must be 'shortest' or 'simple'")
        dummy_agent = PathAgent(path=[], num_attackers=1)
        algo = algorithm if algorithm is not None else NullAlgorithm()
        super().__init__(
            env_builder=env_builder,
            agent=dummy_agent,
            algorithm=algo,
            br_solver=None,
            config=None,
            metrics_logger=metrics_logger,
        )
        self.action_type = action_type
        self.strategy_type = strategy_type
        self.max_path_length = max_path_length
        self.attacker_path_type = attacker_path_type
        self._cached_paths: list[list[int]] | None = None
        self._paths_by_exit: dict[int, list[list[int]]] = {}
        self.exit_nodes_ordered: list[int] = []
        self.path_scores: dict[tuple[int, ...], float] = {}
        self.action_scores: list[float] = []

        tmp_env = self.env_builder()
        self.num_attackers = tmp_env.num_attackers
        self._cached_paths, self._paths_by_exit, self.exit_nodes_ordered = self._candidate_paths(tmp_env)
        if self._cached_paths:
            for p in self._cached_paths:
                self.path_scores[tuple(p)] = 1.0
            if self.action_type == "exit_node":
                self.action_scores = [1.0 for _ in self.exit_nodes_ordered]
            else:
                self.action_scores = [1.0 for _ in self._cached_paths]
        tmp_env.close()

    def collect(self, **kwargs) -> dict[str, Any]:
        return {}

    def evaluate(self, **kwargs) -> dict[str, Any]:
        return {}

    def run(self, **kwargs) -> None:
        return None

    def _candidate_paths(self, env) -> tuple[list[list[int]], dict[int, list[list[int]]], list[int]]:
        env.reset()
        graph: nx.Graph = env.graph
        start_node = env.initial_nodes[0] if env.initial_nodes else 0
        exit_nodes = sorted(list(env.exit_nodes))
        cutoff = self.max_path_length or env.time_horizon
        paths: list[list[int]] = []
        paths_by_exit: dict[int, list[list[int]]] = {e: [] for e in exit_nodes}
        for exit_node in exit_nodes:
            try:
                if self.attacker_path_type == "simple":
                    candidate_paths = nx.all_simple_paths(
                        graph,
                        source=start_node,
                        target=exit_node,
                        cutoff=cutoff,
                    )
                else:
                    candidate_paths = nx.all_shortest_paths(graph, source=start_node, target=exit_node)
                for path in candidate_paths:
                    paths.append(list(path))
                    paths_by_exit[exit_node].append(list(path))
            except nx.NetworkXNoPath:
                continue
        if not paths:
            paths.append([start_node])
            for exit_node in exit_nodes:
                paths_by_exit.setdefault(exit_node, [])
        return paths, paths_by_exit, exit_nodes

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        assert self._cached_paths is not None and len(self._cached_paths) > 0, "cached paths must be initialized"
        assert self.path_scores, "path scores must be initialized"
        if not isinstance(self.agent, PathAgent) or not getattr(self.agent, "path", None):
            path = self._sample_path_by_action_scores()
            self.agent = PathAgent(path=path, num_attackers=self.num_attackers)
        actions, _ = self.agent.act(obs)
        return actions

    def _probs_from_scores(self, scores: list[float]) -> np.ndarray:
        probs = np.asarray(scores, dtype=float)
        if probs.ndim != 1 or probs.size == 0:
            probs = np.ones(1, dtype=float)
        if np.any(probs < 0):
            probs = probs - probs.min() + 1e-6
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = np.ones_like(probs, dtype=float)
            total = probs.sum()
        return probs / total

    def _sample_path_by_action_scores(self) -> list[int]:
        if self.action_type == "exit_node":
            if self.strategy_type == "greedy":
                exit_idx = int(np.argmax(self.action_scores))
            else:
                probs = self._probs_from_scores(self.action_scores)
                exit_idx = int(np.random.choice(len(self.exit_nodes_ordered), p=probs))
            exit_node = self.exit_nodes_ordered[exit_idx]
            candidate_paths = self._paths_by_exit.get(exit_node, [])
            if not candidate_paths:
                logger.info(f"No available paths for exit node {exit_node}, falling back to cached paths")
                candidate_paths = self._cached_paths
            path = candidate_paths[int(np.random.choice(len(candidate_paths)))]
        else:
            if self.strategy_type == "greedy":
                path_idx = int(np.argmax(self.action_scores))
            else:
                probs = self._probs_from_scores(self.action_scores)
                path_idx = int(np.random.choice(len(self._cached_paths), p=probs))
            path = self._cached_paths[path_idx]
        return path

    def random_policy(self, low: float = 1.0, high: float = 10.0) -> list[float]:
        if self.action_type == "exit_node":
            action_count = len(self.exit_nodes_ordered)
        else:
            action_count = len(self._cached_paths or [])
        if action_count == 0:
            raise ValueError("No attacker actions available for random policy generation")

        scores = np.random.uniform(low, high, size=action_count)
        total = float(scores.sum())
        if not np.isfinite(total) or total <= 0:
            scores = np.ones(action_count, dtype=float)
            total = float(scores.sum())
        probs = scores / total
        self.action_scores = [float(score) for score in probs]
        self.path_scores = {}
        if self.action_type == "exit_node":
            for exit_node, score in zip(self.exit_nodes_ordered, self.action_scores):
                for path in self._paths_by_exit.get(exit_node, []):
                    self.path_scores[tuple(path)] = float(score)
        else:
            if self._cached_paths is None:
                raise AssertionError("cached paths must be initialized before random policy generation")
            for path, score in zip(self._cached_paths, self.action_scores):
                self.path_scores[tuple(path)] = float(score)
        return self.action_scores

    def compute_best_response(
        self,
        opponent_runners: Any,
        role: str = "attacker",
        config: dict | None = None,
        meta_strategy: Any | None = None,
    ) -> AgentBase:
        assert self._cached_paths is not None and len(self._cached_paths) > 0, "cached paths must be initialized"
        if config is None or "rollouts_per_path" not in config:
            raise KeyError("rollouts_per_path must be provided in config for compute_best_response.")
        rollouts = config["rollouts_per_path"]
        vec_envs = int(config.get("vec_envs", 1)) if config else 1
        reward_mode = config.get("reward_mode", "utility") if config else "utility"
        if isinstance(opponent_runners, (list, tuple)):
            opponents = list(opponent_runners)
        else:
            opponents = [opponent_runners]
        if not opponents:
            raise ValueError("opponent_runners must be a non-empty list or tuple")
        if meta_strategy is None:
            probs = np.ones(len(opponents), dtype=float) / float(len(opponents))
        else:
            probs = np.asarray(meta_strategy, dtype=float)
            if probs.shape[0] != len(opponents):
                raise ValueError("meta_strategy length must match opponent_runners length")
            if np.any(probs < 0):
                raise ValueError("meta_strategy must be non-negative")
            if not np.isclose(probs.sum(), 1.0):
                raise ValueError("meta_strategy must sum to 1.0")

        pool = VecRolloutPool(self.env_builder, num_envs=vec_envs)

        if self.action_type == "exit_node":
            exit_scores: dict[int, float] = {}
            best_reward = -1e9
            best_exit = None
            for exit_node in self.exit_nodes_ordered:
                exit_paths = self._paths_by_exit.get(exit_node, [])
                if not exit_paths:
                    continue
                avg_reward = self._simulate_exit_vec(pool, exit_paths, opponents, probs, rollouts, reward_mode)
                exit_scores[exit_node] = avg_reward
                for p in exit_paths:
                    self.path_scores[tuple(p)] = avg_reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_exit = exit_node
            if best_exit is None:
                best_agent = PathAgent(path=self._cached_paths[0], num_attackers=self.num_attackers)
            else:
                best_paths = self._paths_by_exit.get(best_exit, [])
                chosen_path = best_paths[0] if best_paths else self._cached_paths[0]
                best_agent = PathAgent(path=chosen_path, num_attackers=self.num_attackers)
            self.action_scores = [exit_scores.get(e, 0.0) for e in self.exit_nodes_ordered]
            self.set_agent(best_agent)
        else:
            paths = self._cached_paths
            best_reward = -1e9
            best_agent = None
            for path in paths:
                avg_reward = self._simulate_path_vec(pool, path, opponents, probs, rollouts, reward_mode)
                self.path_scores[tuple(path)] = avg_reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_agent = PathAgent(path=path, num_attackers=self.num_attackers)
            if best_agent is None:
                best_agent = PathAgent(path=paths[0], num_attackers=self.num_attackers)
            self.set_agent(best_agent)
            self.action_scores = self._build_action_scores()
        pool.close()
        return best_agent

    def _simulate_path_vec(
        self,
        pool: VecRolloutPool,
        path: list[int],
        opponent_runners: list[Any],
        probs: np.ndarray,
        rollouts: int,
        reward_mode: str = "utility",
    ) -> float:
        def on_reset(_: int):
            opp_idx = int(np.random.choice(len(opponent_runners), p=probs))
            opponent_runner = opponent_runners[opp_idx]
            fixed_agent = PathAgent(path=path, num_attackers=self.num_attackers)
            return fixed_agent, opponent_runner

        def on_step_batch(
            env_ids: list[int],
            obs_list: list[dict],
            info_list: list[dict],
            attacker_policies: list[PathAgent],
            defender_policies: list[Any],
        ):
            attacker_actions_list = [
                attacker_policy.act(obs)[0] for attacker_policy, obs in zip(attacker_policies, obs_list)
            ]
            defender_actions_list: list[list[int] | None] = [None for _ in env_ids]

            groups: dict[int, list[int]] = {}
            for idx, defender_policy in enumerate(defender_policies):
                groups.setdefault(id(defender_policy), []).append(idx)

            for group_indices in groups.values():
                defender_policy = defender_policies[group_indices[0]]
                group_obs = [obs_list[i] for i in group_indices]
                group_info = [info_list[i] for i in group_indices]
                if hasattr(defender_policy, "policy_action_batch"):
                    actions_group = defender_policy.policy_action_batch(group_obs, group_info)
                else:
                    actions_group = [
                        defender_policy.policy_action(o, inf) for o, inf in zip(group_obs, group_info)
                    ]
                for local_idx, act in zip(group_indices, actions_group):
                    defender_actions_list[local_idx] = act

            if any(act is None for act in defender_actions_list):
                raise RuntimeError("Failed to compute defender actions for some environments.")
            return attacker_actions_list, [act for act in defender_actions_list], [None for _ in env_ids]

        rewards, _ = pool.run_episodes_batched(
            num_episodes=rollouts,
            on_reset=on_reset,
            on_step_batch=on_step_batch,
            reward_key="attacker",
            reward_mode=reward_mode,
        )
        return float(np.mean(rewards)) if rewards else 0.0

    def _simulate_exit_vec(
        self,
        pool: VecRolloutPool,
        exit_paths: list[list[int]],
        opponent_runners: list[Any],
        probs: np.ndarray,
        rollouts: int,
        reward_mode: str = "utility",
    ) -> float:
        def on_reset(_: int):
            opp_idx = int(np.random.choice(len(opponent_runners), p=probs))
            opponent_runner = opponent_runners[opp_idx]
            chosen_path = exit_paths[int(np.random.choice(len(exit_paths)))]
            fixed_agent = PathAgent(path=chosen_path, num_attackers=self.num_attackers)
            return fixed_agent, opponent_runner

        def on_step_batch(
            env_ids: list[int],
            obs_list: list[dict],
            info_list: list[dict],
            attacker_policies: list[PathAgent],
            defender_policies: list[Any],
        ):
            attacker_actions_list = [
                attacker_policy.act(obs)[0] for attacker_policy, obs in zip(attacker_policies, obs_list)
            ]
            defender_actions_list: list[list[int] | None] = [None for _ in env_ids]

            groups: dict[int, list[int]] = {}
            for idx, defender_policy in enumerate(defender_policies):
                groups.setdefault(id(defender_policy), []).append(idx)

            for group_indices in groups.values():
                defender_policy = defender_policies[group_indices[0]]
                group_obs = [obs_list[i] for i in group_indices]
                group_info = [info_list[i] for i in group_indices]
                if hasattr(defender_policy, "policy_action_batch"):
                    actions_group = defender_policy.policy_action_batch(group_obs, group_info)
                else:
                    actions_group = [
                        defender_policy.policy_action(o, inf) for o, inf in zip(group_obs, group_info)
                    ]
                for local_idx, act in zip(group_indices, actions_group):
                    defender_actions_list[local_idx] = act

            if any(act is None for act in defender_actions_list):
                raise RuntimeError("Failed to compute defender actions for some environments.")
            return attacker_actions_list, [act for act in defender_actions_list], [None for _ in env_ids]

        rewards, _ = pool.run_episodes_batched(
            num_episodes=rollouts,
            on_reset=on_reset,
            on_step_batch=on_step_batch,
            reward_key="attacker",
            reward_mode=reward_mode,
        )
        return float(np.mean(rewards)) if rewards else 0.0

    def _simulate_path(self, env, path: list[int], opponent_runners: list[Any], probs: np.ndarray, rollouts: int) -> float:
        total_reward = 0.0
        for _ in range(rollouts):
            opp_idx = int(np.random.choice(len(opponent_runners), p=probs))
            opponent_runner = opponent_runners[opp_idx]
            agent = PathAgent(path=path, num_attackers=env.num_attackers)
            obs, info = env.reset()
            done = False
            agent.reset_state()
            while not done:
                attacker_actions = agent.act(obs)[0]
                defender_actions = opponent_runner.policy_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(
                    {"attacker_action": attacker_actions, "defender_action": defender_actions}
                )
                done = terminated or truncated
                total_reward += float(reward["attacker"])
        return total_reward / float(rollouts)

    def _simulate_exit(
        self,
        env,
        exit_paths: list[list[int]],
        opponent_runners: list[Any],
        probs: np.ndarray,
        rollouts: int,
    ) -> float:
        total_reward = 0.0
        for _ in range(rollouts):
            opp_idx = int(np.random.choice(len(opponent_runners), p=probs))
            opponent_runner = opponent_runners[opp_idx]
            path = exit_paths[int(np.random.choice(len(exit_paths)))]
            agent = PathAgent(path=path, num_attackers=env.num_attackers)
            obs, info = env.reset()
            done = False
            agent.reset_state()
            while not done:
                attacker_actions = agent.act(obs)[0]
                defender_actions = opponent_runner.policy_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(
                    {"attacker_action": attacker_actions, "defender_action": defender_actions}
                )
                done = terminated or truncated
        return total_reward / float(rollouts)

    def _build_action_scores(self) -> list[float]:
        if self.action_type == "exit_node":
            scores: list[float] = []
            for exit_node in self.exit_nodes_ordered:
                paths = self._paths_by_exit.get(exit_node, [])
                if not paths:
                    scores.append(0.0)
                    continue
                vals = [self.path_scores.get(tuple(p), 0.0) for p in paths]
                scores.append(float(np.mean(vals)))
            return scores
        return [float(self.path_scores.get(tuple(p), 0.0)) for p in self._cached_paths]

    def compute_defender_wcu(self, reward_mode: str = "utility") -> float | None:
        if reward_mode not in {"utility", "win_rate"}:
            raise ValueError("reward_mode must be 'utility' or 'win_rate'")
        if not self.action_scores:
            return None
        max_success = max(float(score) for score in self.action_scores)
        if reward_mode == "utility":
            return -max_success
        return 1 - max_success

    def reset(self) -> None:
        if hasattr(self.agent, "reset_state"):
            self.agent.reset_state()
        if hasattr(self.agent, "path"):
            try:
                self.agent.path = []
            except Exception:
                self.agent.path = None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        agent_path = getattr(self.agent, "path", None) if isinstance(self.agent, PathAgent) else None
        data = {
            "cached_paths": self._cached_paths,
            "path_scores": self.path_scores,
            "action_scores": list(self.action_scores),
            "agent_path": agent_path,
            "num_attackers": self.num_attackers,
        }
        with open(path, "wb") as fp:
            pickle.dump(data, fp)

    def load(self, path: str) -> None:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        saved_paths = data.get("cached_paths")
        current_paths = self._cached_paths
        if current_paths is None:
            raise AssertionError("cached paths must be initialized before loading attacker strategy")
        if saved_paths is None:
            raise AssertionError("saved attacker strategy missing cached_paths")
        assert list(current_paths) == list(saved_paths), "cached_paths mismatch when loading attacker strategy"
        saved_scores = data.get("path_scores", {})
        self.path_scores = {}
        for k, v in saved_scores.items():
            key = tuple(k) if isinstance(k, (list, tuple)) else k
            self.path_scores[key] = float(v)
        agent_path = data.get("agent_path")
        if agent_path:
            self.agent = PathAgent(path=list(agent_path), num_attackers=data.get("num_attackers", self.num_attackers))
            self.agent.reset_state()
        saved_action_scores = data.get("action_scores")
        if saved_action_scores is None:
            raise AssertionError("saved attacker strategy missing action_scores")
        self.action_scores = [float(x) for x in saved_action_scores]

    def clone(self, agent=None) -> "AttackerPathRunner":
        """
        Create a lightweight per-episode copy.

        Rollouts require each environment/episode to have an independent attacker
        agent state (e.g., PathAgent cursor). Cached paths can be shared, but
        per-run score tables may be mutated during BR computation, so we copy
        those to keep clones isolated.
        """
        runner = copy.copy(self)
        runner.path_scores = copy.deepcopy(self.path_scores)
        runner.action_scores = copy.deepcopy(self.action_scores)
        if agent is not None:
            runner.set_agent(agent)
        else:
            runner.set_agent(PathAgent(path=[], num_attackers=self.num_attackers))
        return runner
