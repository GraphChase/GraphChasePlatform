from __future__ import annotations

from typing import Any
import copy
import itertools
import logging
import os

import numpy as np
import torch

from graphchase.interfaces.runner_base import RunnerBase
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.utils import convert2nodeidx_unweighted_graph
from graphchase.envs.vec_rollout_pool import VecRolloutPool

logger = logging.getLogger(__name__)


class AttackerPpoPsroRunner(RunnerBase):
    """
    Runner that trains an attacker policy via PPO as a best response to a fixed
    defender policy population under PSRO.
    """

    def __init__(
        self,
        env_builder,
        agent: PPOAgent,
        algorithm: PPOAlgorithm,
        metrics_logger=None,
        graph_embeddings=None,
        time_horizon: int | None = None,
        embedding_size: int = 1,
    ) -> None:
        super().__init__(
            env_builder=env_builder,
            agent=agent,
            algorithm=algorithm,
            br_solver=None,
            config=None,
            metrics_logger=metrics_logger,
        )
        self.device = agent.device

        if graph_embeddings is None:
            self.graph_embeddings = None
            self.embedding_size = 1
        else:
            self.graph_embeddings = graph_embeddings
            sample = next(iter(self.graph_embeddings.values()), None)
            self.embedding_size = int(len(sample)) if sample is not None else int(embedding_size)
        if self.embedding_size != 1:
            logger.info("Using graph embedding, dimension=%s", self.embedding_size)
        else:
            logger.info("Not using graph embedding")

        self.time_horizon = int(time_horizon) if time_horizon is not None else None
        self.max_branching, self.action_dim, env_time_horizon = self._compute_action_space()
        if self.time_horizon is None:
            self.time_horizon = env_time_horizon
        self._full_action_map, self._full_mask = self._build_full_action_map()

    def _embed_nodes(self, nodes: list[int]) -> list[float]:
        if not self.graph_embeddings:
            return [float(node) for node in nodes]
        embedded: list[float] = []
        for node in nodes:
            if node == 0 or node not in self.graph_embeddings:
                embedded.extend([0.0] * self.embedding_size)
            else:
                vector = self.graph_embeddings[node]
                embedded.extend([float(v) for v in vector])
        return embedded

    def encode_obs(self, obs: dict, cur_time: float | None = None) -> torch.Tensor:
        attacker_state = obs.get("attacker_state", [])
        defender_state = obs.get("defender_state", [])
        attacker_nodes = [
            convert2nodeidx_unweighted_graph((int(pos[0]), int(pos[1]), float(pos[2]))) for pos in attacker_state
        ]
        defender_nodes = [
            convert2nodeidx_unweighted_graph((int(pos[0]), int(pos[1]), float(pos[2]))) for pos in defender_state
        ]
        node_sequence = attacker_nodes + defender_nodes
        parts = self._embed_nodes(node_sequence)
        left_time = 0.0
        if cur_time is not None:
            if self.time_horizon is not None:
                left_time = float(max(int(self.time_horizon) - int(cur_time), 0))
            else:
                left_time = float(cur_time)
        parts.append(float(left_time) / self.time_horizon)
        if parts:
            vec = torch.tensor(parts, dtype=torch.float32)
        else:
            vec = torch.zeros(1, dtype=torch.float32)
        return vec.to(self.device)

    def _compute_action_space(self) -> tuple[int, int, int]:
        env = self.env_builder()
        max_branch = 0
        for node in env.graph.nodes():
            degree = env.graph.degree[node]
            max_branch = max(max_branch, degree + 1)
        action_dim = max_branch ** env.num_attackers
        time_horizon = int(env.time_horizon)
        env.close()
        return max_branch, action_dim, time_horizon

    def _build_full_action_map(self) -> tuple[list[list[int]], torch.Tensor]:
        base_env = self.env_builder()
        attacker_count = base_env.num_attackers
        base_env.close()
        choices = [list(range(self.max_branching)) for _ in range(attacker_count)]
        index_to_action: list[list[int]] = []
        for combo in itertools.product(*choices):
            index_to_action.append([int(a) for a in combo])
        full_mask = torch.ones(len(index_to_action), dtype=torch.bool)
        return index_to_action, full_mask

    def _trim_legal_actions(self, legal_actions: list[list[int]]) -> list[list[int]]:
        trimmed: list[list[int]] = []
        for acts in legal_actions:
            acts_list = list(acts)
            trimmed.append(acts_list[: self.max_branching])
        return trimmed

    def _legal_mask_from_info(self, legal_actions: list[list[int]]) -> torch.Tensor:
        trimmed_actions = self._trim_legal_actions(legal_actions)
        branch_masks: list[list[bool]] = []
        for acts in trimmed_actions:
            valid_len = len(acts)
            mask_row = [True] * valid_len + [False] * max(0, self.max_branching - valid_len)
            branch_masks.append(mask_row)
        mask: list[bool] = []
        for combo in itertools.product(*branch_masks):
            mask.append(all(combo))
        return torch.tensor(mask, dtype=torch.bool)

    def _map_branch_to_node_actions(self, legal_actions: list[list[int]], branch_actions: list[int]) -> list[int]:
        trimmed_actions = self._trim_legal_actions(legal_actions)
        if len(branch_actions) != len(trimmed_actions):
            raise ValueError(
                f"Branch action length {len(branch_actions)} does not match attacker count {len(trimmed_actions)}"
            )
        node_actions: list[int] = []
        for idx, (branch_idx, acts) in enumerate(zip(branch_actions, trimmed_actions)):
            if branch_idx < 0 or branch_idx >= len(acts):
                raise ValueError(f"Invalid branch index {branch_idx} for attacker {idx} with {len(acts)} legal actions")
            node_actions.append(int(acts[branch_idx]))
        return node_actions

    def _attacker_step_batch(
        self,
        obs_list: list[dict],
        info_list: list[dict],
        need_stats: bool = False,
    ) -> tuple[list[list[int]], list[dict[str, Any]] | None]:
        legal_actions_list = [info.get("attacker_legal_action", []) for info in info_list]
        cur_times = [info.get("cur_time") for info in info_list]
        encoded_obs_list = [self.encode_obs(obs, cur_time) for obs, cur_time in zip(obs_list, cur_times)]
        encoded_obs_batch = torch.stack(encoded_obs_list, dim=0)
        legal_masks_list = [self._legal_mask_from_info(acts) for acts in legal_actions_list]
        legal_mask_batch = torch.stack(legal_masks_list, dim=0).to(self.device)
        action_map = self._full_action_map

        with torch.no_grad():
            branch_actions, logprobs, _, values, action_indices = self.agent.get_action_and_value_batch(
                encoded_obs_batch=encoded_obs_batch,
                legal_mask_batch=legal_mask_batch,
                action_map=action_map,
            )

        attacker_actions_list = [
            self._map_branch_to_node_actions(legal_actions, branch_action)
            for legal_actions, branch_action in zip(legal_actions_list, branch_actions)
        ]

        if not need_stats:
            return attacker_actions_list, None

        action_indices_cpu = action_indices.detach().cpu().tolist()
        pre_data_list: list[dict[str, Any]] = []
        for encoded_obs, legal_mask, logprob, value, action_idx in zip(
            encoded_obs_batch, legal_mask_batch, logprobs, values, action_indices_cpu
        ):
            pre_data_list.append(
                {
                    "obs": encoded_obs.detach().cpu(),
                    "legal_mask": legal_mask.detach().cpu(),
                    "action_index": int(action_idx),
                    "logprob": float(logprob.detach().cpu().item()),
                    "value": float(value.detach().cpu().item()),
                }
            )
        return attacker_actions_list, pre_data_list

    def policy_action_batch(self, obs_list: list[dict], info_list: list[dict]) -> list[list[int]]:
        attacker_actions_list, _ = self._attacker_step_batch(obs_list, info_list, need_stats=False)
        return attacker_actions_list

    def _choose_defender_runner(self, defender_runners: list[Any], meta_strategy: Any | None) -> Any:
        if not defender_runners:
            raise ValueError("defender_runners must be a non-empty list")
        if meta_strategy is None:
            probs = np.ones(len(defender_runners), dtype=float) / float(len(defender_runners))
        else:
            probs = np.asarray(meta_strategy, dtype=float)
            if probs.shape[0] != len(defender_runners):
                raise ValueError("meta_strategy length must match defender_runners length")
            total = probs.sum()
            if total <= 0:
                probs = np.ones(len(defender_runners), dtype=float) / float(len(defender_runners))
            else:
                probs = probs / total
        idx = int(np.random.choice(len(defender_runners), p=probs))
        return defender_runners[idx]

    def collect(
        self,
        defender_runners: list[Any],
        meta_strategy: Any | None,
        episodes: int,
        vec_envs: int | None = None,
        pool: VecRolloutPool | None = None,
    ) -> list[dict[str, Any]]:
        batch: list[list[dict[str, Any]]] = []
        num_envs = min(episodes, int(vec_envs or 1))
        owns_pool = pool is None
        if pool is None:
            pool = VecRolloutPool(self.env_builder, num_envs=num_envs)

        def on_reset(_: int):
            defender_runner = self._choose_defender_runner(defender_runners, meta_strategy)
            if hasattr(defender_runner, "clone"):
                defender_episode_runner = defender_runner.clone()
            else:
                defender_episode_runner = copy.deepcopy(defender_runner)
            if hasattr(defender_episode_runner, "reset"):
                defender_episode_runner.reset()
            self.agent.reset_state()
            return self, defender_episode_runner

        def on_step_batch(
            env_ids: list[int],
            obs_list: list[dict],
            info_list: list[dict],
            __: list[Any],
            defender_episode_runners: list[Any],
        ):
            attacker_actions_list, pre_data_list = self._attacker_step_batch(obs_list, info_list, need_stats=True)
            assert pre_data_list is not None

            defender_actions_list: list[list[int] | None] = [None for _ in env_ids]
            groups: dict[int, list[int]] = {}
            for idx, defender_runner in enumerate(defender_episode_runners):
                groups.setdefault(id(defender_runner), []).append(idx)

            for group_indices in groups.values():
                defender_runner = defender_episode_runners[group_indices[0]]
                group_obs = [obs_list[i] for i in group_indices]
                group_info = [info_list[i] for i in group_indices]
                if hasattr(defender_runner, "policy_action_batch"):
                    actions_group = defender_runner.policy_action_batch(group_obs, group_info)
                else:
                    actions_group = [
                        defender_runner.policy_action(obs, info) for obs, info in zip(group_obs, group_info)
                    ]
                for local_idx, act in zip(group_indices, actions_group):
                    defender_actions_list[local_idx] = act

            if any(act is None for act in defender_actions_list):
                raise RuntimeError("Failed to compute defender actions for some environments.")
            return attacker_actions_list, [act for act in defender_actions_list], pre_data_list

        def on_step_post(
            _env_id: int,
            pre_data: dict[str, Any],
            next_obs: dict,
            reward: dict[str, float],
            done: bool,
            next_info: dict,
            __: Any,
            ___: Any,
        ):
            with torch.no_grad():
                encoded_next = self.encode_obs(next_obs, next_info.get("cur_time"))
                next_value = 0.0 if done else float(self.agent.value(encoded_next))
            pre_data.update(
                {
                    "reward": float(reward["attacker"]),
                    "done": float(done),
                    "next_value": float(next_value),
                }
            )
            return pre_data

        _, episode_step_records = pool.run_episodes_batched(
            num_episodes=episodes,
            on_reset=on_reset,
            on_step_batch=on_step_batch,
            reward_key="attacker",
            on_step_post=on_step_post,
        )
        if owns_pool:
            pool.close()
        batch.extend([ep for ep in episode_step_records if ep])
        return batch

    def _format_ppo_batch(self, batch: list[Any]) -> list[dict[str, Any]]:
        if not batch:
            return []
        if isinstance(batch[0], dict) and "obs" in batch[0]:
            return batch

        flat: list[dict[str, Any]] = []
        for episode in batch:
            if episode is None:
                continue
            steps = episode.get("steps") if isinstance(episode, dict) else episode
            if isinstance(episode, dict) and steps is None:
                steps = episode.get("episode_steps")
            if not steps:
                continue
            for step in steps:
                if step is None:
                    continue
                if not isinstance(step, dict):
                    raise TypeError("Expected step records to be dictionaries for PPO batch formatting.")
                flat.append(step)
        return flat

    def evaluate(self, opponent_runner, episodes: int = 2) -> dict[str, Any]:
        env = self.env_builder()
        wins = 0
        total = 0
        for _ in range(episodes):
            if hasattr(opponent_runner, "reset"):
                opponent_runner.reset()
            obs, info = env.reset()
            done = False
            while not done:
                attacker_actions = self.policy_action(obs, info)
                defender_actions = opponent_runner.policy_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(
                    {"attacker_action": attacker_actions, "defender_action": defender_actions}
                )
                done = terminated or truncated
                if done:
                    total += 1
                    if reward["attacker"] > reward["defender"]:
                        wins += 1
        env.close()
        return {"win_rate": wins / max(total, 1)}

    def run(
        self,
        defender_runners: list[Any],
        train_batches: int,
        episodes_per_batch: int,
        meta_strategy: Any | None = None,
        vec_envs: int | None = None,
        log_prefix: str | None = None,
        br_iter: int | str | None = None,
    ) -> dict[str, Any]:
        metrics = {}
        num_envs = min(episodes_per_batch, int(vec_envs or 1))
        pool = VecRolloutPool(self.env_builder, num_envs=num_envs)
        try:
            for batch_idx in range(train_batches):
                batch = self.collect(
                    defender_runners,
                    meta_strategy=meta_strategy,
                    episodes=episodes_per_batch,
                    vec_envs=vec_envs,
                    pool=pool,
                )
                ppo_batch = self._format_ppo_batch(batch)
                metrics = self.algorithm.update(ppo_batch, self.agent)
                if log_prefix and metrics:
                    prefixed = {f"{log_prefix}/{k}": v for k, v in metrics.items()}
                else:
                    prefixed = metrics
                self.log_metrics(
                    prefixed,
                    phase="train",
                    batch=batch_idx,
                    br_iter=br_iter,
                    br_step=batch_idx,
                    step=batch_idx,
                    algorithm=type(self.algorithm).__name__,
                    runner=type(self).__name__,
                )
        finally:
            pool.close()
        return metrics

    def compute_best_response(
        self,
        defender_runners: list[Any],
        role: str = "attacker",
        config: dict | None = None,
        meta_strategy: Any | None = None,
    ):
        if not config or "train_batches" not in config or "episodes_per_batch" not in config:
            raise KeyError("config must include 'train_batches' and 'episodes_per_batch'")
        train_batches = config["train_batches"]
        episodes_per_batch = config["episodes_per_batch"]
        br_iter = config.get("br_iter") if config else None
        vec_envs = int(config.get("vec_envs", 1)) if config else 1
        log_prefix = f"br_{br_iter}" if br_iter is not None else None
        metrics = self.run(
            defender_runners,
            train_batches=train_batches,
            episodes_per_batch=episodes_per_batch,
            meta_strategy=meta_strategy,
            vec_envs=vec_envs,
            log_prefix=log_prefix,
            br_iter=br_iter,
        )
        return {"agent": self.agent, "metrics": metrics}

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        legal_actions = info.get("attacker_legal_action", [])
        cur_time = info.get("cur_time")
        encoded_obs = self.encode_obs(obs, cur_time)
        legal_mask = self._legal_mask_from_info(legal_actions)
        action_map = self._full_action_map
        with torch.no_grad():
            branch_action, _, _, _, _ = self.agent.get_action_and_value(
                encoded_obs=encoded_obs, legal_mask=legal_mask, action_map=action_map
            )
        return self._map_branch_to_node_actions(legal_actions, branch_action)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)

    def load(self, path: str) -> None:
        self.agent.load(path)

    def reset(self) -> None:
        self.agent.reset_state()

    def clone(self, agent_state_dict=None) -> "AttackerPpoPsroRunner":
        agent_cls = type(self.agent)
        algo_cls = type(self.algorithm)

        agent = agent_cls(
            input_dim=self.agent.actor[0].in_features,
            action_dim=self.agent.actor[-1].out_features,
            hidden_dim=self.agent.actor[0].out_features,
            device=self.agent.device,
        )
        if agent_state_dict is not None:
            agent.load_state_dict(agent_state_dict)
        else:
            agent.load_state_dict(copy.deepcopy(self.agent.state_dict()))

        algo = algo_cls(
            learning_rate=self.algorithm.learning_rate,
            critic_learning_rate=self.algorithm.critic_learning_rate,
            gamma=self.algorithm.gamma,
            gae_lambda=self.algorithm.gae_lambda,
            clip_coef=self.algorithm.clip_coef,
            update_epochs=self.algorithm.update_epochs,
            minibatch_size=self.algorithm.minibatch_size,
            entropy_coef=self.algorithm.entropy_coef,
            vf_coef=self.algorithm.vf_coef,
            max_grad_norm=self.algorithm.max_grad_norm,
            device=self.algorithm.device,
        )

        runner_cls = type(self)
        return runner_cls(
            env_builder=self.env_builder,
            agent=agent,
            algorithm=algo,
            metrics_logger=self.metrics_logger,
            graph_embeddings=self.graph_embeddings,
            time_horizon=self.time_horizon,
            embedding_size=self.embedding_size,
        )
