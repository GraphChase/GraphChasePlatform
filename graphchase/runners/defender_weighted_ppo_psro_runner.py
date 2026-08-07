from __future__ import annotations

from typing import Any
import copy

import torch

from graphchase.envs.vec_rollout_pool import VecRolloutPool
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner


class DefenderWeightedPpoPsroRunner(DefenderPretrainPsroRunner):
    """
    Weighted-graph defender runner for PPO-based PSRO.

    Observation encoding directly flattens the attacker and defender position
    triplets and appends the real remaining time:
    [attacker_state, defender_state, left_time].
    """

    def __init__(
        self,
        env_builder,
        agent,
        algorithm,
        metrics_logger=None,
        graph_embeddings=None,
        time_horizon: int | None = None,
        embedding_size: int = 1,
    ) -> None:
        super().__init__(
            env_builder=env_builder,
            agent=agent,
            algorithm=algorithm,
            metrics_logger=metrics_logger,
            graph_embeddings=None,
            time_horizon=time_horizon,
            embedding_size=1,
        )

    def encode_obs(self, obs: dict, cur_time: float | None = None, remaining_time: float | None = None) -> torch.Tensor:
        attacker_state = obs.get("attacker_state", [])
        defender_state = obs.get("defender_state", [])

        parts: list[float] = []
        for pos in attacker_state:
            parts.extend([float(pos[0]), float(pos[1]), float(pos[2])])
        for pos in defender_state:
            parts.extend([float(pos[0]), float(pos[1]), float(pos[2])])

        elapsed_time = 0.0
        if cur_time is not None:
            elapsed_time = float(cur_time)
        if remaining_time is not None:
            elapsed_time += float(1.0 - remaining_time)

        left_time = 0.0
        if self.time_horizon is not None and self.time_horizon > 0:
            left_time = max(float(self.time_horizon) - elapsed_time, 0.0) / float(self.time_horizon)
        parts.append(left_time)

        return torch.tensor(parts, dtype=torch.float32, device=self.device)

    def _defender_step_batch(
        self,
        obs_list: list[dict],
        info_list: list[dict],
        need_stats: bool = False,
    ) -> tuple[list[list[int]], list[dict[str, Any]] | None]:
        legal_actions_list = [info.get("defender_legal_action", []) for info in info_list]
        cur_times = [info.get("cur_time") for info in info_list]
        remaining_times = [info.get("remaining_time") for info in info_list]
        encoded_obs_list = [
            self.encode_obs(obs, cur_time, remaining_time)
            for obs, cur_time, remaining_time in zip(obs_list, cur_times, remaining_times)
        ]
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

        defender_actions_list = [
            self._map_branch_to_node_actions(legal_actions, branch_action)
            for legal_actions, branch_action in zip(legal_actions_list, branch_actions)
        ]

        if not need_stats:
            return defender_actions_list, None

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
        return defender_actions_list, pre_data_list

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        legal_actions = info.get("defender_legal_action", [])
        cur_time = info.get("cur_time")
        remaining_time = info.get("remaining_time")
        encoded_obs = self.encode_obs(obs, cur_time, remaining_time)
        legal_mask = self._legal_mask_from_info(legal_actions)
        action_map = self._full_action_map
        with torch.no_grad():
            branch_action, _, _, _, _ = self.agent.get_action_and_value(
                encoded_obs=encoded_obs,
                legal_mask=legal_mask,
                action_map=action_map,
            )
        return self._map_branch_to_node_actions(legal_actions, branch_action)

    def collect(
        self,
        attacker_runners: list[Any],
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
            attacker_runner = self._choose_attacker_runner(attacker_runners, meta_strategy)
            if hasattr(attacker_runner, "clone"):
                attacker_episode_runner = attacker_runner.clone()
            else:
                attacker_episode_runner = copy.deepcopy(attacker_runner)
            if hasattr(attacker_episode_runner, "reset"):
                attacker_episode_runner.reset()
            self.agent.reset_state()
            return attacker_episode_runner, self

        def on_step_batch(
            env_ids: list[int],
            obs_list: list[dict],
            info_list: list[dict],
            attacker_episode_runners: list[Any],
            __: list[Any],
        ):
            attacker_actions_list = [
                runner.policy_action(obs, info)
                for runner, obs, info in zip(attacker_episode_runners, obs_list, info_list)
            ]
            defender_actions_list, pre_data_list = self._defender_step_batch(
                obs_list, info_list, need_stats=True
            )
            assert pre_data_list is not None
            return attacker_actions_list, defender_actions_list, pre_data_list

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
                encoded_next = self.encode_obs(
                    next_obs,
                    next_info.get("cur_time"),
                    next_info.get("remaining_time"),
                )
                next_value = 0.0 if done else float(self.agent.value(encoded_next))
            pre_data.update(
                {
                    "reward": float(reward["defender"]),
                    "done": float(done),
                    "next_value": float(next_value),
                }
            )
            return pre_data

        _, episode_step_records = pool.run_episodes_batched(
            num_episodes=episodes,
            on_reset=on_reset,
            on_step_batch=on_step_batch,
            reward_key="defender",
            on_step_post=on_step_post,
        )
        if owns_pool:
            pool.close()
        batch.extend([ep for ep in episode_step_records if ep])
        return batch
