from __future__ import annotations

from typing import Any, Callable
import numpy as np


class VecRolloutPool:
    """
    Synchronous vectorized rollout manager.

    Creates multiple environment instances in a single process and interleaves
    episodes across them. Each environment runs one episode at a time; when an
    episode terminates, the env is immediately reset and a new opponent/policy
    pair is sampled via `on_reset`.

    This enables B1-style rollout parallelism (episode-level opponent sampling)
    without multiprocessing. It is also reusable for both attacker BR
    simulation and defender PPO data collection.
    """

    def __init__(self, env_builder: Callable[[], Any], num_envs: int = 1, seed: int | None = None) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.env_builder = env_builder
        self.num_envs = int(num_envs)
        self.envs = [self.env_builder() for _ in range(self.num_envs)]
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                continue

    def run_episodes(
        self,
        num_episodes: int,
        on_reset: Callable[[int], tuple[Any, Any]],
        on_step: Callable[[int, Any, Any, Any, Any], tuple[list[int], list[int], Any | None]],
        reward_key: str = "attacker",
        on_step_post: Callable[[int, Any, Any, Any, bool, Any, Any, Any], Any | None] | None = None,
    ) -> tuple[list[float], list[Any]]:
        """
        Run `num_episodes` episodes across the pool.

        - `on_reset(env_id)` should return (attacker_policy, defender_policy)
          for the next episode on that env (B1 sampling happens here).
        - `on_step(env_id, obs, info, attacker_policy, defender_policy)` returns
          (attacker_action, defender_action, pre_data). `pre_data` is forwarded
          to `on_step_post`.
        - `on_step_post` (optional) can build a record after seeing the outcome
          of the step; its return values are grouped per episode.
        """
        if num_episodes <= 0:
            return [], []

        episode_rewards: list[float] = []
        episode_step_records: list[list[Any]] = []
        env_obs: list[Any] = [None for _ in range(self.num_envs)]
        env_info: list[Any] = [None for _ in range(self.num_envs)]
        env_done: list[bool] = [False for _ in range(self.num_envs)]
        env_policies: list[tuple[Any, Any]] = [None for _ in range(self.num_envs)]  # type: ignore
        env_ep_rewards: list[float] = [0.0 for _ in range(self.num_envs)]
        env_ep_steps: list[list[Any]] = [[] for _ in range(self.num_envs)]

        def _start_episode(env_id: int):
            obs, info = self.envs[env_id].reset()
            attacker_policy, defender_policy = on_reset(env_id)
            for policy in (attacker_policy, defender_policy):
                if policy is None:
                    continue
                if hasattr(policy, "reset"):
                    try:
                        policy.reset()
                    except TypeError:
                        policy.reset()  # type: ignore
                if hasattr(policy, "reset_state"):
                    policy.reset_state()
            env_obs[env_id] = obs
            env_info[env_id] = info
            env_policies[env_id] = (attacker_policy, defender_policy)
            env_done[env_id] = False
            env_ep_rewards[env_id] = 0.0
            env_ep_steps[env_id] = []

        for env_id in range(self.num_envs):
            _start_episode(env_id)

        completed = 0
        while completed < num_episodes:
            for env_id in range(self.num_envs):
                if completed >= num_episodes:
                    break

                if env_done[env_id]:
                    _start_episode(env_id)

                attacker_policy, defender_policy = env_policies[env_id]
                obs = env_obs[env_id]
                info = env_info[env_id]
                attacker_action, defender_action, pre_data = on_step(
                    env_id, obs, info, attacker_policy, defender_policy
                )

                next_obs, reward, terminated, truncated, next_info = self.envs[env_id].step(
                    {"attacker_action": attacker_action, "defender_action": defender_action}
                )
                done = bool(terminated or truncated)

                if isinstance(reward, dict):
                    step_reward = float(reward.get(reward_key, 0.0))
                else:
                    step_reward = float(reward)
                env_ep_rewards[env_id] += step_reward

                if on_step_post is not None:
                    record = on_step_post(
                        env_id,
                        pre_data,
                        next_obs,
                        reward,
                        done,
                        next_info,
                        attacker_policy,
                        defender_policy,
                    )
                    if record is not None:
                        env_ep_steps[env_id].append(record)

                env_obs[env_id] = next_obs
                env_info[env_id] = next_info
                env_done[env_id] = done

                if done:
                    episode_rewards.append(env_ep_rewards[env_id])
                    episode_step_records.append(env_ep_steps[env_id])
                    completed += 1
                    env_done[env_id] = True

        return episode_rewards, episode_step_records

    def run_episodes_batched(
        self,
        num_episodes: int,
        on_reset: Callable[[int], tuple[Any, Any]],
        on_step_batch: Callable[
            [list[int], list[Any], list[Any], list[Any], list[Any]],
            tuple[list[list[int]], list[list[int]], list[Any | None] | None],
        ],
        reward_key: str = "attacker",
        reward_mode: str = "utility",
        on_step_post: Callable[[int, Any, Any, Any, bool, Any, Any, Any], Any | None] | None = None,
    ) -> tuple[list[float], list[Any]]:
        """
        Like run_episodes, but computes actions for all active envs in a batch.

        - `on_step_batch(env_ids, obs_list, info_list, attacker_policies, defender_policies)`
          should return (attacker_actions_list, defender_actions_list, pre_data_list).
        """
        if num_episodes <= 0:
            return [], []
        if reward_mode not in {"utility", "win_rate"}:
            raise ValueError("reward_mode must be 'utility' or 'win_rate'")

        episode_rewards: list[float] = []
        episode_step_records: list[list[Any]] = []
        env_obs: list[Any] = [None for _ in range(self.num_envs)]
        env_info: list[Any] = [None for _ in range(self.num_envs)]
        env_done: list[bool] = [False for _ in range(self.num_envs)]
        env_policies: list[tuple[Any, Any]] = [None for _ in range(self.num_envs)]  # type: ignore
        env_ep_rewards: list[float] = [0.0 for _ in range(self.num_envs)]
        env_ep_steps: list[list[Any]] = [[] for _ in range(self.num_envs)]

        def _start_episode(env_id: int):
            obs, info = self.envs[env_id].reset()
            attacker_policy, defender_policy = on_reset(env_id)
            for policy in (attacker_policy, defender_policy):
                if policy is None:
                    continue
                if hasattr(policy, "reset"):
                    try:
                        policy.reset()
                    except TypeError:
                        policy.reset()  # type: ignore
                if hasattr(policy, "reset_state"):
                    policy.reset_state()
            env_obs[env_id] = obs
            env_info[env_id] = info
            env_policies[env_id] = (attacker_policy, defender_policy)
            env_done[env_id] = False
            env_ep_rewards[env_id] = 0.0
            env_ep_steps[env_id] = []

        for env_id in range(self.num_envs):
            _start_episode(env_id)

        completed = 0
        while completed < num_episodes:
            remaining = num_episodes - completed
            active_ids: list[int] = []
            active_obs: list[Any] = []
            active_info: list[Any] = []
            active_att_policies: list[Any] = []
            active_def_policies: list[Any] = []

            for env_id in range(self.num_envs):
                if len(active_ids) >= remaining:
                    break
                if env_done[env_id]:
                    _start_episode(env_id)
                attacker_policy, defender_policy = env_policies[env_id]
                active_ids.append(env_id)
                active_obs.append(env_obs[env_id])
                active_info.append(env_info[env_id])
                active_att_policies.append(attacker_policy)
                active_def_policies.append(defender_policy)

            attacker_actions_list, defender_actions_list, pre_data_list = on_step_batch(
                active_ids, active_obs, active_info, active_att_policies, active_def_policies
            )
            if pre_data_list is None:
                pre_data_list = [None for _ in active_ids]
            if len(attacker_actions_list) != len(active_ids) or len(defender_actions_list) != len(active_ids):
                raise ValueError("on_step_batch must return action lists matching active env count")

            for local_idx, env_id in enumerate(active_ids):
                attacker_action = attacker_actions_list[local_idx]
                defender_action = defender_actions_list[local_idx]
                pre_data = pre_data_list[local_idx]

                next_obs, reward, terminated, truncated, next_info = self.envs[env_id].step(
                    {"attacker_action": attacker_action, "defender_action": defender_action}
                )
                done = bool(terminated or truncated)

                if isinstance(reward, dict):
                    step_reward = float(reward.get(reward_key, 0.0))
                else:
                    step_reward = float(reward)
                env_ep_rewards[env_id] += step_reward

                if on_step_post is not None:
                    record = on_step_post(
                        env_id,
                        pre_data,
                        next_obs,
                        reward,
                        done,
                        next_info,
                        active_att_policies[local_idx],
                        active_def_policies[local_idx],
                    )
                    if record is not None:
                        env_ep_steps[env_id].append(record)

                env_obs[env_id] = next_obs
                env_info[env_id] = next_info
                env_done[env_id] = done

                if done:
                    episode_rewards.append(env_ep_rewards[env_id])
                    episode_step_records.append(env_ep_steps[env_id])
                    completed += 1
                    env_done[env_id] = True

        if reward_mode == "win_rate":
            episode_rewards = [0.0 if r == -1 else r for r in episode_rewards]

        return episode_rewards, episode_step_records
