from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any

import numpy as np
import torch
import dgl

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.envs.vec_rollout_pool import VecRolloutPool
from graphchase.solver.grasper.grasper_game import GrasperGame
from graphchase.solver.grasper.grasper_mappo.envs.env import RL_Env
from graphchase.solver.grasper.grasper_mappo.runner_shared.env_runner import EnvRunner
from graphchase.solver.grasper.utils.utils import obs_query, shared_obs_query
from graphchase.solver.grasper.utils.graph_learning_utils import get_dgl_graph
from graphchase.solver.grasper.graph_learning.encoder import PreModel

logger = logging.getLogger(__name__)


class GrasperMappoAgent:
    def __init__(self, runner: "GrasperMappoPsroRunner") -> None:
        self.runner = runner

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.runner.env_runner.trainer_ft.policy.actor.state_dict(),
            "critic": self.runner.env_runner.trainer_ft.policy.critic.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if "actor" in state:
            self.runner.env_runner.trainer_ft.policy.actor.load_state_dict(state["actor"])
        if "critic" in state:
            self.runner.env_runner.trainer_ft.policy.critic.load_state_dict(state["critic"])


class GrasperMappoPsroRunner:
    def __init__(
        self,
        mappo_args,
        args,
        game: GrasperGame,
        node_embs: np.ndarray | None = None,
        pooled_node_emb: np.ndarray | None = None,
        hypernet_state: dict[str, Any] | None = None,
    ) -> None:
        self.mappo_args = mappo_args
        self.args = args
        self.game = game
        self.device = args.device
        self.graph_emb_model: PreModel | None = None

        feat = self.game.get_graph_info()
        self.args.node_num = len(self.game.node_list)
        self.args.feat_dim = feat.shape[1]
        self.args.defender_num = self.game._defender_num

        self.env = RL_Env(game, action_type=args.action_type)
        self.env_runner = EnvRunner(self.env, mappo_args, args)

        if hypernet_state is not None:
            self.env_runner.trainer.policy.actor.load_state_dict(hypernet_state["actor"])
            self.env_runner.trainer.policy.critic.load_state_dict(hypernet_state["critic"])

        self.node_embs = node_embs
        self.pooled_node_emb = pooled_node_emb
        self.T = self._build_time_embedding()

        if self.args.use_end_to_end:
            self._init_from_hypernet_e2e()
        else:
            if self.node_embs is None or self.pooled_node_emb is None:
                self.node_embs, self.pooled_node_emb = self._compute_graph_embeddings()
            self._init_from_hypernet(self.pooled_node_emb)

        if self.args.use_emb_layer and self.args.load_pretrain_model:
            self.load_emb_layer()

        self.agent = GrasperMappoAgent(self)

    def _build_time_embedding(self) -> np.ndarray:
        horizon = int(getattr(self.args, "max_time_horizon_for_state_emb", self.game._time_horizon + 1))
        t = np.zeros(horizon, dtype=float)
        idx = int(self.game._time_horizon)
        if 0 <= idx < horizon:
            t[idx] = 1.0
        elif horizon > 0:
            t[-1] = 1.0
        return t

    def _compute_graph_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        feat = self.game.get_graph_info()
        graph_emb_model = PreModel(feat.shape[1], self.args.gnn_hidden_dim, self.args.gnn_output_dim, self.args.gnn_num_layer, self.args.gnn_dropout)
        graph_emb_model.to(self.device)
        if self.args.load_graph_emb_model:
            graph_model_path = self.args.graph_emb_model_path
            if graph_model_path is None:
                args = self.args
                game_pool_str = f"_gp{args.pool_size}" if args.load_game_pool_file else ""
                differ_size_str = "_ds" if args.graph_type == "Grid_Graph" and args.differ_size else ""
                file_name_suffix = (
                    f"_type_{args.graph_type}"
                    f"{differ_size_str}_ep{args.edge_probability}{game_pool_str}_layer{args.gnn_num_layer}_"
                    f"hidden{args.gnn_hidden_dim}_out{args.gnn_output_dim}_dnum{args.min_num_defender}_"
                    f"{args.max_num_defender}_enum{args.min_num_exit}_{args.max_num_exit}_mep{args.min_attacker_pth_len}.pt"
                )
                default_name = f"checkpoint_epoch1000{file_name_suffix}"
                graph_model_path = os.path.join(args.pre_pretrain_save_path, default_name)
                if os.path.isdir(args.pre_pretrain_save_path):
                    pattern = re.compile(rf"^checkpoint_epoch(?P<epoch>\d+){re.escape(file_name_suffix)}$")
                    candidates = []
                    for name in os.listdir(args.pre_pretrain_save_path):
                        match = pattern.match(name)
                        if match:
                            candidates.append((int(match.group("epoch")), name))
                    if candidates:
                        _, best_name = max(candidates, key=lambda item: item[0])
                        graph_model_path = os.path.join(args.pre_pretrain_save_path, best_name)
                        logger.info("Auto-selected graph model checkpoint: %s", graph_model_path)
            if graph_model_path and os.path.isfile(graph_model_path):
                graph_emb_model.load(torch.load(graph_model_path))
                logger.info("Load graph model checkpoint: %s", graph_model_path)
            else:
                logger.info("Graph model checkpoint not found; using random init (%s)", graph_model_path)
        graph_emb_model.eval()
        self.graph_emb_model = graph_emb_model
        with torch.no_grad():
            hg = get_dgl_graph(self.game)
            hg = hg.to(self.device)
            feat_tensor = hg.ndata["attr"]
            node_embs, pooled_node_emb = graph_emb_model.embed(hg, feat_tensor)
        return node_embs.cpu().numpy(), pooled_node_emb.cpu().numpy()

    def _init_from_hypernet(self, pooled_node_emb: np.ndarray) -> None:
        graph_embs = torch.FloatTensor(pooled_node_emb).to(self.device)
        Ts = torch.FloatTensor(self.T).to(self.device)
        wa, ba = self.env_runner.trainer.policy.actor.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
        self.env_runner.trainer_ft.policy.actor.init_paras(wa, ba)
        wc, bc = self.env_runner.trainer.policy.critic.base.get_weight(graph_embs.unsqueeze(0), Ts.unsqueeze(0))
        self.env_runner.trainer_ft.policy.critic.init_paras(wc, bc)

    def _init_from_hypernet_e2e(self) -> None:
        hg = get_dgl_graph(self.game)
        hgs_batch = dgl.batch([hg]).to(self.device)
        Ts = torch.FloatTensor(np.array([self.T])).to(self.device)
        wa, ba, node_embs, pooled_node_emb = self.env_runner.trainer.policy.actor.base.get_weight(hgs_batch, Ts)
        self.env_runner.trainer_ft.policy.actor.init_paras(wa, ba)
        self.node_embs = node_embs.cpu().numpy()
        self.pooled_node_emb = pooled_node_emb.cpu().numpy()
        wc, bc, _, _ = self.env_runner.trainer.policy.critic.base.get_weight(hgs_batch, Ts)
        self.env_runner.trainer_ft.policy.critic.init_paras(wc, bc)

    def load_emb_layer(self) -> None:
        self.env_runner.trainer_ft.policy.actor.init_emb_layer(
            self.env_runner.trainer.policy.actor.base.node_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.actor.base.time_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.actor.base.agent_id_emb_layer.state_dict(),
        )
        self.env_runner.trainer_ft.policy.critic.init_emb_layer(
            self.env_runner.trainer.policy.critic.base.node_idx_emb_layer.state_dict(),
            self.env_runner.trainer.policy.critic.base.time_idx_emb_layer.state_dict(),
        )

    def _build_obs(self, obs: dict, info: dict) -> tuple[np.ndarray, np.ndarray]:
        attacker_states = obs.get("attacker_state", [])
        defender_states = obs.get("defender_state", [])
        attacker_node = 0
        if isinstance(attacker_states, np.ndarray):
            has_attacker = attacker_states.size > 0
        else:
            has_attacker = len(attacker_states) > 0
        if has_attacker:
            attacker_node = self._position_to_node(tuple(attacker_states[0]))
        if isinstance(defender_states, np.ndarray):
            defender_iter = defender_states.tolist()
        else:
            defender_iter = defender_states
        defender_nodes = [self._position_to_node(tuple(pos)) for pos in defender_iter]
        cur_time = int(info.get("cur_time", 0))
        shared_state = [attacker_node] + defender_nodes + [cur_time]
        shared_obs = np.array([shared_state for _ in range(self.env.defender_num)])
        obs_list = []
        for idx, node in enumerate(defender_nodes):
            obs_list.append([attacker_node, node, cur_time, idx])
        return shared_obs, np.array(obs_list)

    def _position_to_node(self, position: Any) -> int:
        start, end, dist = position
        if start == end or dist <= 0:
            return int(end)
        return int(start)

    def _build_action_mask(self, legal_actions: list[list[int]], action_dim: int) -> np.ndarray:
        mask = np.zeros((len(legal_actions), action_dim), dtype=bool)
        for i, legal in enumerate(legal_actions):
            valid_len = min(len(legal), action_dim)
            if valid_len > 0:
                mask[i, :valid_len] = True
        return mask

    def _format_pooled_emb(self, batch_size: int) -> torch.Tensor | None:
        if not self.args.use_augmentation or self.pooled_node_emb is None:
            return None
        pooled = np.array(self.pooled_node_emb, dtype=np.float32)
        if pooled.ndim == 1:
            pooled = np.repeat(pooled.reshape(1, -1), batch_size, axis=0)
        return torch.FloatTensor(pooled).to(self.device)

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        shared_obs, obs_arr = self._build_obs(obs, info)
        if self.args.perfect_info:
            obs_arr = np.concatenate((shared_obs, np.expand_dims(obs_arr[:, -1], 1)), axis=1)
        if self.args.use_node_emb:
            obs_arr = obs_query(self.node_embs, obs_arr, self.args.max_time_horizon_for_state_emb, info.get("cur_time", 0), self.game.node_to_idx)
            if self.args.use_augmentation:
                obs_arr = np.concatenate((obs_arr, np.repeat(self.pooled_node_emb.reshape(1, -1), obs_arr.shape[0], axis=0)), axis=1)

        if self.args.use_emb_layer:
            obs_tensor = torch.LongTensor(obs_arr).to(self.device)
        else:
            obs_tensor = torch.FloatTensor(obs_arr).to(self.device)

        pooled_emb = self._format_pooled_emb(obs_tensor.shape[0])
        legal_actions = info.get("defender_legal_action", [])
        if not legal_actions:
            legal_actions = self.env._env._legal_actions(is_attacker=False)
        action_mask = self._build_action_mask(legal_actions, self.env.action_dim)
        action_mask_tensor = torch.BoolTensor(action_mask).to(self.device)
        actions = self.env_runner.trainer_ft.policy.act(
            obs_tensor,
            pooled_emb,
            action_mask=action_mask_tensor,
            batch=True,
        )
        action_indices = actions.detach().cpu().numpy().reshape(-1).astype(int).tolist()
        return [legal[action_idx] for action_idx, legal in zip(action_indices, legal_actions)]

    def compute_best_response(self, attacker_runners, config: dict | None = None, meta_strategy: Any | None = None):
        if config is None:
            raise ValueError("config must be provided for compute_best_response")
        train_batches = int(config.get("train_batches", 1))
        episodes_per_batch = int(config.get("episodes_per_batch", 1))
        train_num_per_ite = int(self.args.ppo_epoch)
        vec_envs = int(config.get("vec_envs", 1))
        num_envs = min(episodes_per_batch, max(1, vec_envs))
        num_defender = self.env.defender_num

        if not attacker_runners:
            raise ValueError("attacker_runners must be a non-empty list")

        def _choose_attacker_runner():
            if meta_strategy is None:
                probs = np.ones(len(attacker_runners), dtype=float) / float(len(attacker_runners))
            else:
                probs = np.asarray(meta_strategy, dtype=float)
                if probs.shape[0] != len(attacker_runners) or probs.sum() <= 0:
                    probs = np.ones(len(attacker_runners), dtype=float) / float(len(attacker_runners))
                else:
                    probs = probs / probs.sum()
            idx = int(np.random.choice(len(attacker_runners), p=probs))
            runner = attacker_runners[idx]
            if hasattr(runner, "clone"):
                runner = runner.clone()
            else:
                runner = copy.deepcopy(runner)
            if hasattr(runner, "reset"):
                runner.reset()
            return runner

        def _expand_pooled_node_emb(pooled_node_emb, repeat: int):
            if pooled_node_emb is None:
                return None
            pooled = np.array(pooled_node_emb, dtype=np.float32)
            if pooled.ndim == 1:
                pooled = np.repeat(pooled.reshape(1, -1), repeat, axis=0)
            return pooled

        def _encode_obs(raw_obs, info):
            shared_obs, obs_arr = self._build_obs(raw_obs, info)
            if self.args.perfect_info:
                obs_arr = np.concatenate((shared_obs, np.expand_dims(obs_arr[:, -1], 1)), axis=1)
            pooled_node_emb = self.pooled_node_emb
            if self.args.use_node_emb:
                cur_time = int(info.get("cur_time", 0))
                shared_obs = shared_obs_query(
                    self.node_embs,
                    shared_obs,
                    self.args.max_time_horizon_for_state_emb,
                    cur_time,
                    self.game.node_to_idx,
                )
                obs_arr = obs_query(
                    self.node_embs,
                    obs_arr,
                    self.args.max_time_horizon_for_state_emb,
                    cur_time,
                    self.game.node_to_idx,
                )
                if self.args.use_augmentation:
                    pooled_node_emb = _expand_pooled_node_emb(pooled_node_emb, shared_obs.shape[0])
                    shared_obs = np.concatenate((shared_obs, pooled_node_emb), axis=1)
                    obs_arr = np.concatenate((obs_arr, pooled_node_emb), axis=1)
            else:
                if not self.args.use_augmentation:
                    pooled_node_emb = None
                else:
                    pooled_node_emb = _expand_pooled_node_emb(pooled_node_emb, shared_obs.shape[0])
            return shared_obs, obs_arr, pooled_node_emb

        env_builder = lambda: UNSGEnv(self.game.settings)
        pool = VecRolloutPool(env_builder, num_envs=num_envs)
        def on_reset(env_id: int):
            attacker_episode_runner = _choose_attacker_runner()
            return attacker_episode_runner, None

        def on_step_batch(env_ids, obs_list, info_list, attacker_episode_runners, __):
            attacker_actions_list = [
                runner.policy_action(obs, info)
                for runner, obs, info in zip(attacker_episode_runners, obs_list, info_list)
            ]
            shared_obs_list = []
            obs_arr_list = []
            pooled_emb_list = []
            legal_actions_list = []
            action_masks_list = []
            for env_id, obs, info in zip(env_ids, obs_list, info_list):
                shared_obs, obs_arr, pooled_node_emb = _encode_obs(obs, info)
                shared_obs_list.append(shared_obs)
                obs_arr_list.append(obs_arr)
                pooled_emb_list.append(pooled_node_emb)
                legal_actions = info.get("defender_legal_action", [])
                if not legal_actions:
                    legal_actions = pool.envs[env_id]._legal_actions(is_attacker=False)
                legal_actions_list.append(legal_actions)
                action_masks_list.append(self._build_action_mask(legal_actions, self.env.action_dim))

            batch_shared_obs = np.concatenate(shared_obs_list, axis=0)
            batch_obs = np.concatenate(obs_arr_list, axis=0)
            pooled_batch = None
            if self.args.use_augmentation:
                pooled_parts = []
                for pooled in pooled_emb_list:
                    if pooled is None:
                        pooled_parts = []
                        break
                    pooled_parts.append(pooled)
                if pooled_parts:
                    pooled_batch = np.concatenate(pooled_parts, axis=0)

            if self.args.use_emb_layer:
                shared_tensor = torch.LongTensor(batch_shared_obs).to(self.device)
                obs_tensor = torch.LongTensor(batch_obs).to(self.device)
            else:
                shared_tensor = torch.FloatTensor(batch_shared_obs).to(self.device)
                obs_tensor = torch.FloatTensor(batch_obs).to(self.device)
            pooled_tensor = torch.FloatTensor(pooled_batch).to(self.device) if pooled_batch is not None else None
            action_mask_batch = np.concatenate(action_masks_list, axis=0)
            action_mask_tensor = torch.BoolTensor(action_mask_batch).to(self.device)

            with torch.no_grad():
                values, actions, action_log_probs = self.env_runner.trainer_ft.policy.get_actions(
                    shared_tensor,
                    obs_tensor,
                    pooled_tensor,
                    action_mask=action_mask_tensor,
                    batch=True,
                )

            values = values.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            action_log_probs = action_log_probs.detach().cpu().numpy()

            batch_envs = len(env_ids)
            values = values.reshape(batch_envs, num_defender, -1)
            actions = actions.reshape(batch_envs, num_defender, -1)
            action_log_probs = action_log_probs.reshape(batch_envs, num_defender, -1)

            defender_actions_list = []
            pre_data_list = []
            for idx, info in enumerate(info_list):
                action_indices = np.atleast_1d(actions[idx].squeeze(-1)).astype(int).tolist()
                legal_actions = legal_actions_list[idx]
                defender_actions = [legal[action_idx] for action_idx, legal in zip(action_indices, legal_actions)]
                defender_actions_list.append(defender_actions)
                pre_data_list.append(
                    {
                        "shared_obs": shared_obs_list[idx],
                        "obs": obs_arr_list[idx],
                        "values": values[idx],
                        "actions": actions[idx],
                        "action_log_probs": action_log_probs[idx],
                        "pooled_node_emb": pooled_emb_list[idx],
                        "action_mask": action_masks_list[idx],
                    }
                )
            return attacker_actions_list, defender_actions_list, pre_data_list

        def on_step_post(env_id, pre_data, next_obs, reward, done, next_info, __, ___):
            next_shared_obs, _, _ = _encode_obs(next_obs, next_info)
            reward_val = float(reward.get("defender", 0.0)) if isinstance(reward, dict) else float(reward)
            rewards = np.full((num_defender, 1), reward_val, dtype=np.float32)
            dones = np.full((num_defender, 1), bool(done), dtype=bool)
            pre_data.update(
                {
                    "rewards": rewards,
                    "dones": dones,
                    "next_shared_obs": next_shared_obs,
                }
            )
            return pre_data

        try:
            for _ in range(train_batches):
                self.env_runner.trainer_ft.policy.actor.eval()
                self.env_runner.trainer_ft.policy.critic.eval()
                _, episode_step_records = pool.run_episodes_batched(
                    num_episodes=episodes_per_batch,
                    on_reset=on_reset,
                    on_step_batch=on_step_batch,
                    reward_key="defender",
                    on_step_post=on_step_post,
                )
                last_shared_obs = None
                last_pooled_node_emb = None
                for episode_steps_record in episode_step_records:
                    if not episode_steps_record:
                        continue
                    for step in episode_steps_record:
                        masks = np.ones((num_defender, 1), dtype=np.float32)
                        if np.any(step["dones"]):
                            masks[step["dones"].astype(bool)] = 0.0
                        self.env_runner.buffer_ft.insert(
                            step["shared_obs"],
                            step["obs"],
                            step["actions"],
                            step["action_log_probs"],
                            step["values"],
                            step["rewards"],
                            masks,
                            None,
                            step["pooled_node_emb"],
                            action_masks=step["action_mask"],
                        )
                    self.env_runner.buffer_ft.episode_length.append(len(episode_steps_record))
                    last_shared_obs = episode_steps_record[-1]["next_shared_obs"]
                    last_pooled_node_emb = episode_steps_record[-1]["pooled_node_emb"]
                if last_shared_obs is not None:
                    self.env_runner.compute_ft(last_shared_obs, last_pooled_node_emb)
                    self.env_runner.train_ft(train_num_per_ite=train_num_per_ite)
        finally:
            pool.close()

    def reset(self) -> None:
        if hasattr(self.env, "initialize_attacker_strategy"):
            self.env.initialize_attacker_strategy()

    def clone(self, agent_state_dict: dict[str, Any] | None = None):
        hypernet_state = {
            "actor": copy.deepcopy(self.env_runner.trainer.policy.actor.state_dict()),
            "critic": copy.deepcopy(self.env_runner.trainer.policy.critic.state_dict()),
        }
        cloned = GrasperMappoPsroRunner(
            self.mappo_args,
            self.args,
            self.game,
            node_embs=self.node_embs,
            pooled_node_emb=self.pooled_node_emb,
            hypernet_state=hypernet_state,
        )
        if agent_state_dict is not None:
            cloned.agent.load_state_dict(agent_state_dict)
        return cloned

    def save(self, file_name: str) -> None:
        policy_actor = self.env_runner.trainer_ft.policy.actor
        policy_critic = self.env_runner.trainer_ft.policy.critic
        torch.save(policy_actor.state_dict(), file_name + "_actor.pt")
        torch.save(policy_critic.state_dict(), file_name + "_critic.pt")
        if self.graph_emb_model is not None:
            self.graph_emb_model.save(file_name + "_graph_emb.pt")
