import copy
import math
import time
import torch
import numpy as np
import sys
import logging
import os
import pickle
import re
from datetime import datetime
import os.path as osp

import dgl

from .envs.env import RL_Env
from .runner_shared.env_runner import EnvRunner as Runner
from .config import get_config, get_train_utility_results_dir, get_mtl_model_results_dir
from graphchase.graph.gnn_graph import load_game_pool
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.envs.vec_rollout_pool import VecRolloutPool
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.agents.attacker_path_agent import PathAgent
from ..grasper_game import GrasperGame
from ..graph_learning.encoder import PreModel
from ..utils.graph_learning_utils import get_dgl_graph
from ..utils.utils import shared_obs_query, obs_query, get_demonstration

logger = logging.getLogger(__name__)


def grasper_mappo_mtl(args):
    """
        Methods: 1. End to End: GNN + Hypernetwork + torch.Embedding (i.e., currently only support using state rep. layer)
                    !!! During fine-tunning, may need to distinguish the hyper_outputs of actor and critic. !!!

                2. Use GNN node embedding to represent the state. This requires a GNN model (w/ or w/o pretrain).
                   Two cases: i) use augmentation: pooled GNN node embedding will be concatenated to state
                             ii) no augmentation

                3. Use torch.Embedding layer to encode the state. This requires a GNN model (w/ or w/o pretrain).
                   Two cases: i) use augmentation: pooled GNN node embedding will be concatenated to state
                             ii) no augmentation

                4. Use raw state as input: args.use_node_emb = False and args.use_emb_layer = False
                   This requires a GNN model (w/ or w/o pretrain).
                   Two cases: i) use augmentation: pooled GNN node embedding will be concatenated to state
                             ii) no augmentation
    """
    action_type = args.action_type
    if not args.load_game_pool_file:
        raise ValueError("grasper_mappo_mtl requires --load_game_pool_file to be set")
    game_pool = load_game_pool(args)
    if not game_pool:
        raise ValueError("Loaded game pool is empty")
    game_pool_str = f"_gp{args.pool_size}"
    settings = np.random.choice(game_pool)
    game = GrasperGame(settings, args, action_type=action_type, compute_path=True)
    differ_size_str = ""
    parser = get_config()
    mappo_args = parser.parse_known_args(sys.argv[1:])[0]
    if args.update_every_n_episodes < 0:
        mappo_args.ppo_epoch = 15
    if args.use_end_to_end:
        args.use_emb_layer = True   # currently only support using state rep. layer
        args.use_node_emb = False
    assert not (args.use_emb_layer and args.use_node_emb), \
        "Error: choose one of the options: i) use embedding layer to get the state embedding, " \
        "ii) query the node embedding to represent the state embedding."
    if args.use_emb_layer:
        args.use_node_emb = False
    if args.use_node_emb:
        args.use_emb_layer = False
    setup_str = f"E2E: {args.use_end_to_end}, EL: {args.use_emb_layer}, NE: {args.use_node_emb}, Aug: {args.use_augmentation}, AS: {args.use_act_supervisor}"
    feat = game.get_graph_info()
    args.node_num = len(game.node_list)
    args.feat_dim = feat.shape[1]
    args.defender_num = game._defender_num

    env = RL_Env(game, action_type=action_type)
    runner = Runner(env, mappo_args, args)

    graph_emb_model = None
    if not args.use_end_to_end:
        graph_emb_model = PreModel(
            feat.shape[1],
            args.gnn_hidden_dim,
            args.gnn_output_dim,
            args.gnn_num_layer,
            args.gnn_dropout,
        )
        graph_emb_model.to(args.device)
        if args.load_graph_emb_model:
            print("Load pretrained graph model ...")
            graph_model_path = args.graph_emb_model_path
            if graph_model_path is None:
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
            graph_emb_model.load(torch.load(graph_model_path))
            print(graph_model_path)
        graph_emb_model.eval()

    def _t2n(tensor):
        return tensor.detach().cpu().numpy()

    def _position_to_node(position):
        start, end, dist = position
        if start == end or dist <= 0:
            return int(end)
        return int(start)

    def _build_obs(raw_obs, info, defender_num):
        attacker_states = raw_obs.get("attacker_state", [])
        defender_states = raw_obs.get("defender_state", [])
        attacker_node = 0
        if isinstance(attacker_states, np.ndarray):
            has_attacker = attacker_states.size > 0
        else:
            has_attacker = len(attacker_states) > 0
        if has_attacker:
            attacker_node = _position_to_node(tuple(attacker_states[0]))
        if isinstance(defender_states, np.ndarray):
            defender_iter = defender_states.tolist()
        else:
            defender_iter = defender_states
        defender_nodes = [_position_to_node(tuple(pos)) for pos in defender_iter]
        cur_time = int(info.get("cur_time", 0))
        shared_state = [attacker_node] + defender_nodes + [cur_time]
        shared_obs = np.array([shared_state for _ in range(defender_num)])
        obs_list = [[attacker_node, node, cur_time, idx] for idx, node in enumerate(defender_nodes)]
        return shared_obs, np.array(obs_list)

    def _encode_obs(raw_obs, info, current_game, node_embeddings, pooled_embeddings):
        shared_obs, obs = _build_obs(raw_obs, info, current_game._defender_num)
        demo_obs = obs.copy()
        if args.perfect_info:
            obs = np.concatenate((shared_obs, np.expand_dims(obs[:, -1], 1)), axis=1)
        if args.use_node_emb:
            cur_time = int(info.get("cur_time", 0))
            shared_obs = shared_obs_query(
                node_embeddings,
                shared_obs,
                args.max_time_horizon_for_state_emb,
                cur_time,
                current_game.node_to_idx,
            )
            obs = obs_query(
                node_embeddings,
                obs,
                args.max_time_horizon_for_state_emb,
                cur_time,
                current_game.node_to_idx,
            )
            if args.use_augmentation:
                shared_obs = np.concatenate((shared_obs, pooled_embeddings), axis=1)
                obs = np.concatenate((obs, pooled_embeddings), axis=1)
        return shared_obs, obs, demo_obs

    def _map_actions(action_indices, legal_actions):
        mapped = []
        for idx, legal in zip(action_indices, legal_actions):
            if idx < 0 or idx >= len(legal):
                raise ValueError(f"Invalid action index {idx} for legal actions of length {len(legal)}")
            mapped.append(int(legal[idx]))
        return mapped

    def _build_action_mask(legal_actions, action_dim):
        mask = np.zeros((len(legal_actions), action_dim), dtype=bool)
        for i, legal in enumerate(legal_actions):
            valid_len = min(len(legal), action_dim)
            if valid_len > 0:
                mask[i, :valid_len] = True
        return mask

    def _build_time_embeddings(current_game):
        t_vec = [0] * args.max_time_horizon_for_state_emb
        t_idx = min(int(current_game._time_horizon), args.max_time_horizon_for_state_emb - 1)
        t_vec[t_idx] = 1
        return np.array([t_vec for _ in range(current_game._defender_num)])

    batch_size = int(args.num_sample)
    context = {
        "pool": None,
        "attacker_runner": None,
        "node_embs": None,
        "pooled_node_embs": None,
        "hgs": None,
        "hgs_batch": None,
        "Ts": None,
        "env_builder": None,
    }

    def _prepare_game_context(current_game):
        runner.reset(current_game)
        env_builder = lambda: UNSGEnv(current_game.settings)
        if context["pool"] is not None:
            context["pool"].close()
        context["pool"] = VecRolloutPool(env_builder, num_envs=batch_size)
        context["env_builder"] = env_builder
        context["attacker_runner"] = AttackerPathRunner(
            env_builder=env_builder,
            action_type=args.action_type,
            strategy_type="mix",
            max_path_length=current_game._time_horizon,
        )
        if args.use_end_to_end:
            hg = get_dgl_graph(current_game)
            hgs = [hg for _ in range(current_game._defender_num)]
            context["hgs"] = hgs
            context["hgs_batch"] = dgl.batch(hgs).to(args.device)
            context["node_embs"] = None
            context["pooled_node_embs"] = None
        else:
            with torch.no_grad():
                hg = get_dgl_graph(current_game)
                hg = hg.to(args.device)
                feat = hg.ndata["attr"]
                node_embs, pooled_node_emb = graph_emb_model.embed(hg, feat)
                node_embs = node_embs.cpu().numpy()
                pooled_node_emb = pooled_node_emb.cpu().numpy()
                pooled_node_embs = np.array([pooled_node_emb for _ in range(runner.num_defender)])
            context["node_embs"] = node_embs
            context["pooled_node_embs"] = pooled_node_embs
            context["hgs"] = None
            context["hgs_batch"] = None
        context["Ts"] = _build_time_embeddings(current_game)

    _prepare_game_context(game)

    time_list, reward_list, aloss_list, vloss_list, itera_list = [], [], [], [], []
    start_iter = 0

    if args.checkpoint > 0:
        adaption_reward = 0
        game_perf = 0
        episode_count = 0
        start_iter = args.checkpoint
        if args.use_cache:
            # currently not used
            actor_checkpoint = f"data/pretrain_models/grasper_mappo/graph_100_probability_1.0_poa_1.0/pretrain_model/not_e2e/num_gtst5_5_10_1_iter{args.checkpoint}_bsize256_node_feat3_gnn2_128_32_dnum5_5_enum8_8_T6_12_mep6_une-1_use_el_load_gem_pi1_as1_0.1_0.0_1500000_actor.pt"
            critic_checkpoint = f"data/pretrain_models/grasper_mappo/graph_100_probability_1.0_poa_1.0/pretrain_model/not_e2e/num_gtst5_5_10_1_iter{args.checkpoint}_bsize256_node_feat3_gnn2_128_32_dnum5_5_enum8_8_T6_12_mep6_une-1_use_el_load_gem_pi1_as1_0.1_0.0_1500000_critic.pt"
            result_pth = "data/pretrain_models/grasper_mappo/graph_100_probability_1.0_poa_1.0/utility_record/not_e2e/seed101_num_gtst5_5_10_1_bsize256_node_feat3_gnn2_128_32_dnum5_5_enum8_8_T6_12_mep6_une-1_use_el_load_gem_pi1_as1_0.1_0.0_1500000_reward.pik"
        else:
            actor_checkpoint = get_mtl_model_results_dir(game, action_type, args, args.checkpoint) + "_actor.pt"
            critic_checkpoint = get_mtl_model_results_dir(game, action_type, args, args.checkpoint) + "_critic.pt"
            result_pth = get_train_utility_results_dir(game, action_type, args)
        print("Load Checkpoint {} ......".format(args.checkpoint))
        runner.load_checkpoint(actor_checkpoint, critic_checkpoint)
        if os.path.exists(result_pth) and os.path.getsize(result_pth) > 0:
            data = pickle.load(open(result_pth, 'rb'))
            reward_list, time_list, aloss_list, vloss_list, itera_list = data['reward_list'], data['time_list'], data['aloss_list'], data['vloss_list'], data['itera_list']
            for i in range(len(reward_list)):
                logger.info("Loaded reward history: iter=%s reward=%.6f", itera_list[i], reward_list[i])

    # train
    min_update_episodes = args.num_games * args.num_task * batch_size
    update_game_freq = args.num_task * batch_size
    update_every_n_episodes = None
    if args.update_every_n_episodes > 0:
        update_every_n_episodes = max(1, int(math.ceil(args.update_every_n_episodes / float(batch_size)))) * batch_size
    start_ = datetime.now().replace(microsecond=0)
    start_time = time.time()

    game_perf_threshold_min, game_perf_threshold_max = -1.0, 0.9    # currently not used
    adaption_reward = 0
    game_perf = 0
    episode_count = 0

    iteration = start_iter
    while iteration < args.num_iterations:
        start_time_iter = time.time()
        runner.trainer.policy.actor.eval()
        runner.trainer.policy.critic.eval()

        if iteration % update_game_freq == 0:
            adaption_reward = 0
            game_perf = 0
            episode_count = 0
            settings = np.random.choice(game_pool)
            game = GrasperGame(settings, args, action_type=action_type, compute_path=True)
            _prepare_game_context(game)

        attacker_runner = context["attacker_runner"]
        if attacker_runner is None:
            raise ValueError("Attacker runner has not been initialized")
        attacker_runner.random_policy()
        attacker_runner.agent = PathAgent(path=[], num_attackers=attacker_runner.num_attackers)

        pool = context["pool"]
        node_embs = context["node_embs"]
        pooled_node_embs = context["pooled_node_embs"]
        Ts = context["Ts"]
        hgs = context["hgs"]
        hgs_batch = context["hgs_batch"]
        if pool is None or Ts is None:
            raise ValueError("Rollout pool has not been initialized")

        Ts_tensor = torch.FloatTensor(Ts).to(args.device)
        pooled_tensor = None
        if not args.use_end_to_end:
            pooled_tensor = torch.FloatTensor(pooled_node_embs).to(args.device)

        def on_reset(_: int):
            attacker_episode_runner = copy.deepcopy(attacker_runner)
            attacker_episode_runner.agent = PathAgent(
                path=[],
                num_attackers=attacker_episode_runner.num_attackers,
            )
            return attacker_episode_runner, None

        def on_step_batch(
            env_ids,
            obs_list,
            info_list,
            attacker_episode_runners,
            __,
        ):
            attacker_actions_list = []
            defender_actions_list = []
            pre_data_list = []
            for env_id, obs, info, attacker_episode_runner in zip(
                env_ids, obs_list, info_list, attacker_episode_runners
            ):
                attacker_actions = attacker_episode_runner.policy_action(obs, info)
                attacker_actions_list.append(attacker_actions)

                shared_obs, obs_encoded, demo_obs = _encode_obs(
                    obs, info, game, node_embs, pooled_node_embs
                )
                if args.use_emb_layer:
                    shared_obs_tensor = torch.LongTensor(shared_obs).to(args.device)
                    obs_tensor = torch.LongTensor(obs_encoded).to(args.device)
                else:
                    shared_obs_tensor = torch.FloatTensor(shared_obs).to(args.device)
                    obs_tensor = torch.FloatTensor(obs_encoded).to(args.device)

                legal_actions = info.get("defender_legal_action", [])
                if not legal_actions:
                    legal_actions = pool.envs[env_id]._legal_actions(is_attacker=False)
                action_mask = _build_action_mask(legal_actions, runner.env.action_dim)
                action_mask_tensor = torch.BoolTensor(action_mask).to(args.device)

                if args.use_end_to_end:
                    values, actions, action_log_probs = runner.trainer.policy.get_actions(
                        hgs_batch,
                        Ts_tensor,
                        shared_obs_tensor,
                        obs_tensor,
                        action_mask=action_mask_tensor,
                        batch=True,
                    )
                else:
                    values, actions, action_log_probs = runner.trainer.policy.get_actions(
                        pooled_tensor,
                        Ts_tensor,
                        shared_obs_tensor,
                        obs_tensor,
                        action_mask=action_mask_tensor,
                        batch=True,
                    )

                values = _t2n(values)
                actions = _t2n(actions)
                action_log_probs = _t2n(action_log_probs)

                action_indices = np.atleast_1d(actions.squeeze(-1)).astype(int).tolist()
                defender_actions = _map_actions(action_indices, legal_actions)
                defender_actions_list.append(defender_actions)

                demo_act_probs = None
                if args.use_act_supervisor:
                    path = getattr(attacker_episode_runner.agent, "path", None)
                    exit_node = path[-1] if path else 0
                    demo_act_probs = get_demonstration(demo_obs, game, exit_node)

                pre_data_list.append(
                    {
                        "shared_obs": shared_obs,
                        "obs": obs_encoded,
                        "values": values,
                        "actions": actions,
                        "action_log_probs": action_log_probs,
                        "demo_act_probs": demo_act_probs,
                        "action_mask": action_mask,
                    }
                )
            return attacker_actions_list, defender_actions_list, pre_data_list

        def on_step_post(
            _env_id,
            pre_data,
            next_obs,
            reward,
            done,
            next_info,
            __,
            ___,
        ):
            shared_obs_next, obs_next, _ = _encode_obs(
                next_obs, next_info, game, node_embs, pooled_node_embs
            )
            pre_data.update(
                {
                    "reward": float(reward["defender"]),
                    "done": bool(done),
                    "next_shared_obs": shared_obs_next,
                    "next_obs": obs_next,
                }
            )
            return pre_data

        episode_rewards, episode_step_records = pool.run_episodes_batched(
            num_episodes=batch_size,
            on_reset=on_reset,
            on_step_batch=on_step_batch,
            reward_key="defender",
            on_step_post=on_step_post,
        )

        for episode_steps in episode_step_records:
            if not episode_steps:
                continue
            for step in episode_steps:
                reward_val = float(step["reward"])
                rewards = np.full((runner.num_defender, 1), reward_val, dtype=np.float32)
                masks = np.ones((runner.num_defender, 1), dtype=np.float32)
                if step["done"]:
                    masks[:] = 0.0
                if args.use_end_to_end:
                    runner.buffer.insert(
                        hgs,
                        Ts,
                        step["shared_obs"],
                        step["obs"],
                        step["actions"],
                        step["action_log_probs"],
                        step["values"],
                        rewards,
                        masks,
                        step["action_mask"],
                        step["demo_act_probs"],
                    )
                else:
                    runner.buffer.insert(
                        pooled_node_embs,
                        Ts,
                        step["shared_obs"],
                        step["obs"],
                        step["actions"],
                        step["action_log_probs"],
                        step["values"],
                        rewards,
                        masks,
                        step["action_mask"],
                        step["demo_act_probs"],
                        args.use_cache,
                    )

            next_shared_obs = episode_steps[-1]["next_shared_obs"]
            if args.use_emb_layer:
                next_shared_tensor = torch.LongTensor(next_shared_obs).to(args.device)
            else:
                next_shared_tensor = torch.FloatTensor(next_shared_obs).to(args.device)
            if args.use_end_to_end:
                next_values = runner.trainer.policy.get_values(hgs_batch, Ts_tensor, next_shared_tensor, batch=True)
                next_values = _t2n(next_values)
                runner.buffer.compute_returns(next_values, runner.trainer.value_normalizer)
            else:
                next_values = runner.trainer.policy.get_values(pooled_tensor, Ts_tensor, next_shared_tensor, batch=True)
                next_values = _t2n(next_values)
                runner.buffer.compute_returns(next_values, runner.trainer.value_normalizer, args.use_cache)
            if args.use_cache:
                runner.buffer.episode_length_cache.append(len(episode_steps))
            else:
                runner.buffer.episode_length.append(len(episode_steps))

        batch_reward = float(np.sum(episode_rewards)) if episode_rewards else 0.0
        adaption_reward += batch_reward
        game_perf += batch_reward
        episode_count += len(episode_rewards)

        current_episode = iteration + batch_size
        if args.use_cache and current_episode >= args.checkpoint + update_game_freq and current_episode % update_game_freq == 0:
            game_perf /= max(episode_count, 1)
            if game_perf_threshold_min < game_perf < game_perf_threshold_max:
                runner.buffer.store_cache()

        # update the policy
        min_update_eps = min_update_episodes
        if args.checkpoint == 0:
            if update_every_n_episodes is not None:
                # do not entirely clear the buffer after each train step
                if current_episode > min_update_eps and current_episode % update_every_n_episodes == 0:
                    train_infos = runner.train()
                    aloss_list.append(train_infos['value_loss'])
                    vloss_list.append(train_infos['policy_loss'])
            else:
                # entirely clear the buffer after each train step (currently used)
                if current_episode >= min_update_eps and current_episode % min_update_eps == 0:
                    train_infos = runner.train()
                    aloss_list.append(train_infos['value_loss'])
                    vloss_list.append(train_infos['policy_loss'])
        else:
            if update_every_n_episodes is not None:
                # do not entirely clear the buffer after each train step
                if current_episode >= args.checkpoint + min_update_eps and current_episode % update_every_n_episodes == 0:
                    train_infos = runner.train()
                    aloss_list.append(train_infos['value_loss'])
                    vloss_list.append(train_infos['policy_loss'])
            else:
                # entirely clear the buffer after each train step
                if current_episode >= args.checkpoint + min_update_eps and current_episode % min_update_eps == 0:
                    train_infos = runner.train()
                    aloss_list.append(train_infos['value_loss'])
                    vloss_list.append(train_infos['policy_loss'])

        if args.use_cache:
            # currently not used
            if game_perf_threshold_min < game_perf < game_perf_threshold_max:
                print('{}, Iteration: {}/{}, Train Reward: {:.6f}, MEP: {}, Time: {:.6f}'
                      .format(setup_str, current_episode, args.num_iterations, game_perf, args.min_attacker_pth_len, time.time() - start_time_iter))
        else:
            print_every = update_game_freq * 10 if update_game_freq > 1 else 1000
            if current_episode % print_every == 0 or current_episode >= args.num_iterations:
                print('{}, Iteration: {}/{}, Train Reward: {:.6f}, MEP: {}, GPSize: {}, Time: {:.6f}'
                      .format(setup_str, current_episode, args.num_iterations, adaption_reward / max(episode_count, 1),
                              args.min_attacker_pth_len, args.pool_size, time.time() - start_time_iter))

        record_every = update_game_freq if update_game_freq > 1 else 1000
        if current_episode % record_every == 0 or current_episode >= args.num_iterations:
            itera_list.append(current_episode)
            logger.info("Train reward: iter=%s reward=%.6f", current_episode, adaption_reward / max(episode_count, 1))

            end_time = time.time()
            time_list.append(end_time - start_time)
            reward_list.append(adaption_reward / max(episode_count, 1))

        save_res_every = (100 * update_game_freq) if update_game_freq > 1 else 100000
        if iteration == 0 or current_episode % save_res_every == 0 or current_episode >= args.num_iterations:
            pickle.dump({'reward_list': reward_list, 'time_list': time_list, 'aloss_list': aloss_list,
                         'vloss_list': vloss_list, 'itera_list': itera_list},
                        open(get_train_utility_results_dir(game, action_type, args), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if current_episode % args.save_every == 0 or current_episode >= args.num_iterations:
            runner.save(get_mtl_model_results_dir(game, action_type, args, current_episode))
        iteration = current_episode

    if context["pool"] is not None:
        context["pool"].close()

    print("============================================================================================")
    end_ = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_)
    print("Finished training at (GMT) : ", end_)
    print("Total training time  : ", end_ - start_)
    print("============================================================================================")
