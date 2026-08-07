from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from os.path import exists, join

import numpy as np

from graphchase.graph.game_settings import build_game_settings

logger = logging.getLogger(__name__)


class NFSPRunner:
    def __init__(self, args) -> None:
        self.args = args
        self._prepare_device()
        self._seed_everything()

    def _prepare_device(self) -> None:
        use_cuda = bool(self.args.use_cuda)
        device_id = int(self.args.device_id)
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        self.args.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    def _seed_everything(self) -> None:
        seed = int(self.args.seed)
        random.seed(seed)
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self) -> None:
        from graphchase.solver.nfsp import agent as nfsp_agent
        from graphchase.solver.nfsp import buffer as nfsp_buffer
        from graphchase.solver.nfsp import env as nfsp_env
        from graphchase.solver.nfsp import maps as nfsp_maps
        from graphchase.solver.nfsp import model as nfsp_model
        from graphchase.solver.nfsp.run_br import evaluate

        save_path = directory_config(self.args.save_path, self.args.save_folder)
        store_args(self.args, save_path)

        settings = build_game_settings(self.args)
        game_map = nfsp_maps.Maps(settings)
        environment = nfsp_env.Env(settings)

        defender = create_defender(game_map, self.args, nfsp_agent, nfsp_buffer, nfsp_model)
        attacker = create_attacker(game_map, self.args, nfsp_agent, nfsp_buffer, nfsp_model)

        if self.args.exact_br:
            raise ValueError("Exact best response evaluation is not supported in this port.")

        if self.args.attacker_mode == "bandit":
            train_bandit_attacker(
                environment,
                defender,
                attacker,
                evaluate,
                max_episodes=self.args.max_episodes,
                train_br_freq=self.args.train_br_freq,
                train_avg_freq=self.args.train_avg_freq,
                check_freq=self.args.check_freq,
                check_from=self.args.check_from,
                display_freq=self.args.display_freq,
                min_to_train=self.args.min_to_train,
                br_batch_size=self.args.br_batch_size,
                avg_batch_size=self.args.avg_batch_size,
                update_attacker_freq=self.args.display_freq * 10,
                save_path=save_path,
                args=self.args,
            )
        else:
            train(
                environment,
                defender,
                attacker,
                evaluate,
                max_episodes=self.args.max_episodes,
                train_br_freq=self.args.train_br_freq,
                train_avg_freq=self.args.train_avg_freq,
                check_freq=self.args.check_freq,
                check_from=self.args.check_from,
                display_freq=self.args.display_freq,
                min_to_train=self.args.min_to_train,
                br_batch_size=self.args.br_batch_size,
                avg_batch_size=self.args.avg_batch_size,
                save_path=save_path,
            )


def create_defender(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model):
    br_buffer = nfsp_buffer.ReplayBuffer(args.br_buffer_capacity)
    avg_buffer = nfsp_buffer.ReservoirBuffer(args.avg_buffer_capacity)
    if args.defender_rl_mode == "drrn":
        defender_br_net = nfsp_model.DRRN(
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            naive=args.if_naivedrrn,
            num_defender=game_map.num_defender,
            out_mode="rl",
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
        defender_br = nfsp_agent.AgentDRRN(
            defender_br_net,
            br_buffer,
            epsilon_start=0.05,
            epsilon_end=0.001,
            epsilon_decay_duration=args.max_episodes * args.time_horizon * args.br_prob,
            lr=args.br_lr,
            opt_scheduler=False,
            player_idx=0,
            Map=game_map.adjlist,
        )
    elif args.defender_rl_mode == "ma":
        defender_br_net = nfsp_model.AA_MA(
            game_map.max_actions,
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            num_defender=game_map.num_defender,
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
        defender_br = nfsp_agent.AgentMADQN(
            defender_br_net,
            br_buffer,
            epsilon_start=0.05,
            epsilon_end=0.001,
            epsilon_decay_duration=args.max_episodes * args.time_horizon * args.br_prob,
            lr=args.br_lr,
            opt_scheduler=False,
            player_idx=0,
            Map=game_map.adjlist,
        )
    elif args.defender_rl_mode == "aa":
        defender_br_net = nfsp_model.AA_MA(
            pow(game_map.num_nodes + 1, game_map.num_defender),
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            num_defender=game_map.num_defender,
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
        defender_br = nfsp_agent.AgentAADQN(
            defender_br_net,
            br_buffer,
            epsilon_start=0.05,
            epsilon_end=0.001,
            epsilon_decay_duration=args.max_episodes * args.time_horizon * args.br_prob,
            lr=args.br_lr,
            opt_scheduler=False,
            player_idx=0,
            Map=game_map.adjlist,
        )
    else:
        raise ValueError("Unknown defender_rl_mode.")
    if args.br_warmup_path:
        import torch

        defender_br.policy_net.load_state_dict(torch.load(args.br_warmup_path, map_location=args.device))

    if args.defender_sl_mode == "drrn":
        avg_net = nfsp_model.DRRN(
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            naive=args.if_naivedrrn,
            num_defender=game_map.num_defender,
            out_mode="sl",
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
    elif args.defender_sl_mode == "ma":
        avg_net = nfsp_model.AA_MA(
            game_map.max_actions,
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            num_defender=game_map.num_defender,
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
    elif args.defender_sl_mode == "aa":
        avg_net = nfsp_model.AA_MA(
            pow(game_map.num_nodes + 1, game_map.num_defender),
            game_map.num_nodes,
            args.time_horizon,
            args.embedding_size,
            args.hidden_size,
            args.relevant_v_size,
            num_defender=game_map.num_defender,
            seq_mode=args.seq_mode,
            Map=game_map,
            pre_embedding_path=args.pre_embedding_path,
        )
    else:
        raise ValueError("Unknown defender_sl_mode.")
    defender = nfsp_agent.AgentNFSP(defender_br, avg_net, avg_buffer, br_prob=args.br_prob, avg_lr=args.avg_lr, sl_mode=args.defender_sl_mode)
    return defender


def create_attacker(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model):
    if args.attacker_mode == "bandit":
        attacker_br = nfsp_agent.AttackerBandit(
            game_map.exits,
            game_map.attacker_init,
            game_map.adjlist,
            args.time_horizon,
            capacity=int(1e4),
            args=args,
        )
        attacker = nfsp_agent.NFSPAttackerBandit(attacker_br, br_prob=args.br_prob)
    else:
        br_buffer = nfsp_buffer.ReplayBuffer(args.br_buffer_capacity)
        avg_buffer = nfsp_buffer.ReservoirBuffer(args.avg_buffer_capacity)
        if args.attacker_mode == "drrn":
            attacker_br_net = nfsp_model.DRRN(
                game_map.num_nodes,
                args.time_horizon,
                args.embedding_size,
                args.hidden_size,
                args.relevant_v_size,
                naive=args.if_naivedrrn,
                num_defender=None,
                out_mode="rl",
            )
            attacker_br = nfsp_agent.AgentDRRN(
                attacker_br_net,
                br_buffer,
                epsilon_start=0.05,
                epsilon_end=0.001,
                epsilon_decay_duration=args.max_episodes * args.time_horizon * args.br_prob,
                lr=args.br_lr,
                s_q_expl=False,
                opt_scheduler=False,
                player_idx=1,
                Map=game_map.adjlist,
            )
            avg_net = nfsp_model.DRRN(
                game_map.num_nodes,
                args.time_horizon,
                args.embedding_size,
                args.hidden_size,
                args.relevant_v_size,
                naive=args.if_naivedrrn,
                num_defender=None,
                out_mode="sl",
            )
        elif args.attacker_mode == "aa":
            attacker_br_net = nfsp_model.AA_MA(
                game_map.num_nodes + 1,
                game_map.num_nodes,
                args.time_horizon,
                args.embedding_size,
                args.hidden_size,
                args.relevant_v_size,
                num_defender=None,
                seq_mode=args.seq_mode,
                Map=game_map,
            )
            attacker_br = nfsp_agent.AgentAADQN(
                attacker_br_net,
                br_buffer,
                epsilon_start=0.05,
                epsilon_end=0.001,
                epsilon_decay_duration=args.max_episodes * args.time_horizon * args.br_prob,
                lr=args.br_lr,
                opt_scheduler=False,
                player_idx=1,
                Map=game_map.adjlist,
            )
            avg_net = nfsp_model.AA_MA(
                game_map.num_nodes + 1,
                game_map.num_nodes,
                args.time_horizon,
                args.embedding_size,
                args.hidden_size,
                args.relevant_v_size,
                num_defender=None,
                seq_mode=args.seq_mode,
                Map=game_map,
            )
        else:
            raise ValueError("Unknown attacker_mode.")
        attacker = nfsp_agent.AgentNFSP(attacker_br, avg_net, avg_buffer, br_prob=args.br_prob, avg_lr=args.avg_lr, sl_mode=args.attacker_mode)
    return attacker


def directory_config(path, fold_name=None):
    if not exists(path):
        os.makedirs(path)
    if fold_name is None:
        fold_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = join(path, str(fold_name))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(join(save_path, "DEFENDER"), exist_ok=True)
    os.makedirs(join(save_path, "ATTACKER"), exist_ok=True)
    return save_path


def store_args(args, save_path):
    parser_dict = vars(args).copy()
    logger.info("NFSP args: %s", json.dumps(parser_dict, indent=2, default=str))
    with open(join(save_path, "args.json"), "w", encoding="utf-8") as file_obj:
        json.dump(parser_dict, file_obj, indent=2, default=str)


def train(
    environment,
    Defender,
    Attacker,
    evaluate_fn,
    max_episodes=int(5e6),
    train_br_freq=3,
    train_avg_freq=32,
    check_freq=500,
    check_from=0,
    display_freq=1000,
    min_to_train=1000,
    br_batch_size=64,
    avg_batch_size=128,
    exact_br=None,
    save_path="./",
):
    from graphchase.solver.nfsp import buffer as nfsp_buffer

    if exact_br:
        defender_utility = exact_br.Defender_Utility(Defender)
        utilities = [(0, defender_utility)]
        logger.info("Before training, Defender utility is %.4f", defender_utility)
    else:
        running_expl = []
    start_time = time.time()
    for episode in range(1, max_episodes + 1):
        game_state = environment.reset()
        Defender.sample_mode()
        Attacker.sample_mode()
        while not game_state.is_end():
            defender_obs, attacker_obs = game_state.obs()
            def_current_legal_action, att_current_legal_action = game_state.legal_action()

            defender_a = Defender.select_action([defender_obs], [def_current_legal_action], is_evaluation=False)
            attacker_a = Attacker.select_action([attacker_obs], [att_current_legal_action], is_evaluation=False)
            game_state = environment.simu_step(defender_a, attacker_a)

            def_next_obs, att_next_obs = game_state.obs()
            def_reward, att_reward = game_state.reward()
            is_end = game_state.is_end()

            Defender.br_buffer.add(nfsp_buffer.Transition(defender_obs, defender_a, def_reward, def_next_obs, is_end))
            Attacker.br_buffer.add(nfsp_buffer.Transition(attacker_obs, attacker_a, att_reward, att_next_obs, is_end))

            if Defender.is_br:
                Defender.avg_buffer.add(nfsp_buffer.Sample(defender_obs, defender_a))
            if Attacker.is_br:
                Attacker.avg_buffer.add(nfsp_buffer.Sample(attacker_obs, attacker_a))

        if episode % train_br_freq == 0:
            if len(Defender.br_buffer) > min_to_train:
                transitions = Defender.br_buffer.sample(br_batch_size)
                Defender.learning_br_net(transitions)
            if len(Attacker.br_buffer) > min_to_train:
                transitions = Attacker.br_buffer.sample(br_batch_size)
                Attacker.learning_br_net(transitions)

        if episode % train_avg_freq == 0:
            if len(Defender.avg_buffer) > min_to_train:
                samples = Defender.avg_buffer.sample(avg_batch_size)
                Defender.learning_avg_net(samples)
            if len(Attacker.avg_buffer) > min_to_train:
                samples = Attacker.avg_buffer.sample(avg_batch_size)
                Attacker.learning_avg_net(samples)

        if episode % check_freq == 0 and episode >= check_from:
            eps = episode / (time.time() - start_time)
            remain_time = (max_episodes - episode) / 60.0 / eps
            logger.info("Episode: %s, store model. EPS: %.2f, Time left: %.2f min.", episode, eps, remain_time)
            Defender.save_model(join(save_path, "DEFENDER"), episode)
            Attacker.save_model(join(save_path, "ATTACKER"), episode)

        if episode % display_freq == 0:
            eps = episode / (time.time() - start_time)
            remain_time = (max_episodes - episode) / 60.0 / eps
            if exact_br:
                defender_utility = exact_br.Defender_Utility(Defender)
                utilities.append((episode, defender_utility))
                logger.info(
                    "Episode: %s, Defender Utility: %.4f, EPS: %.2f, Time left: %.2f min",
                    episode,
                    defender_utility,
                    eps,
                    remain_time,
                )
                np.save(join(save_path, "DEFENDER_UTILITY.npy"), utilities)
            else:
                Defender.is_br = True
                Attacker.is_br = False
                attacker_avg_return, _ = evaluate_fn(environment, Defender, Attacker, 0, 100)
                log = f"BR Defender return : {attacker_avg_return}\n"
                Defender.is_br = False
                Attacker.is_br = True
                defender_avg_return, _ = evaluate_fn(environment, Defender, Attacker, 1, 100)
                log += f"BR Attacker return : {defender_avg_return}\n"
                log += f"Episode : {episode} , EPS: {eps: .2f}, Time left: {remain_time: .2f} min.\n\n"
                with open(join(save_path, "log.txt"), "a", encoding="utf-8") as file_obj:
                    file_obj.write(log)
                logger.info(log.strip())
                nash_conv = attacker_avg_return + defender_avg_return
                running_expl.append((episode, nash_conv, attacker_avg_return, defender_avg_return))
                np.save(join(save_path, "RunningExpl.npy"), running_expl)


def decay_expl(expl_rate, episode, decay_length):
    start = expl_rate
    end = expl_rate * 0.1
    expl = end + (start - end) * (1 - episode / decay_length)
    return expl


def train_bandit_attacker(
    environment,
    Defender,
    Attacker,
    evaluate_fn,
    max_episodes=int(5e6),
    train_br_freq=3,
    train_avg_freq=32,
    check_freq=500,
    check_from=0,
    display_freq=1000,
    min_to_train=1000,
    br_batch_size=64,
    avg_batch_size=128,
    update_attacker_freq=1000,
    exact_br=None,
    save_path="./",
    args=None,
):
    from graphchase.solver.nfsp import buffer as nfsp_buffer
    from graphchase.solver.nfsp.run_br import evaluate_episode

    class FixedExitAttacker:
        def __init__(self, base_attacker, exit_node):
            self.base_attacker = base_attacker
            self.exit_node = exit_node

        def reset(self):
            paths = self.base_attacker.BrAgent.paths.get(self.exit_node, [])
            if not paths:
                raise ValueError(f"No paths found for attacker exit {self.exit_node}")
            self.base_attacker.BrAgent.selected_exit = self.exit_node
            self.base_attacker.BrAgent.set_path()

        def select_action(self, observation, legal_actions, is_evaluation=True):
            return self.base_attacker.select_action(observation, legal_actions, is_evaluation)

    def _exit_value(environment, defender, attacker, reward_mode, num_episodes):
        if reward_mode == "win_rate":
            wins = 0
            for _ in range(num_episodes):
                attacker_return = evaluate_episode(environment, defender, attacker, 1)
                if attacker_return > 0:
                    wins += 1
            return wins / num_episodes
        attacker_avg_return, _ = evaluate_fn(environment, defender, attacker, 1, num_episodes)
        return attacker_avg_return

    if exact_br:
        defender_utility = exact_br.Defender_Utility(Defender)
        utilities = [(0, defender_utility)]
        logger.info("Before training, Defender utility is %.4f", defender_utility)
    else:
        running_expl = []
    start_time = time.time()
    for episode in range(1, max_episodes + 1):
        game_state = environment.reset()
        Defender.sample_mode(exlp_prob=decay_expl(args.d_expl, episode, 5e6))
        action = Attacker.sample_mode(exlp_prob=args.a_expl)
        attacker_return = 0.0
        while not game_state.is_end():
            defender_obs, attacker_obs = game_state.obs()
            def_current_legal_action, att_current_legal_action = game_state.legal_action()

            defender_a = Defender.select_action([defender_obs], [def_current_legal_action], is_evaluation=False)
            attacker_a = Attacker.select_action([attacker_obs], [att_current_legal_action], is_evaluation=False)
            game_state = environment.simu_step(defender_a, attacker_a)

            def_next_obs, att_next_obs = game_state.obs()
            def_reward, att_reward = game_state.reward()
            is_end = game_state.is_end()
            if not Attacker.is_expl:
                Defender.br_buffer.add(nfsp_buffer.Transition(defender_obs, defender_a, def_reward, def_next_obs, is_end))
            attacker_return += att_reward
            if Defender.is_br:
                Defender.avg_buffer.add(nfsp_buffer.Sample(defender_obs, defender_a))
        if not Defender.is_expl:
            Attacker.BrAgent.update(action, attacker_return)
        if episode % train_br_freq == 0:
            if len(Defender.br_buffer) > min_to_train:
                transitions = Defender.br_buffer.sample(br_batch_size)
                Defender.learning_br_net(transitions)

        if episode % train_avg_freq == 0:
            if len(Defender.avg_buffer) > min_to_train:
                samples = Defender.avg_buffer.sample(avg_batch_size)
                Defender.learning_avg_net(samples)

        if episode % update_attacker_freq == 0:
            Attacker.update_N_a()
            Attacker.save_model(join(save_path, "ATTACKER"), episode)

        if episode % check_freq == 0 and episode >= check_from:
            eps = episode / (time.time() - start_time)
            remain_time = (max_episodes - episode) / 60.0 / eps
            logger.info("Episode: %s, store model. EPS: %.2f, Time left: %.2f min.", episode, eps, remain_time)
            Defender.save_model(join(save_path, "DEFENDER"), episode)

        if episode % display_freq == 0:
            eps = episode / (time.time() - start_time)
            remain_time = (max_episodes - episode) / 60.0 / eps
            if exact_br:
                defender_utility = exact_br.Defender_Utility(Defender)
                utilities.append((episode, defender_utility))
                logger.info(
                    "Episode: %s, Defender Utility: %.4f, EPS: %.2f, Time left: %.2f min",
                    episode,
                    defender_utility,
                    eps,
                    remain_time,
                )
                np.save(join(save_path, "DEFENDER_UTILITY.npy"), utilities)
            else:
                # Defender.set_mode("br")
                # Attacker.set_mode("avg")
                # attacker_avg_return, _ = evaluate_fn(environment, Defender, Attacker, 0, 300)
                # log = f"BR Defender return : {attacker_avg_return}\n"
                # Defender.set_mode("avg")
                # Attacker.set_mode("br")
                # defender_avg_return, _ = evaluate_fn(environment, Defender, Attacker, 1, 300)
                # log += f"BR Attacker return : {defender_avg_return}\n"
                # log += f"Episode : {episode} , EPS: {eps: .2f}, Time left: {remain_time: .2f} min.\n\n"
                # with open(join(save_path, "log.txt"), "a", encoding="utf-8") as file_obj:
                #     file_obj.write(log)
                # logger.info(log.strip())
                # nash_conv = attacker_avg_return + defender_avg_return
                # running_expl.append((episode, nash_conv, attacker_avg_return, defender_avg_return))
                # np.save(join(save_path, "RunningExpl.npy"), running_expl)

                Defender.set_mode("avg")
                reward_mode = args.reward_mode if args and hasattr(args, "reward_mode") else "utility"
                eval_episodes = 1000
                exit_values = []
                candidate_exits = getattr(Attacker.BrAgent, "reachable_exits", Attacker.BrAgent.exits)
                for exit_node in candidate_exits:
                    paths = Attacker.BrAgent.paths.get(exit_node, [])
                    if not paths:
                        logger.warning("Skip exit %s with no feasible paths.", exit_node)
                        continue
                    fixed_attacker = FixedExitAttacker(Attacker, exit_node)
                    exit_value = _exit_value(environment, Defender, fixed_attacker, reward_mode, eval_episodes)
                    exit_values.append((exit_node, exit_value))

                if exit_values:
                    worst_exit, max_exit_value = max(exit_values, key=lambda item: item[1])
                    if reward_mode == "win_rate":
                        defender_wcu = 1 - max_exit_value
                    else:
                        defender_wcu = -max_exit_value
                else:
                    worst_exit = -1
                    max_exit_value = float("nan")
                    defender_wcu = float("nan")

                log = f"Reward mode : {reward_mode}\nAttacker exit values:\n"
                for exit_node, value in exit_values:
                    log += f"Exit {exit_node} value : {value}\n"
                log += f"Worst exit : {worst_exit}, Attacker value : {max_exit_value}\n"
                log += f"Worst-case defender utility : {defender_wcu}\n"
                log += f"Episode : {episode} , EPS: {eps: .2f}, Time left: {remain_time: .2f} min.\n\n"
                with open(join(save_path, "log.txt"), "a", encoding="utf-8") as file_obj:
                    file_obj.write(log)
                logger.info(log.strip())
                running_expl.append((episode, defender_wcu, max_exit_value, worst_exit))
                np.save(join(save_path, "RunningExpl.npy"), running_expl)
