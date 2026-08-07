from __future__ import annotations

import logging
import os
import pprint
import random
import time
from multiprocessing import Pipe, Process, set_start_method

import numpy as np

from graphchase.graph.game_settings import build_game_settings
from graphchase.solver.nsgzero import agent
from graphchase.solver.nsgzero.execute import train_execute_episode, test_execute_episode, worker
from graphchase.solver.nsgzero.game import Game
from graphchase.solver.nsgzero.utils import Logger, time_left, time_str

logger = logging.getLogger(__name__)


def run(args) -> None:
    if not hasattr(args, "ex_results_path"):
        raise ValueError("args.ex_results_path must be set before calling run")
    logger.info("Experiment Parameters:\n%s", pprint.pformat(vars(args), indent=4, width=1))

    stats_logger = Logger(logger)
    if args.use_tensorboard:
        stats_logger.setup_tb(args.ex_results_path)

    settings = build_game_settings(args)
    game = Game(settings)

    defender = agent.MctsDefender(game, args)
    if args.att_type == "random":
        attacker = agent.RandomAttacker(game, args)
    elif args.att_type == "nfsp":
        attacker = agent.NFSPAttacker(game, args)
    else:
        raise ValueError(f"Unknown attacker type: {args.att_type}")

    use_multiprocessing = not args.debug_single_process
    worker_count = args.num_workers if use_multiprocessing else 1
    if use_multiprocessing:
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        parent_conns, worker_conns = zip(*[Pipe() for _ in range(args.num_workers)])
        processes = [
            Process(
                target=worker,
                args=(worker_conn, parent_conn, train_execute_episode, test_execute_episode, game, defender, attacker),
            )
            for worker_conn, parent_conn in zip(worker_conns, parent_conns)
        ]
        for process in processes:
            process.daemon = True
            process.start()
    else:
        parent_conns = ()
        processes = []
        logger.info("Debug single-process mode enabled; rollouts run in the main process.")

    start_time = time.time()
    last_time = start_time
    logger.info("Beginning training for %s episodes", args.max_episodes)

    last_train_e = 0
    last_test_e = -args.test_every - 1
    last_save_e = 0
    last_log_e = 0

    e = 0
    while e < args.max_episodes:
        if use_multiprocessing:
            for parent_conn in parent_conns:
                parent_conn.send("train_epi")
                if attacker.require_update:
                    parent_conn.send((attacker.act_est, attacker.N_acts))
            for parent_conn in parent_conns:
                trajectory = parent_conn.recv()
                defender.add_trajectory(trajectory)
                if attacker.require_update:
                    selected_exit, ret, is_br = parent_conn.recv()
                    attacker.update(selected_exit, ret)
                    if is_br:
                        idx = attacker.exits.index(selected_exit)
                        attacker.cache[idx] += 1
        else:
            output = train_execute_episode(game, defender, attacker)
            if attacker.require_update:
                trajectory, a_v = output
            else:
                trajectory = output
                a_v = None
            defender.add_trajectory(trajectory)
            if attacker.require_update:
                selected_exit, ret, is_br = a_v
                attacker.update(selected_exit, ret)
                if is_br:
                    idx = attacker.exits.index(selected_exit)
                    attacker.cache[idx] += 1
        e += worker_count

        if len(defender.buffer) >= args.train_from and (e - last_train_e) / args.train_every >= 1.0:
            v_loss, def_pre_loss, att_pre_loss = defender.learn()
            stats_logger.log_stat("v_loss", v_loss.item(), e)
            stats_logger.log_stat("def_pre_loss", def_pre_loss.item(), e)
            stats_logger.log_stat("att_pre_loss", att_pre_loss.item(), e)
            last_train_e = e

        if (e - last_test_e) / args.test_every >= 1.0:
            logger.info("episodes: %s / %s", e, args.max_episodes)
            last_test_e = e
            total_reward = 0.0
            count = 0
            for _ in range(int(args.test_nepisodes // worker_count)):
                if use_multiprocessing:
                    for parent_conn in parent_conns:
                        parent_conn.send("test_epi")
                    for parent_conn in parent_conns:
                        reward = parent_conn.recv()
                        total_reward += reward
                        count += 1
                else:
                    reward = test_execute_episode(game, defender, attacker, prior=False, temp=1)
                    total_reward += reward
                    count += 1
            if count > 0:
                total_reward /= count
            stats_logger.log_stat("test_return", total_reward, e)

        if args.save_model and (e - last_save_e) / args.save_every >= 1.0:
            last_save_e = e
            save_path = os.path.join(args.ex_results_path, "models", str(e))
            os.makedirs(save_path, exist_ok=True)
            logger.info("Saving models to %s", save_path)
            defender.save_models(save_path)

        if (e - last_log_e) / args.log_every >= 1.0:
            logger.info(
                "Estimated time left: %s. Time passed: %s",
                time_left(last_time, last_log_e, e, args.max_episodes),
                time_str(time.time() - start_time),
            )
            last_time = time.time()
            stats_logger.log_stat("episodes", e, e)
            stats_logger.print_recent_stats()
            last_log_e = e
            if args.att_type == "nfsp":
                prob = attacker.N_acts / attacker.N_acts.sum()
                prob = np.around(prob, decimals=4)
                logger.info("Average Prob: %s", prob)
                logger.info("Action Value Est: %s", attacker.act_est)
                values = [attacker.act_est[key] for key in attacker.exits]
                if args.reward_mode == "win_rate":
                    worst_case = 1 - values[arg_max(values)]
                else:
                    worst_case = -values[arg_max(values)]
                logger.info("Worst Case Est: %s", worst_case)

    if use_multiprocessing:
        for parent_conn in parent_conns:
            parent_conn.send("close")

        for process in processes:
            process.join()


def arg_max(state_action):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)
