from __future__ import annotations

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from graphchase.graph.game_settings import build_game_settings
from graphchase.solver.nsgzero.agent import MctsDefender
from graphchase.solver.nsgzero.game import Game
from graphchase.solver_cfgs.nsgzero_cfgs_template import build_parser
from graphchase.utils import load_yaml_config


class ExitPathAttacker:
    def __init__(self, game: Game, exit_node: int, paths_by_exit: dict[int, list[list[int]]], all_paths: list[list[int]]):
        self.game = game
        self.exit_node = exit_node
        self.paths_by_exit = paths_by_exit
        self.all_paths = all_paths
        self.path: list[int] | None = None
        self.t = 1

    def reset(self) -> None:
        candidate_paths = self.paths_by_exit.get(self.exit_node, [])
        if not candidate_paths:
            candidate_paths = self.all_paths
        if not candidate_paths:
            self.path = [self.game.attacker_init[0]]
        else:
            self.path = random.choice(candidate_paths)
        self.t = 1

    def select_act(self, obs=None) -> int:
        if self.path is None:
            raise ValueError("Attacker path not initialized; call reset() first")
        if self.t >= len(self.path):
            act = self.path[-1]
        else:
            act = self.path[self.t]
        self.t += 1
        return int(act)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSGZero defender checkpoints via WCU.")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/nsgzero_cfgs.yaml", help="Path to NSGZero YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint episode to load; defaults to latest")
    parser.add_argument("--rollouts_per_exit", type=int, default=1000, help="Simulations per exit node")
    parser.add_argument("--defender_training_time_horizon", type=int, default=None, help="Time horizon used when training the defender network")
    return parser.parse_args()


def prepare_device(args) -> torch.device:
    use_cuda = bool(args.use_cuda)
    device_id = int(args.device_id)
    if torch.cuda.is_available() and use_cuda:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    args.device = device
    args.cuda = torch.cuda.is_available()
    return device


def resolve_checkpoint(models_root: str, checkpoint: int | None) -> tuple[int, str]:
    if not os.path.isdir(models_root):
        raise ValueError(f"Models directory {models_root} does not exist")
    if checkpoint is not None:
        checkpoint_path = os.path.join(models_root, str(checkpoint))
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint} not found under {models_root}")
        return checkpoint, checkpoint_path
    candidates: list[int] = []
    for name in os.listdir(models_root):
        if name.isdigit():
            candidates.append(int(name))
    if not candidates:
        raise ValueError(f"No checkpoint folders found under {models_root}")
    latest = max(candidates)
    return latest, os.path.join(models_root, str(latest))


def evaluate_exit(
    game: Game,
    defender: MctsDefender,
    exit_node: int,
    paths_by_exit: dict[int, list[list[int]]],
    all_paths: list[list[int]],
    rollouts: int,
) -> tuple[float, float]:
    attacker = ExitPathAttacker(game, exit_node, paths_by_exit, all_paths)
    attacker_wins = 0
    defender_rewards: list[float] = []
    for _ in range(rollouts):
        game.reset()
        defender.reset()
        attacker.reset()
        while not game.current_state.is_end():
            defender_obs, attacker_obs = game.current_state.obs()
            defender_act = defender.select_act(defender_obs, prior=False, temp=1)
            attacker_act = attacker.select_act(attacker_obs)
            game.step(defender_act, attacker_act)
        defender_reward = float(game.current_state.reward()[0])
        defender_rewards.append(defender_reward)
        if defender_reward < 0:
            attacker_wins += 1
    attacker_win_rate = attacker_wins / float(max(rollouts, 1))
    defender_avg = float(np.mean(defender_rewards)) if defender_rewards else 0.0
    return attacker_win_rate, defender_avg


def main() -> None:
    eval_args = parse_eval_args()
    logging.basicConfig(level=logging.INFO)

    args = load_yaml_config(eval_args.config, build_parser, eval_args.overrides)
    if isinstance(args.graph_metadata, str):
        args.graph_metadata = json.loads(args.graph_metadata)
    elif args.graph_metadata is None:
        args.graph_metadata = {}
    if eval_args.defender_training_time_horizon is not None:
        args.defender_training_time_horizon = eval_args.defender_training_time_horizon

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepare_device(args)

    settings = build_game_settings(args)
    game = Game(settings)
    defender = MctsDefender(game, args)

    save_root = args.save_path
    ex_results_path = os.path.join(save_root, args.run_id)
    models_root = os.path.join(ex_results_path, "models")
    checkpoint, checkpoint_path = resolve_checkpoint(models_root, eval_args.checkpoint)
    defender.load_models(checkpoint_path)

    paths_by_exit, all_paths = game.build_paths_by_exit(cutoff=game.time_horizon)

    exit_results: list[tuple[int, int, float, float]] = []
    for exit_node in game.exits:
        attacker_win_rate, defender_avg = evaluate_exit(
            game,
            defender,
            exit_node,
            paths_by_exit,
            all_paths,
            eval_args.rollouts_per_exit,
        )
        actual_exit = game.reverse_node_map.get(exit_node, exit_node)
        exit_results.append((exit_node, actual_exit, attacker_win_rate, defender_avg))
        logging.info(
            "Exit %s (node %s): attacker win_rate=%.4f defender_avg=%.4f",
            exit_node,
            actual_exit,
            attacker_win_rate,
            defender_avg,
        )

    if exit_results:
        max_attacker_win = max(result[2] for result in exit_results)
        defender_wcu = 1 - max_attacker_win
        logging.info("Loaded checkpoint=%s from %s", checkpoint, checkpoint_path)
        logging.info("Defender WCU (win_rate): %.6f", defender_wcu)
    else:
        logging.warning("No exits available to evaluate.")


if __name__ == "__main__":
    main()

# run example:
# python -m graphchase.evaluation.nsgzero_evaluation --config graphchase/solver_cfgs/nsgzero_cfgs_5_5_5.yaml --rollouts_per_exit 1000 --set time_horizon=10 --defender_training_time_horizon 5