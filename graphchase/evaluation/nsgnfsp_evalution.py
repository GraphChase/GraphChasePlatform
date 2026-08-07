from __future__ import annotations

import argparse
import logging
import os
import random
from os.path import join

import numpy as np

from graphchase.graph.game_settings import build_game_settings
from graphchase.solver_cfgs.nsgnfsp_cfgs_template import build_parser
from graphchase.utils import load_yaml_config

logger = logging.getLogger(__name__)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSGNFSP checkpoints with worst-case utility (WCU).")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/nsgnfsp_cfgs.yaml", help="Path to NSGNFSP YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--save_path", type=str, required=True, help="Path to saved run directory containing DEFENDER/ATTACKER")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint episode to load; defaults to latest in DEFENDER")
    return parser.parse_args()


def prepare_device(args) -> None:
    use_cuda = bool(args.use_cuda)
    device_id = int(args.device_id)
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    args.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_save_dirs(save_path: str) -> tuple[str, str]:
    if not os.path.isdir(save_path):
        raise ValueError(f"Save path {save_path} does not exist.")
    defender_dir = join(save_path, "DEFENDER")
    attacker_dir = join(save_path, "ATTACKER")
    if not os.path.isdir(defender_dir):
        raise ValueError(f"DEFENDER directory not found under {save_path}")
    if not os.path.isdir(attacker_dir):
        raise ValueError(f"ATTACKER directory not found under {save_path}")
    return defender_dir, attacker_dir


def resolve_checkpoint(defender_dir: str, checkpoint: int | None) -> str | None:
    if checkpoint is not None:
        prefix = str(checkpoint)
        avg_path = join(defender_dir, f"{prefix}avg_net.pt")
        br_path = join(defender_dir, f"{prefix}br_net.pt")
        if not os.path.isfile(avg_path) or not os.path.isfile(br_path):
            raise ValueError(f"Checkpoint {checkpoint} not found in {defender_dir}")
        return prefix

    numeric_prefixes = []
    has_plain = False
    for name in os.listdir(defender_dir):
        if not name.endswith("avg_net.pt"):
            continue
        prefix = name[: -len("avg_net.pt")]
        if prefix == "":
            if os.path.isfile(join(defender_dir, "br_net.pt")):
                has_plain = True
            continue
        try:
            prefix_id = int(prefix)
        except ValueError:
            continue
        br_path = join(defender_dir, f"{prefix}br_net.pt")
        if os.path.isfile(br_path):
            numeric_prefixes.append(prefix_id)

    if numeric_prefixes:
        return str(max(numeric_prefixes))
    if has_plain:
        return None
    raise ValueError(f"No defender checkpoints found in {defender_dir}")


class FixedExitAttacker:
    def __init__(self, base_attacker, exit_node: int):
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


def compute_exit_value(environment, defender, attacker, reward_mode: str, num_episodes: int) -> float:
    from graphchase.solver.nfsp.run_br import evaluate, evaluate_episode

    if reward_mode == "win_rate":
        wins = 0
        for _ in range(num_episodes):
            attacker_return = evaluate_episode(environment, defender, attacker, 1)
            if attacker_return > 0:
                wins += 1
        return wins / float(num_episodes)
    attacker_avg_return, _ = evaluate(environment, defender, attacker, 1, num_episodes)
    return attacker_avg_return


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli_args = parse_eval_args()
    args = load_yaml_config(cli_args.config, build_parser, cli_args.overrides)
    prepare_device(args)
    seed_everything(int(args.seed))

    if args.attacker_mode != "bandit":
        raise ValueError("This evaluation script currently supports attacker_mode='bandit' only.")

    defender_dir, _ = resolve_save_dirs(cli_args.save_path)
    checkpoint_prefix = resolve_checkpoint(defender_dir, cli_args.checkpoint)

    from graphchase.runners.nfsp_runner import create_attacker, create_defender
    from graphchase.solver.nfsp import agent as nfsp_agent
    from graphchase.solver.nfsp import buffer as nfsp_buffer
    from graphchase.solver.nfsp import env as nfsp_env
    from graphchase.solver.nfsp import maps as nfsp_maps
    from graphchase.solver.nfsp import model as nfsp_model
    print(args.time_horizon)
    settings = build_game_settings(args)
    game_map = nfsp_maps.Maps(settings)
    environment = nfsp_env.Env(settings)
    defender = create_defender(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model)
    attacker = create_attacker(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model)

    defender.load_model(defender_dir, checkpoint_prefix)
    checkpoint_label = "latest" if checkpoint_prefix is None else checkpoint_prefix
    logger.info("Loaded defender checkpoint: %s", checkpoint_label)

    defender.set_mode("avg")
    reward_mode = args.reward_mode
    eval_episodes = 1000
    exit_values: list[tuple[int, float]] = []
    candidate_exits = getattr(attacker.BrAgent, "reachable_exits", attacker.BrAgent.exits)
    for exit_node in candidate_exits:
        paths = attacker.BrAgent.paths.get(exit_node, [])
        if not paths:
            logger.warning("Skip exit %s with no feasible paths.", exit_node)
            continue
        fixed_attacker = FixedExitAttacker(attacker, exit_node)
        exit_value = compute_exit_value(environment, defender, fixed_attacker, reward_mode, eval_episodes)
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
    log += f"Checkpoint : {checkpoint_label}, Episodes : {eval_episodes}\n"
    logger.info(log.strip())


if __name__ == "__main__":
    main()

# run example:
# python -m graphchase.evaluation.nsgnfsp_evalution --config graphchase/solver_cfgs/nsgnfsp_cfgs_7_7_1.yaml --save_path "graphchase/results/nsgnfsp/2026-01-21 02:03:42" --set time_horizon=10
