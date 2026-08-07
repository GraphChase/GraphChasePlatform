from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import numpy as np
import torch

from graphchase.utils import load_yaml_config
from graphchase.solver_cfgs.grasper_mappo_cfgs_template import build_parser
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.runners.grasper_mappo_runner import GrasperMappoRunner
from graphchase.solver.grasper.utils.game_config import get_game
from graphchase.solver.grasper.grasper_mappo_psro_runner import GrasperMappoPsroRunner


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Grasper MAPPO PSRO defender strategies via WCU.")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/grasper_mappo_cfgs_grid_7_7_1.yaml", help="Path to grasper MAPPO YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--meta_strategy_iter", type=int, default=None, help="Meta-strategy iteration to load; defaults to latest iteration")
    return parser.parse_args()


def list_meta_iterations(directory: str) -> list[int]:
    prefix = "meta_strategy_iter_"
    iters: list[int] = []
    if not os.path.isdir(directory):
        return iters
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".npy"):
            try:
                idx = int(name[len(prefix) : -4])
                iters.append(idx)
            except ValueError:
                continue
    return sorted(iters)


def choose_iteration(defender_iters: list[int], override: int | None) -> int:
    if not defender_iters:
        raise ValueError("No meta_strategy_iter files found in defender directory")
    if override is not None and override in defender_iters:
        return override
    if override is not None and override not in defender_iters:
        logging.warning("Requested meta_strategy_iter %s not found; falling back to latest %s", override, defender_iters[-1])
    return defender_iters[-1]


def load_meta_strategy(directory: str, iteration: int, strategy_count: int) -> np.ndarray:
    path = os.path.join(directory, f"meta_strategy_iter_{iteration}.npy")
    if not os.path.isfile(path):
        logging.warning("Meta strategy file %s not found; falling back to uniform", path)
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    raw = np.asarray(np.load(path), dtype=float).flatten()
    if raw.shape[0] != strategy_count:
        logging.warning("Meta strategy length %s does not match strategy count %s; defaulting to uniform", raw.shape[0], strategy_count)
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    raw = np.clip(raw, a_min=0.0, a_max=None)
    total = raw.sum()
    if total <= 0:
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    return raw / total


def infer_meta_strategy_length(directory: str, iteration: int) -> int | None:
    path = os.path.join(directory, f"meta_strategy_iter_{iteration}.npy")
    if not os.path.isfile(path):
        return None
    raw = np.asarray(np.load(path), dtype=float).flatten()
    if raw.size == 0:
        return None
    return int(raw.shape[0])


def list_defender_strategy_paths(defender_dir: str) -> list[tuple[int, str, str]]:
    if not os.path.isdir(defender_dir):
        raise ValueError(f"Defender directory {defender_dir} does not exist")
    strategies: list[tuple[int, str, str]] = []
    for name in sorted(os.listdir(defender_dir)):
        if not name.startswith("strategy_") or not name.endswith("_actor.pt"):
            continue
        prefix = name[: -len("_actor.pt")]
        critic_name = f"{prefix}_critic.pt"
        critic_path = os.path.join(defender_dir, critic_name)
        if not os.path.isfile(critic_path):
            continue
        idx_part = prefix[len("strategy_") :]
        if idx_part.endswith(".pt"):
            idx_part = idx_part[: -3]
        try:
            idx = int(idx_part)
        except ValueError:
            idx = len(strategies)
        actor_path = os.path.join(defender_dir, name)
        strategies.append((idx, actor_path, critic_path))
    strategies.sort(key=lambda item: item[0])
    return strategies


def load_defender_runners(
    args,
    game,
    defender_dir: str,
    max_strategies: int | None = None,
) -> list[GrasperMappoPsroRunner]:
    runners: list[GrasperMappoPsroRunner] = []
    shared_graph_emb_path = os.path.join(defender_dir, "shared_graph_emb.pt")
    strategies = list_defender_strategy_paths(defender_dir)
    if max_strategies is not None:
        if max_strategies <= 0:
            raise ValueError(f"max_strategies must be positive, got {max_strategies}")
        if max_strategies < len(strategies):
            strategies = strategies[:max_strategies]
        elif max_strategies > len(strategies):
            logging.warning(
                "Requested %s strategies but only %s found; loading all available.",
                max_strategies,
                len(strategies),
            )
    for _, actor_path, critic_path in strategies:
        graph_emb_path = None
        if (args.use_node_emb or args.use_augmentation) and not args.use_end_to_end:
            if os.path.isfile(shared_graph_emb_path):
                graph_emb_path = shared_graph_emb_path
            elif args.graph_emb_model_path and os.path.isfile(args.graph_emb_model_path):
                graph_emb_path = args.graph_emb_model_path
            else:
                raise ValueError(
                    f"Graph embedding model missing for defender strategy {actor_path}. "
                    f"Expected {shared_graph_emb_path}.",
                )
        runner_args = copy.deepcopy(args)
        if graph_emb_path is not None:
            runner_args.load_graph_emb_model = True
            runner_args.graph_emb_model_path = graph_emb_path
        runner_mappo_args = argparse.Namespace(**vars(runner_args))
        runner_mappo_args.num_defender = game._defender_num
        runner_mappo_args.device = runner_args.device
        runner = GrasperMappoPsroRunner(runner_mappo_args, runner_args, game)
        actor_state = torch.load(actor_path, map_location=runner.device)
        critic_state = torch.load(critic_path, map_location=runner.device)
        runner.env_runner.trainer_ft.policy.actor.load_state_dict(actor_state)
        runner.env_runner.trainer_ft.policy.critic.load_state_dict(critic_state)
        runner.env_runner.trainer_ft.policy.actor.eval()
        runner.env_runner.trainer_ft.policy.critic.eval()
        runners.append(runner)
    return runners


def evaluate_defender_wcu(
    args,
    env_builder,
    defender_runners: list[GrasperMappoPsroRunner],
    defender_meta: np.ndarray,
) -> tuple[float | None, list[float]]:
    attacker_runner = AttackerPathRunner(
        env_builder=env_builder,
        action_type=args.action_type,
        strategy_type=args.strategy_type,
        max_path_length=args.time_horizon,
        attacker_path_type=args.attacker_path_type,
    )
    attacker_runner.compute_best_response(
        defender_runners,
        config={
            "rollouts_per_path": args.rollouts_per_attacker_action,
            "vec_envs": args.vec_envs,
            "reward_mode": "win_rate",
        },
        meta_strategy=defender_meta,
    )
    wcu = attacker_runner.compute_defender_wcu(reward_mode="win_rate")
    return wcu, list(attacker_runner.action_scores)


def main() -> None:
    eval_args = parse_eval_args()
    logging.basicConfig(level=logging.INFO)

    args = load_yaml_config(eval_args.config, build_parser, eval_args.overrides)
    if args.graph_metadata:
        if isinstance(args.graph_metadata, str):
            args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}

    runner = GrasperMappoRunner(args)
    runner._seed_everything()

    game, action_type, _ = get_game(args)
    runner._prepare_psro_models(game, action_type)

    env_builder = runner._build_env_builder(game)

    defender_dir = os.path.join(args.save_path, "defender")

    defender_iters = list_meta_iterations(defender_dir)
    iteration = choose_iteration(defender_iters, eval_args.meta_strategy_iter)
    strategy_limit = infer_meta_strategy_length(defender_dir, iteration)
    defender_runners = load_defender_runners(args, game, defender_dir, max_strategies=strategy_limit)
    if not defender_runners:
        raise ValueError("No defender strategies were loaded from the training directory")
    defender_meta = load_meta_strategy(defender_dir, iteration, len(defender_runners))
    logging.info("Using meta_strategy_iter=%s (defender=%s)", iteration, len(defender_runners))

    wcu, action_scores = evaluate_defender_wcu(args, env_builder, defender_runners, defender_meta)
    if wcu is None:
        logging.warning("Defender WCU could not be computed (empty action scores)")
    else:
        logging.info("Defender WCU (win_rate): %.6f", wcu)
        logging.info("Attacker action scores (win_rate): %s", action_scores)


if __name__ == "__main__":
    main()

# run example:
# python -m graphchase.evaluation.grasper_mappo_evaluation --config graphchase/solver_cfgs/grasper_mappo_cfgs_grid_7_7_1.yaml  --set rollouts_per_attacker_action=1000 --set time_horizon=7
