from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch

from graphchase.graph.game_settings import GameSettings
from graphchase.solver.nsgzero.agent import MctsDefender
from graphchase.solver.nsgzero.game import Game
from graphchase.solver_cfgs.nsgzero_cfgs_template import build_parser
from graphchase.utils import load_yaml_config
from graphchase.build_game_sample import build_symmetric_adjacency, build_asymmetric_adjacency


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSGZero defender checkpoints on weighted graphs (WCU).")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/nsgzero_cfgs.yaml", help="Path to NSGZero YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint episode to load; defaults to latest")
    parser.add_argument("--rollouts_per_exit", type=int, default=1000, help="Simulations per exit node")
    parser.add_argument("--defender_training_time_horizon", type=int, default=None, help="Time horizon used when training the defender network")
    parser.add_argument("--weighted_graph_json", type=str, default=None, help="Optional JSON file with adjacency_matrix/node_ids for weighted graphs")
    parser.add_argument("--weighted_graph_json_out", type=str, default=None, help="Optional JSON output path to save weighted adjacency")
    return parser.parse_args()


def _parse_weight_range(raw_range, default_range: tuple[float, float]) -> tuple[float, float]:
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        return float(raw_range[0]), float(raw_range[1])
    return default_range


def _load_weighted_graph_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weighted graph JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Weighted graph JSON must contain an object at the top level")
    return data


def _save_weighted_graph_json(path: str, metadata: dict) -> None:
    payload = {}
    for key in ("adjacency_matrix", "node_ids", "weight_mode", "weight_range", "weight_seed", "source_gpickle"):
        if key in metadata:
            payload[key] = metadata[key]
    if "adjacency_matrix" not in payload or "node_ids" not in payload:
        raise ValueError("Weighted graph metadata must include adjacency_matrix and node_ids to save JSON")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def craft_design_weighted(adjacency_matrix: list[list[float]], node_ids: list[int]) -> list[list[float]]:
    overrides = [
        (9, 8, 1.1),
        (8, 15, 1.5),
        (15, 22, 0.8),
        (9, 2, 0.8),
        (2, 3, 1.4),
        (3, 4, 1.5),
        (25, 18, 1.9),
        (18, 11, 2.1),
        (11, 4, 1.8),
        (25, 24, 2.3),
        (24, 23, 2.6),
        (23, 22, 1.9),
    ]
    index = {node: i for i, node in enumerate(node_ids)}
    for u, v, weight in overrides:
        if u not in index or v not in index:
            logging.warning("Override edge (%s, %s) not in node_ids", u, v)
            continue
        i, j = index[u], index[v]
        adjacency_matrix[i][j] = float(weight)
        adjacency_matrix[j][i] = float(weight)
    return adjacency_matrix


def build_weighted_metadata(base_graph: nx.Graph, metadata: dict, seed: int) -> dict:
    if "adjacency_matrix" in metadata and "node_ids" in metadata:
        return metadata

    weight_mode = metadata.get("weight_mode", "symmetric")
    weight_range = _parse_weight_range(metadata.get("weight_range"), (0.5, 3.0))
    weight_seed = int(metadata.get("weight_seed", seed))

    if weight_mode == "asymmetric":
        adjacency_matrix, node_ids = build_asymmetric_adjacency(base_graph=base_graph, seed=weight_seed, weight_range=weight_range)
    else:
        if weight_mode != "symmetric":
            logging.warning("Unknown weight_mode %s; defaulting to symmetric.", weight_mode)
        adjacency_matrix, node_ids = build_symmetric_adjacency(base_graph=base_graph, seed=weight_seed, weight_range=weight_range)

    adjacency_matrix = craft_design_weighted(adjacency_matrix, node_ids)
    new_metadata = dict(metadata)
    new_metadata.update(
        {
            "adjacency_matrix": adjacency_matrix,
            "node_ids": node_ids,
            "weight_mode": weight_mode,
            "weight_range": [float(weight_range[0]), float(weight_range[1])],
            "weight_seed": weight_seed,
        }
    )
    return new_metadata


def build_weighted_settings(args) -> GameSettings:
    with open(args.graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)
    metadata = dict(args.graph_metadata or {})
    if getattr(args, "weighted_graph_json", None):
        weighted_metadata = _load_weighted_graph_json(args.weighted_graph_json)
        metadata.update(weighted_metadata)
    metadata.setdefault("source_gpickle", args.graph_gpickle_path)
    metadata = build_weighted_metadata(base_graph, metadata, args.seed)
    if getattr(args, "weighted_graph_json_out", None):
        _save_weighted_graph_json(args.weighted_graph_json_out, metadata)
    return GameSettings(
        graph=base_graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata=metadata,
        use_weighted_graph=True,
    )


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


def _defender_present(node: int, time_point: float, dest_nodes: list[int], arrival_times: list[float], eps: float) -> bool:
    for dest, arrival in zip(dest_nodes, arrival_times):
        if dest == node and arrival - time_point <= eps:
            return True
    return False


def _earliest_defender_arrival(node: int, dest_nodes: list[int], arrival_times: list[float], after_time: float, eps: float) -> float | None:
    earliest = None
    for dest, arrival in zip(dest_nodes, arrival_times):
        if dest != node:
            continue
        if arrival - after_time <= eps:
            continue
        if earliest is None or arrival < earliest:
            earliest = arrival
    return earliest


def _initial_defender_internal(game: Game) -> list[int]:
    if len(game.defender_init) == 1 and isinstance(game.defender_init[0], tuple):
        return list(game.defender_init[0])
    return [int(node) for node in game.defender_init]


def simulate_weighted_episode(
    settings: GameSettings,
    game: Game,
    defender: MctsDefender,
    path_internal: list[int],
    time_horizon: float,
    stay_cost: float,
    eps: float = 1e-8,
) -> bool:
    if len(settings.attacker_init) != 1:
        raise ValueError("Weighted evaluation currently supports a single attacker.")
    if not path_internal:
        raise ValueError("Attacker path must contain at least one node.")

    attacker_internal = int(game.attacker_init[0])
    if path_internal[0] != attacker_internal:
        raise ValueError(f"Attacker path must start at {attacker_internal}, got {path_internal[0]}.")

    defender_internal_nodes = _initial_defender_internal(game)
    if attacker_internal in defender_internal_nodes:
        return False
    if attacker_internal in game.exits:
        return True

    neighbor_map = settings.neighbor_map or {node: set(settings.neighbors(node)) for node in settings.graph.nodes()}
    current_time = 0.0
    attacker_index = 0
    attacker_target_internal: int | None = None
    attacker_remaining = 0.0
    attacker_his = [attacker_internal]

    attacker_actual = int(game.reverse_node_map[attacker_internal])
    while current_time < time_horizon - eps:
        obs = (attacker_his, tuple(defender_internal_nodes))
        defender_actions_internal = defender.select_act(obs, prior=False, temp=1)
        if len(defender_actions_internal) != len(defender_internal_nodes):
            raise ValueError("Defender action length does not match number of defenders.")

        defender_dest_internal: list[int] = []
        defender_dest_actual: list[int] = []
        defender_arrivals: list[float] = []
        round_costs: list[float] = []
        for action_internal, start_internal in zip(defender_actions_internal, defender_internal_nodes):
            start_actual = int(game.reverse_node_map[start_internal])
            action_actual = int(game.reverse_node_map[action_internal])
            if action_actual == start_actual:
                dest_actual = start_actual
                arrival = current_time
                travel_cost = stay_cost
                dest_internal = start_internal
            else:
                if action_actual not in neighbor_map.get(start_actual, set()):
                    raise ValueError(f"Defender action {action_actual} not neighbor of node {start_actual}.")
                dest_actual = action_actual
                travel_cost = float(settings.edge_weight(start_actual, action_actual))
                arrival = current_time + travel_cost
                dest_internal = action_internal
            defender_dest_internal.append(dest_internal)
            defender_dest_actual.append(dest_actual)
            defender_arrivals.append(arrival)
            round_costs.append(travel_cost)

        round_duration = max(round_costs) if round_costs else stay_cost
        if round_duration <= 0:
            round_duration = stay_cost
        round_end = min(current_time + round_duration, time_horizon)

        time_cursor = current_time
        while True:
            if attacker_remaining <= eps:
                if _defender_present(attacker_actual, time_cursor, defender_dest_actual, defender_arrivals, eps):
                    return False
                if attacker_internal in game.exits:
                    return True
                if time_cursor >= round_end - eps:
                    break
                if attacker_index >= len(path_internal) - 1:
                    next_arrival = _earliest_defender_arrival(attacker_actual, defender_dest_actual, defender_arrivals, time_cursor, eps)
                    if next_arrival is not None and next_arrival <= round_end + eps:
                        return False
                    time_cursor = round_end
                    break
                next_internal = path_internal[attacker_index + 1]
                next_actual = int(game.reverse_node_map[next_internal])
                attacker_target_internal = next_internal
                attacker_remaining = float(settings.edge_weight(attacker_actual, next_actual))
                if attacker_remaining <= eps:
                    attacker_remaining = 0.0
                    attacker_internal = attacker_target_internal
                    attacker_actual = next_actual
                    attacker_index += 1
                    attacker_target_internal = None
                continue

            arrival_time = time_cursor + attacker_remaining
            if arrival_time <= round_end + eps:
                time_cursor = arrival_time
                attacker_remaining = 0.0
                attacker_internal = int(attacker_target_internal)
                attacker_actual = int(game.reverse_node_map[attacker_internal])
                attacker_index += 1
                attacker_target_internal = None
                continue
            attacker_remaining -= round_end - time_cursor
            time_cursor = round_end
            break

        current_time = round_end
        if current_time >= time_horizon - eps:
            return False
        defender_internal_nodes = defender_dest_internal
        defender_actual_nodes = defender_dest_actual
        virtual_step = int(current_time)
        target_len = virtual_step + 1
        while len(attacker_his) < target_len:
            attacker_his.append(attacker_internal)

    return False


def _reward_from_win(attacker_win: bool, reward_mode: str) -> float:
    if reward_mode == "utility":
        return -1.0 if attacker_win else 1.0
    if reward_mode == "win_rate":
        return 0.0 if attacker_win else 1.0
    raise ValueError("reward_mode must be 'utility' or 'win_rate'")


def _select_attacker_path(exit_node: int, paths_by_exit: dict[int, list[list[int]]], all_paths: list[list[int]], start_node: int) -> list[int]:
    candidate_paths = paths_by_exit.get(exit_node, [])
    if not candidate_paths:
        candidate_paths = all_paths
    if not candidate_paths:
        return [start_node]
    return random.choice(candidate_paths)


def evaluate_exit_weighted(
    settings: GameSettings,
    game: Game,
    defender: MctsDefender,
    exit_node: int,
    paths_by_exit: dict[int, list[list[int]]],
    all_paths: list[list[int]],
    rollouts: int,
) -> tuple[float, float]:
    attacker_wins = 0
    defender_rewards: list[float] = []
    reward_mode = getattr(defender.args, "reward_mode", "utility")
    stay_cost = float(settings.metadata.get("stay_cost", 1.0))
    time_horizon = float(settings.time_horizon)
    start_node = int(game.attacker_init[0])
    for _ in range(rollouts):
        defender.reset()
        path = _select_attacker_path(exit_node, paths_by_exit, all_paths, start_node)
        attacker_win = simulate_weighted_episode(
            settings=settings,
            game=game,
            defender=defender,
            path_internal=path,
            time_horizon=time_horizon,
            stay_cost=stay_cost,
        )
        if attacker_win:
            attacker_wins += 1
        defender_rewards.append(_reward_from_win(attacker_win, reward_mode))
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
    if getattr(args, "weighted_graph_json", None) is None:
        args.weighted_graph_json = eval_args.weighted_graph_json
    if getattr(args, "weighted_graph_json_out", None) is None:
        args.weighted_graph_json_out = eval_args.weighted_graph_json_out
    if eval_args.defender_training_time_horizon is not None:
        args.defender_training_time_horizon = eval_args.defender_training_time_horizon

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepare_device(args)

    settings = build_weighted_settings(args)
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
        attacker_win_rate, defender_avg = evaluate_exit_weighted(
            settings,
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
# python -m graphchase.evaluation.nsgzero_weighted_evaluation --config graphchase/solver_cfgs/nsgzero_cfgs_5_5_5.yaml --rollouts_per_exit 1000 --set time_horizon=10 --defender_training_time_horizon 5
