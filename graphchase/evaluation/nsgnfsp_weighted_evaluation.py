from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from itertools import product
from os.path import join

import networkx as nx
import numpy as np
import torch

from graphchase.build_game_sample import build_asymmetric_adjacency, build_symmetric_adjacency
from graphchase.graph.game_settings import GameSettings
from graphchase.solver_cfgs.nsgnfsp_cfgs_template import build_parser
from graphchase.utils import load_yaml_config

logger = logging.getLogger(__name__)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSGNFSP checkpoints on weighted graphs (WCU).")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/nsgnfsp_cfgs.yaml", help="Path to NSGNFSP YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--save_path", type=str, required=True, help="Path to saved run directory containing DEFENDER/ATTACKER")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint episode to load; defaults to latest in DEFENDER")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Rollouts per exit node when estimating WCU")
    parser.add_argument("--weighted_graph_json", type=str, default=None, help="Optional JSON file with adjacency_matrix/node_ids for weighted graphs")
    parser.add_argument("--weighted_graph_json_out", type=str, default=None, help="Optional JSON output path to save weighted adjacency")
    return parser.parse_args()


def prepare_device(args) -> None:
    use_cuda = bool(args.use_cuda)
    device_id = int(args.device_id)
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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
    if args.weighted_graph_json:
        weighted_metadata = _load_weighted_graph_json(args.weighted_graph_json)
        metadata.update(weighted_metadata)
    metadata.setdefault("source_gpickle", args.graph_gpickle_path)
    metadata = build_weighted_metadata(base_graph, metadata, args.seed)
    if args.weighted_graph_json_out:
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


def _split_action(action):
    if isinstance(action, tuple):
        if len(action) == 0:
            raise ValueError("Empty action tuple is invalid.")
        if len(action) >= 2:
            return action[0], action[1]
        return action[0], None
    return action, None


def _joint_action_is_legal(action, per_agent_legal: list[list[int]]) -> bool:
    if len(per_agent_legal) != len(list(action)):
        return False
    return all(int(a) in legal for a, legal in zip(action, per_agent_legal))


def _map_branch_to_node_actions(legal_actions: list[list[int]], branch_actions) -> list[int]:
    branch_list = list(branch_actions)
    if len(branch_list) != len(legal_actions):
        raise ValueError(f"Branch action length {len(branch_list)} does not match defender count {len(legal_actions)}")
    node_actions: list[int] = []
    for idx, (branch_idx, acts) in enumerate(zip(branch_list, legal_actions)):
        if branch_idx < 0 or branch_idx >= len(acts):
            raise ValueError(f"Invalid branch index {branch_idx} for defender {idx} with {len(acts)} actions")
        node_actions.append(int(acts[branch_idx]))
    return node_actions


def _resolve_defender_action(action, legal_actions, per_agent_legal: list[list[int]], multi_defender: bool):
    action_value, action_idx = _split_action(action)
    resolved = action_value
    if action_idx is not None:
        idx = int(action_idx)
        if idx < 0 or idx >= len(legal_actions):
            raise ValueError(f"Action index {idx} out of range for legal actions.")
        resolved = legal_actions[idx]
    if isinstance(resolved, int) and resolved not in legal_actions and 0 <= resolved < len(legal_actions):
        resolved = legal_actions[resolved]

    if multi_defender:
        if isinstance(resolved, (list, tuple)):
            if not _joint_action_is_legal(resolved, per_agent_legal):
                resolved = _map_branch_to_node_actions(per_agent_legal, resolved)
            if isinstance(resolved, list):
                resolved = tuple(resolved)
        else:
            raise ValueError("Defender joint action must be a list or tuple.")
    else:
        if isinstance(resolved, (list, tuple)) and len(resolved) == 1:
            resolved = resolved[0]

    if resolved not in legal_actions:
        raise ValueError(f"Resolved action {resolved} is not legal for this state.")
    return resolved


def _defender_legal_actions(map_adjlist: dict[int, list[int]], defender_nodes: list[int], multi_defender: bool) -> tuple[list, list[list[int]]]:
    per_agent = [list(map_adjlist[node]) for node in defender_nodes]
    if multi_defender:
        joint_actions = list(product(*per_agent))
        return joint_actions, per_agent
    return per_agent[0], per_agent


def simulate_weighted_episode(
    settings: GameSettings,
    map_adjlist: dict[int, list[int]],
    defender,
    path: list[int],
    time_horizon: float,
    stay_cost: float,
    eps: float = 1e-8,
) -> bool:
    if len(settings.attacker_init) != 1:
        raise ValueError("Weighted evaluation currently supports a single attacker.")
    if not path:
        raise ValueError("Attacker path must contain at least one node.")

    attacker_node = int(settings.attacker_init[0])
    if path[0] != attacker_node:
        raise ValueError(f"Attacker path must start at {attacker_node}, got {path[0]}.")

    defender_nodes = [int(n) for n in settings.defender_init]
    if attacker_node in defender_nodes:
        return False
    if attacker_node in settings.exit_nodes:
        return True

    neighbor_map = settings.neighbor_map or {node: set(settings.neighbors(node)) for node in settings.graph.nodes()}
    current_time = 0.0
    attacker_index = 0
    attacker_target: int | None = None
    attacker_remaining = 0.0
    attacker_history = [attacker_node]
    multi_defender = len(defender_nodes) > 1

    while current_time < time_horizon - eps:
        defender_obs = (attacker_history, tuple(defender_nodes) if multi_defender else defender_nodes[0])
        defender_legal, per_agent = _defender_legal_actions(map_adjlist, defender_nodes, multi_defender)
        defender_action = defender.select_action([defender_obs], [defender_legal], is_evaluation=True)
        resolved_action = _resolve_defender_action(defender_action, defender_legal, per_agent, multi_defender)

        if multi_defender:
            defender_dest_nodes = [int(node) for node in resolved_action]
        else:
            defender_dest_nodes = [int(resolved_action)]

        defender_arrivals: list[float] = []
        round_costs: list[float] = []
        for dest, start in zip(defender_dest_nodes, defender_nodes):
            if dest == start:
                arrival = current_time
                travel_cost = stay_cost
            else:
                if dest not in neighbor_map.get(start, set()):
                    raise ValueError(f"Defender action {dest} not neighbor of node {start}.")
                travel_cost = float(settings.edge_weight(start, dest))
                arrival = current_time + travel_cost
            defender_arrivals.append(arrival)
            round_costs.append(travel_cost)

        round_duration = max(round_costs) if round_costs else stay_cost
        if round_duration <= 0:
            round_duration = stay_cost
        round_end = min(current_time + round_duration, time_horizon)

        time_cursor = current_time
        while True:
            if attacker_remaining <= eps:
                if _defender_present(attacker_node, time_cursor, defender_dest_nodes, defender_arrivals, eps):
                    return False
                if attacker_node in settings.exit_nodes:
                    return True
                if time_cursor >= round_end - eps:
                    break
                if attacker_index >= len(path) - 1:
                    next_arrival = _earliest_defender_arrival(attacker_node, defender_dest_nodes, defender_arrivals, time_cursor, eps)
                    if next_arrival is not None and next_arrival <= round_end + eps:
                        return False
                    time_cursor = round_end
                    break
                next_node = path[attacker_index + 1]
                attacker_target = next_node
                attacker_remaining = float(settings.edge_weight(attacker_node, next_node))
                if attacker_remaining <= eps:
                    attacker_remaining = 0.0
                    attacker_node = attacker_target
                    attacker_index += 1
                    attacker_target = None
                continue

            arrival_time = time_cursor + attacker_remaining
            if arrival_time <= round_end + eps:
                time_cursor = arrival_time
                attacker_remaining = 0.0
                attacker_node = int(attacker_target)
                attacker_index += 1
                attacker_target = None
                continue
            attacker_remaining -= round_end - time_cursor
            time_cursor = round_end
            break

        current_time = round_end
        if current_time >= time_horizon - eps:
            return False
        defender_nodes = defender_dest_nodes
        attacker_history.append(attacker_node)

    return False


def _attacker_reward(attacker_win: bool, reward_mode: str) -> float:
    if reward_mode == "win_rate":
        return 1.0 if attacker_win else 0.0
    if reward_mode == "utility":
        return 1.0 if attacker_win else -1.0
    raise ValueError("reward_mode must be 'utility' or 'win_rate'")


def compute_exit_value_weighted(
    settings: GameSettings,
    map_adjlist: dict[int, list[int]],
    defender,
    paths: list[list[int]],
    reward_mode: str,
    num_episodes: int,
) -> float:
    if not paths:
        return float("nan")
    stay_cost = float(settings.metadata.get("stay_cost", 1.0))
    time_horizon = float(settings.time_horizon)
    total_reward = 0.0
    for _ in range(num_episodes):
        defender.reset()
        path = random.choice(paths)
        attacker_win = simulate_weighted_episode(settings, map_adjlist, defender, path, time_horizon, stay_cost)
        total_reward += _attacker_reward(attacker_win, reward_mode)
    return total_reward / float(max(num_episodes, 1))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli_args = parse_eval_args()
    args = load_yaml_config(cli_args.config, build_parser, cli_args.overrides)
    prepare_device(args)
    seed_everything(int(args.seed))

    args.weighted_graph_json = cli_args.weighted_graph_json
    args.weighted_graph_json_out = cli_args.weighted_graph_json_out

    if args.graph_metadata:
        if isinstance(args.graph_metadata, str):
            args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}

    if args.attacker_mode != "bandit":
        raise ValueError("This evaluation script currently supports attacker_mode='bandit' only.")

    defender_dir, _ = resolve_save_dirs(cli_args.save_path)
    checkpoint_prefix = resolve_checkpoint(defender_dir, cli_args.checkpoint)

    from graphchase.runners.nfsp_runner import create_attacker, create_defender
    from graphchase.solver.nfsp import agent as nfsp_agent
    from graphchase.solver.nfsp import buffer as nfsp_buffer
    from graphchase.solver.nfsp import maps as nfsp_maps
    from graphchase.solver.nfsp import model as nfsp_model

    settings = build_weighted_settings(args)
    game_map = nfsp_maps.Maps(settings)
    defender = create_defender(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model)
    attacker = create_attacker(game_map, args, nfsp_agent, nfsp_buffer, nfsp_model)

    defender.load_model(defender_dir, checkpoint_prefix)
    checkpoint_label = "latest" if checkpoint_prefix is None else checkpoint_prefix
    logger.info("Loaded defender checkpoint: %s", checkpoint_label)

    defender.set_mode("avg")
    reward_mode = args.reward_mode
    eval_episodes = int(cli_args.eval_episodes)
    exit_values: list[tuple[int, float]] = []
    candidate_exits = getattr(attacker.BrAgent, "reachable_exits", attacker.BrAgent.exits)
    for exit_node in candidate_exits:
        paths = attacker.BrAgent.paths.get(exit_node, [])
        if not paths:
            logger.warning("Skip exit %s with no feasible paths.", exit_node)
            continue
        exit_value = compute_exit_value_weighted(settings, game_map.adjlist, defender, paths, reward_mode, eval_episodes)
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
# python -m graphchase.evaluation.nsgnfsp_weighted_evaluation --config graphchase/solver_cfgs/nsgnfsp_cfgs_7_7_1.yaml --save_path "graphchase/results/nsgnfsp/771" --set time_horizon=10 --weighted_graph_json graphchase/graph/custom_graph/7_7_1_weighted_graph_configuration.json
