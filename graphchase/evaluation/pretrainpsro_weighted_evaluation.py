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

from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.graph.game_settings import GameSettings
from graphchase.graph.embedding_graph import maybe_train_graph_embeddings
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner
from graphchase.utils import load_yaml_config
from graphchase.build_game_sample import build_symmetric_adjacency, build_asymmetric_adjacency


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrain-PSRO defender strategies on weighted graphs (WCU).")
    parser.add_argument(
        "--config",
        type=str,
        default="graphchase/solver_cfgs/pretrain_psro_cfgs.yaml",
        help="Path to pretrain-PSRO YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set seed=123",
    )
    parser.add_argument(
        "--meta_strategy_iter",
        type=int,
        default=None,
        help="Meta-strategy iteration to load; defaults to latest iteration",
    )
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
    # 775 design weights
    # overrides = [
    #     (9, 8, 1.1),
    #     (8, 15, 1.5),
    #     (15, 22, 0.8),
    #     (9, 2, 0.8),
    #     (2, 3, 1.4),
    #     (3, 4, 1.5),
    #     (25, 18, 1.9),
    #     (18, 11, 2.1),
    #     (11, 4, 1.8),
    #     (25, 24, 2.3),
    #     (24, 23, 2.6),
    #     (23, 22, 1.9),
    # ]

    # 771 design weights
    overrides = [
        (8, 9, 0.3),
        (9, 10, 0.5),
        (10, 11, 0.2),
        (15, 16, 0.7),
        (16, 17, 0.4),
        (17, 18, 0.6),
        (8, 15, 0.5),
        (9, 16, 0.8),
        (10, 17, 0.1),
        (11, 18, 0.4),
        (3, 4, 1.1),
        (4, 11, 4.5),
        (15, 22, 1.2),
        (22, 23, 4.8),
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
        adjacency_matrix, node_ids = build_asymmetric_adjacency(
            base_graph=base_graph,
            seed=weight_seed,
            weight_range=weight_range,
        )
    else:
        if weight_mode != "symmetric":
            logging.warning("Unknown weight_mode %s; defaulting to symmetric.", weight_mode)
        adjacency_matrix, node_ids = build_symmetric_adjacency(
            base_graph=base_graph,
            seed=weight_seed,
            weight_range=weight_range,
        )

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


def build_env_builder(settings: GameSettings) -> callable:
    def _make_env():
        return UNSGEnv(settings)

    return _make_env


def infer_embedding_size(args, graph_embeddings=None) -> int:
    if graph_embeddings:
        first = next(iter(graph_embeddings.values()))
        return int(len(first))
    if not args.graph_embeddings:
        return 1
    return args.emb_size * 2 if args.line_order == "all" else args.emb_size


def compute_input_dim(env: UNSGEnv, embedding_size: int) -> int:
    attacker_count = env.num_attackers
    defender_count = env.num_defenders
    base_len = attacker_count + defender_count
    return base_len * embedding_size + 1


def compute_action_dim(env: UNSGEnv) -> int:
    max_branch = 0
    for node in env.graph.nodes():
        degree = env.graph.degree[node]
        max_branch = max(max_branch, degree + 1)
    return max_branch ** env.num_defenders


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
        logging.warning(
            "Requested meta_strategy_iter %s not found; falling back to latest %s",
            override,
            defender_iters[-1],
        )
    return defender_iters[-1]


def load_meta_strategy(directory: str, iteration: int, strategy_count: int) -> np.ndarray:
    path = os.path.join(directory, f"meta_strategy_iter_{iteration}.npy")
    if not os.path.isfile(path):
        logging.warning("Meta strategy file %s not found; falling back to uniform", path)
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    raw = np.asarray(np.load(path), dtype=float).flatten()
    if raw.shape[0] != strategy_count:
        logging.warning(
            "Meta strategy length %s does not match strategy count %s; defaulting to uniform",
            raw.shape[0],
            strategy_count,
        )
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    raw = np.clip(raw, a_min=0.0, a_max=None)
    total = raw.sum()
    if total <= 0:
        return np.ones(strategy_count, dtype=float) / float(max(strategy_count, 1))
    return raw / total


def load_defender_runners(
    args,
    env_builder,
    input_dim: int,
    action_dim: int,
    device: torch.device,
    defender_dir: str,
    graph_embeddings=None,
    embedding_size: int = 1,
) -> list[DefenderPretrainPsroRunner]:
    runners: list[DefenderPretrainPsroRunner] = []
    if not os.path.isdir(defender_dir):
        raise ValueError(f"Defender directory {defender_dir} does not exist")
    strategy_entries = []
    for name in os.listdir(defender_dir):
        if not name.startswith("strategy_") or not name.endswith(".pt"):
            continue
        try:
            idx = int(name[len("strategy_") : -3])
        except ValueError:
            continue
        strategy_entries.append((idx, name))

    for _, name in sorted(strategy_entries, key=lambda item: item[0]):
        path = os.path.join(defender_dir, name)
        algo = PPOAlgorithm(
            learning_rate=args.ppo_actor_lr,
            critic_learning_rate=args.ppo_critic_lr,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_lambda,
            clip_coef=args.ppo_clip,
            update_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_batch_size,
            entropy_coef=args.entropy_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_vloss=True,
            device=device,
        )
        agent = PPOAgent(input_dim=input_dim, action_dim=action_dim, hidden_dim=args.ppo_hidden_dim, device=device)
        runner = DefenderPretrainPsroRunner(
            env_builder=env_builder,
            agent=agent,
            algorithm=algo,
            metrics_logger=None,
            graph_embeddings=graph_embeddings,
            time_horizon=args.time_horizon,
            embedding_size=embedding_size,
        )
        runner.load(path)
        runner.agent.eval()
        runners.append(runner)
    return runners


def build_candidate_paths(
    settings: GameSettings,
    attacker_path_type: str,
    max_path_length: int | None,
) -> tuple[list[list[int]], dict[int, list[list[int]]], list[int]]:
    graph = settings.graph
    start_node = settings.attacker_init[0] if settings.attacker_init else 0
    exit_nodes = sorted(list(settings.exit_nodes))
    cutoff = max_path_length or settings.time_horizon
    paths: list[list[int]] = []
    paths_by_exit: dict[int, list[list[int]]] = {e: [] for e in exit_nodes}
    for exit_node in exit_nodes:
        try:
            if attacker_path_type == "simple":
                candidate_paths = nx.all_simple_paths(graph, source=start_node, target=exit_node, cutoff=cutoff)
            else:
                candidate_paths = nx.all_shortest_paths(graph, source=start_node, target=exit_node)
            for path in candidate_paths:
                path_list = list(path)
                paths.append(path_list)
                paths_by_exit[exit_node].append(path_list)
        except nx.NetworkXNoPath:
            continue
    if not paths:
        paths.append([start_node])
        for exit_node in exit_nodes:
            paths_by_exit.setdefault(exit_node, [])
    return paths, paths_by_exit, exit_nodes


def _legal_actions_for_nodes(neighbor_map: dict[int, set[int]], nodes: list[int]) -> list[list[int]]:
    actions: list[list[int]] = []
    for node in nodes:
        neighbors = sorted(neighbor_map.get(node, set()))
        actions.append([0] + neighbors)
    return actions


def _build_obs(attacker_nodes: list[int], defender_nodes: list[int]) -> dict:
    attacker_state = np.array([(node, node, 0.0) for node in attacker_nodes], dtype=np.float32)
    defender_state = np.array([(node, node, 0.0) for node in defender_nodes], dtype=np.float32)
    return {"attacker_state": attacker_state, "defender_state": defender_state}


def _defender_present(
    node: int,
    time_point: float,
    dest_nodes: list[int],
    arrival_times: list[float],
    eps: float,
) -> bool:
    for dest, arrival in zip(dest_nodes, arrival_times):
        if dest == node and arrival - time_point <= eps:
            return True
    return False


def _earliest_defender_arrival(
    node: int,
    dest_nodes: list[int],
    arrival_times: list[float],
    after_time: float,
    eps: float,
) -> float | None:
    earliest = None
    for dest, arrival in zip(dest_nodes, arrival_times):
        if dest != node:
            continue
        if arrival - after_time <= eps:
            continue
        if earliest is None or arrival < earliest:
            earliest = arrival
    return earliest


def simulate_weighted_episode(
    settings: GameSettings,
    defender_runner: DefenderPretrainPsroRunner,
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
    attacker_target = None
    attacker_remaining = 0.0

    while current_time < time_horizon - eps:
        obs = _build_obs([attacker_node], defender_nodes)
        info = {
            "cur_time": int(min(current_time, time_horizon)),
            "defender_legal_action": _legal_actions_for_nodes(neighbor_map, defender_nodes),
            "attacker_legal_action": _legal_actions_for_nodes(neighbor_map, [attacker_node]),
        }
        defender_actions = defender_runner.policy_action(obs, info)
        if len(defender_actions) != len(defender_nodes):
            raise ValueError("Defender action length does not match number of defenders.")

        defender_dest_nodes: list[int] = []
        defender_arrivals: list[float] = []
        round_costs: list[float] = []
        for action, start in zip(defender_actions, defender_nodes):
            action_node = int(action)
            if action_node == 0:
                dest = start
                arrival = current_time
                travel_cost = stay_cost
            else:
                if action_node not in neighbor_map.get(start, set()):
                    raise ValueError(f"Defender action {action_node} not neighbor of node {start}.")
                dest = action_node
                travel_cost = float(settings.edge_weight(start, action_node))
                arrival = current_time + travel_cost
            defender_dest_nodes.append(dest)
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
                    next_arrival = _earliest_defender_arrival(
                        attacker_node, defender_dest_nodes, defender_arrivals, time_cursor, eps
                    )
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
                attacker_node = attacker_target
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

    return False


def _reward_from_win(attacker_win: bool, reward_mode: str) -> float:
    if reward_mode == "utility":
        return 1.0 if attacker_win else -1.0
    if reward_mode == "win_rate":
        return 1.0 if attacker_win else 0.0
    raise ValueError("reward_mode must be 'utility' or 'win_rate'")


def _sample_defender(defender_runners: list[DefenderPretrainPsroRunner], probs: np.ndarray) -> DefenderPretrainPsroRunner:
    idx = int(np.random.choice(len(defender_runners), p=probs))
    return defender_runners[idx]


def evaluate_defender_wcu_weighted(
    args,
    settings: GameSettings,
    defender_runners: list[DefenderPretrainPsroRunner],
    defender_meta: np.ndarray,
) -> tuple[float | None, list[float]]:
    paths, paths_by_exit, exit_nodes_ordered = build_candidate_paths(
        settings=settings,
        attacker_path_type=args.attacker_path_type,
        max_path_length=args.time_horizon,
    )
    if not defender_runners:
        raise ValueError("defender_runners must be a non-empty list.")

    probs = np.asarray(defender_meta, dtype=float)
    if probs.shape[0] != len(defender_runners):
        raise ValueError("meta_strategy length must match defender_runners length")
    if np.any(probs < 0):
        raise ValueError("meta_strategy must be non-negative")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("meta_strategy must sum to 1.0")

    reward_mode = "win_rate"
    rollouts = int(args.rollouts_per_attacker_action)
    stay_cost = float(settings.metadata.get("stay_cost", 1.0))
    time_horizon = float(settings.time_horizon)

    action_scores: list[float] = []
    if args.action_type == "exit_node":
        for exit_node in exit_nodes_ordered:
            exit_paths = paths_by_exit.get(exit_node, [])
            if not exit_paths:
                action_scores.append(0.0)
                continue
            total_reward = 0.0
            for _ in range(rollouts):
                defender_runner = _sample_defender(defender_runners, probs)
                chosen_path = exit_paths[int(np.random.choice(len(exit_paths)))]
                attacker_win = simulate_weighted_episode(
                    settings=settings,
                    defender_runner=defender_runner,
                    path=chosen_path,
                    time_horizon=time_horizon,
                    stay_cost=stay_cost,
                )
                total_reward += _reward_from_win(attacker_win, reward_mode)
            action_scores.append(total_reward / float(rollouts))
    else:
        for path in paths:
            total_reward = 0.0
            for _ in range(rollouts):
                defender_runner = _sample_defender(defender_runners, probs)
                attacker_win = simulate_weighted_episode(
                    settings=settings,
                    defender_runner=defender_runner,
                    path=path,
                    time_horizon=time_horizon,
                    stay_cost=stay_cost,
                )
                total_reward += _reward_from_win(attacker_win, reward_mode)
            action_scores.append(total_reward / float(rollouts))

    if not action_scores:
        return None, []
    max_success = max(float(score) for score in action_scores)
    wcu = 1.0 - max_success
    return wcu, action_scores


def main() -> None:
    eval_args = parse_eval_args()
    logging.basicConfig(level=logging.INFO)

    args = load_yaml_config(eval_args.config, build_parser, eval_args.overrides)
    if args.graph_metadata:
        if isinstance(args.graph_metadata, str):
            args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    if getattr(args, "weighted_graph_json", None) is None:
        args.weighted_graph_json = eval_args.weighted_graph_json
    if getattr(args, "weighted_graph_json_out", None) is None:
        args.weighted_graph_json_out = eval_args.weighted_graph_json_out

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")

    graph_embeddings = maybe_train_graph_embeddings(args, logging.getLogger(__name__))

    settings = build_weighted_settings(args)
    env_builder = build_env_builder(settings)
    temp_env = env_builder()
    embedding_size = infer_embedding_size(args, graph_embeddings)
    input_dim = compute_input_dim(temp_env, embedding_size)
    action_dim = compute_action_dim(temp_env)
    temp_env.close()

    save_root = args.save_path
    defender_dir = os.path.join(save_root, "defender")
    defender_runners = load_defender_runners(
        args,
        env_builder,
        input_dim,
        action_dim,
        device,
        defender_dir,
        graph_embeddings=graph_embeddings,
        embedding_size=embedding_size,
    )
    if not defender_runners:
        raise ValueError("No defender strategies were loaded from the training directory")

    defender_iters = list_meta_iterations(defender_dir)
    iteration = choose_iteration(defender_iters, eval_args.meta_strategy_iter)
    defender_meta = load_meta_strategy(defender_dir, iteration, len(defender_runners))
    logging.info("Using meta_strategy_iter=%s (defender=%s)", iteration, len(defender_runners))

    wcu, action_scores = evaluate_defender_wcu_weighted(args, settings, defender_runners, defender_meta)
    if wcu is None:
        logging.warning("Defender WCU could not be computed (empty action scores)")
    else:
        logging.info("Defender WCU (win_rate): %.6f", wcu)
        logging.info("Attacker action scores (win_rate): %s", action_scores)


if __name__ == "__main__":
    main()

# run example:
# python -m graphchase.evaluation.pretrainpsro_weighted_evaluation --config graphchase/solver_cfgs/pretrain_psro_cfgs_7_7_5.yaml  --weighted_graph_json_out graphchase/graph/custom_graph/7_7_5_weighted_graph_configuration.json --set rollouts_per_attacker_action=1000 --set time_horizon=10
