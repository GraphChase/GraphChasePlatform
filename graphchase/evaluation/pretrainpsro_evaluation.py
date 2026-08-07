from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import numpy as np
import torch

from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.graph.game_settings import GameSettings
from graphchase.graph.embedding_graph import maybe_train_graph_embeddings
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.utils import load_yaml_config


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrain-PSRO defender strategies via WCU.")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/pretrain_psro_cfgs.yaml", help="Path to pretrain-PSRO YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--meta_strategy_iter", type=int, default=None, help="Meta-strategy iteration to load; defaults to latest iteration")
    return parser.parse_args()


def build_env_builder(args) -> callable:
    with open(args.graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)
    metadata = dict(args.graph_metadata or {})
    metadata.setdefault("source_gpickle", args.graph_gpickle_path)
    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata=metadata,
        use_weighted_graph=bool(getattr(args, "use_weighted_graph", False)),
    )

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


def evaluate_defender_wcu(
    args,
    env_builder,
    defender_runners: list[DefenderPretrainPsroRunner],
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")

    graph_embeddings = maybe_train_graph_embeddings(args, logging.getLogger(__name__))

    env_builder = build_env_builder(args)
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

    wcu, action_scores = evaluate_defender_wcu(args, env_builder, defender_runners, defender_meta)
    if wcu is None:
        logging.warning("Defender WCU could not be computed (empty action scores)")
    else:
        logging.info("Defender WCU (win_rate): %.6f", wcu)
        logging.info("Attacker action scores (win_rate): %s", action_scores)


if __name__ == "__main__":
    main()

# run example:
# python -m graphchase.evaluation.pretrainpsro_evaluation --config graphchase/solver_cfgs/pretrain_psro_cfgs_7_7_1.yaml --set rollouts_per_attacker_action=1000 --set time_horizon=7