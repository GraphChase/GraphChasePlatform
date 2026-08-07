from __future__ import annotations

import argparse
import logging
import pickle
import random

import numpy as np
import torch

from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.graph.game_settings import GameSettings
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.utils import WandbLogger, load_yaml_config, save_experiment_config
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.runners.attacker_weighted_ppo_psro_runner import AttackerWeightedPpoPsroRunner
from graphchase.runners.defender_weighted_ppo_psro_runner import DefenderWeightedPpoPsroRunner
from graphchase.solver.ppo_psro_solver import PpoPsroSolver

logger = logging.getLogger(__name__)


def normalize_graph_edge_weights(graph) -> None:
    weights: list[float] = []
    for _, _, data in graph.edges(data=True):
        if not isinstance(data, dict) or "weight" not in data:
            raise ValueError("Weighted graph must provide a numeric 'weight' attribute on every edge")
        weights.append(float(data["weight"]))
    if not weights:
        raise ValueError("Weighted graph must contain at least one edge")

    mean_weight = float(sum(weights) / len(weights))
    if mean_weight <= 0:
        raise ValueError("Mean edge weight must be positive")

    for _, _, data in graph.edges(data=True):
        data["weight"] = float(data["weight"]) / mean_weight

    logger.info("Normalized graph edge weights by mean %.6f", mean_weight)


def build_env_builder(args):
    with open(args.graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)
    normalize_graph_edge_weights(base_graph)
    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata={"source": args.graph_gpickle_path},
        use_weighted_graph=True,
    )

    def _make_env():
        return UNSGEnv(settings)

    return _make_env


def compute_input_dim(env: UNSGEnv) -> int:
    attacker_count = env.num_attackers
    defender_count = env.num_defenders
    return (attacker_count + defender_count) * 3 + 1


def compute_action_dim(env: UNSGEnv, agent_count: int) -> int:
    max_branch = 0
    for node in env.graph.nodes():
        degree = env.graph.degree[node]
        max_branch = max(max_branch, degree + 1)
    return max_branch ** agent_count


def build_ppo_algorithm(args, device: torch.device) -> PPOAlgorithm:
    return PPOAlgorithm(
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


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run weighted PPO-vs-PPO PSRO with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="graphchase/solver_cfgs/both_ppo_mumbai_weighted.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set seed=123",
    )
    config_args = parser.parse_args()
    args = load_yaml_config(config_args.config, build_parser, config_args.overrides)
    config_path = save_experiment_config(args, args.save_path)
    logger.info("Saved experiment config to %s", config_path)
    logger.info("Weighted PPO runner does not use graph embedding")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")

    metrics_logger = None
    wandb_handle = None
    if args.use_wandb:
        try:
            wandb_handle = WandbLogger(project=args.wandb_project, config={"seed": args.seed, "time_horizon": args.time_horizon})
            metrics_logger = wandb_handle
        except Exception as exc:
            logger.warning("Failed to initialize wandb logging, proceeding without it: %s", exc)

    env_builder = build_env_builder(args)
    temp_env = env_builder()
    input_dim = compute_input_dim(temp_env)
    attacker_action_dim = compute_action_dim(temp_env, temp_env.num_attackers)
    defender_action_dim = compute_action_dim(temp_env, temp_env.num_defenders)
    temp_env.close()

    attacker_agent = PPOAgent(
        input_dim=input_dim,
        action_dim=attacker_action_dim,
        hidden_dim=args.ppo_hidden_dim,
        device=device,
    )
    defender_agent = PPOAgent(
        input_dim=input_dim,
        action_dim=defender_action_dim,
        hidden_dim=args.ppo_hidden_dim,
        device=device,
    )
    attacker_runner = AttackerWeightedPpoPsroRunner(
        env_builder=env_builder,
        agent=attacker_agent,
        algorithm=build_ppo_algorithm(args, device),
        metrics_logger=metrics_logger,
        time_horizon=args.time_horizon,
    )
    defender_runner = DefenderWeightedPpoPsroRunner(
        env_builder=env_builder,
        agent=defender_agent,
        algorithm=build_ppo_algorithm(args, device),
        metrics_logger=metrics_logger,
        time_horizon=args.time_horizon,
    )

    solver = PpoPsroSolver(
        env_builder=env_builder,
        attacker_runner=attacker_runner,
        defender_runner=defender_runner,
        args=args,
    )
    result = solver.solve()

    logger.info("Meta strategies (attacker, defender): %s", result["meta_strategies"])
    logger.info("Meta games (attacker payoff, defender payoff): %s", result["meta_games"])
    logger.info("Saved attacker and defender policies to %s", args.save_path)
    if wandb_handle is not None:
        wandb_handle.finish()


if __name__ == "__main__":
    main()
