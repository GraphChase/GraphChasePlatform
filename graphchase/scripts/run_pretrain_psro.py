from __future__ import annotations

import os
import pickle
import random
import logging
import numpy as np
import torch
import argparse
from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.graph.game_settings import GameSettings
from graphchase.graph.embedding_graph import maybe_train_graph_embeddings
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.utils import WandbLogger, load_yaml_config, save_experiment_config
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner
from graphchase.solver.psro_solver import PSRO

logger = logging.getLogger(__name__)


def build_env_builder(args):
    graph_path = args.graph_gpickle_path
    with open(graph_path, "rb") as fp:
        base_graph = pickle.load(fp)
    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata={"source": graph_path},
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
        max_branch = max(max_branch, degree + 1)  # neighbors + stay
    return max_branch ** env.num_defenders


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Pretrain-PSRO with YAML config")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/pretrain_psro_cfgs.yaml", help="Path to YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    config_args = parser.parse_args()
    args = load_yaml_config(config_args.config, build_parser, config_args.overrides)
    if args.pretrain_model_path is None:
        args.pretrain_model_path = os.path.join(args.save_path, "pretrain_model.pt")
    config_path = save_experiment_config(args, args.save_path)
    logger.info("Saved experiment config to %s", config_path)
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
        except Exception as exc:  # best-effort wandb
            logger.warning("Failed to initialize wandb logging, proceeding without it: %s", exc)

    graph_embeddings = maybe_train_graph_embeddings(args, logger)

    env_builder = build_env_builder(args)
    temp_env = env_builder()
    embedding_size = infer_embedding_size(args, graph_embeddings)
    input_dim = compute_input_dim(temp_env, embedding_size)
    action_dim = compute_action_dim(temp_env)
    temp_env.close()

    ppo_algo = PPOAlgorithm(
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

    defender_agent = PPOAgent(input_dim=input_dim, action_dim=action_dim, hidden_dim=args.ppo_hidden_dim, device=device)
    defender_runner = DefenderPretrainPsroRunner(
        env_builder=env_builder,
        agent=defender_agent,
        algorithm=ppo_algo,
        metrics_logger=metrics_logger,
        graph_embeddings=graph_embeddings,
        time_horizon=args.time_horizon,
        embedding_size=embedding_size,
    )

    attacker_runner = AttackerPathRunner(
        env_builder=env_builder,
        action_type=args.action_type,
        strategy_type=args.strategy_type,
        max_path_length=args.time_horizon,
        attacker_path_type=args.attacker_path_type,
        metrics_logger=metrics_logger,
    )

    if args.load_pretrained_model:
        if not args.pretrain_model_path:
            raise ValueError("pretrain_model_path must be set when load_pretrained_model is enabled")
        if not os.path.isfile(args.pretrain_model_path):
            raise FileNotFoundError(f"Pretrain model not found at {args.pretrain_model_path}")
        defender_runner.load(args.pretrain_model_path)
    elif args.pretrain_iterations > 0:
        if not args.pretrain_model_path:
            raise ValueError("pretrain_model_path must be set when pretraining to save the model")
        logger.info(
            "Starting defender pretrain: iterations=%s tasks=%s episodes_per_task=%s",
            args.pretrain_iterations,
            args.pretrain_tasks,
            args.pretrain_episodes_per_task,
        )
        defender_runner.pretrain(
            attacker_runner=attacker_runner,
            num_iterations=args.pretrain_iterations,
            num_tasks=args.pretrain_tasks,
            episodes_per_task=args.pretrain_episodes_per_task,
            vec_envs=args.vec_envs,
            save_path=args.pretrain_model_path,
        )
        logger.info("Saved defender pretrain model to %s", args.pretrain_model_path)

    psro = PSRO(env_builder=env_builder, attacker_runner=attacker_runner, defender_runner=defender_runner, args=args)
    result = psro.solve()

    logger.info("Meta strategies (attacker, defender): %s", result["meta_strategies"])
    logger.info("Meta games (attacker payoff, defender payoff): %s", result["meta_games"])
    logger.info("Saved defender policy to %s", args.save_path)
    if wandb_handle is not None:
        wandb_handle.finish()


if __name__ == "__main__":
    main()
