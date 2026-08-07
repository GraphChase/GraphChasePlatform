from __future__ import annotations

import argparse
import json
import os


def comma_separated_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(item) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="New GraphChase Pretrain-PSRO runner")
    # Graph arguments
    parser.add_argument("--graph_gpickle_path", type=str, default="graphchase/graph/custom_graph/7_7_grid_graph.gpickle", help="Path to gpickle graph file")
    parser.add_argument("--attacker_init", type=comma_separated_ints, default=comma_separated_ints("25"), help="Comma-separated attacker start nodes")
    parser.add_argument("--defender_init", type=comma_separated_ints, default=comma_separated_ints("9, 28, 44, 46"), help="Comma-separated defender start nodes")
    parser.add_argument("--exit_nodes", type=comma_separated_ints, default=comma_separated_ints("4, 22, 43, 49"), help="Comma-separated exit nodes")
    parser.add_argument("--time_horizon", type=int, default=7)
    parser.add_argument("--graph_metadata", type=str, default=None, help="Optional JSON string for custom graph metadata")
    parser.add_argument("--use_weighted_graph", action="store_true", default=False, help="Use edge weights from gpickle graphs")

    # Graph embedding arguments
    parser.add_argument("--graph_embeddings", action="store_true", help="Enable graph embedding training/loading")
    parser.add_argument("--load_embeddings", type=str, default=None, help="Optional .pkl to load precomputed embeddings and skip training")
    parser.add_argument("--save_dir", type=str, default=os.path.join("graphchase", "graph", "custom_graph", "graph_embeddings"), help="Directory to store embedding artifacts")
    parser.add_argument("--emb_size", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--node_information_type", choices=["all", "min"], default="all", help="Use all exit distances or minimum only")
    parser.add_argument("--no_normalize_info", action="store_true", help="Disable normalization of node information rows")
    parser.add_argument("--similarity", choices=["cosine", "dot"], default="cosine", help="Similarity type for information proximity")
    parser.add_argument("--line_order", choices=["first", "second", "all"], default="all", help="LINE order to optimize")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per order")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--neg_samples", type=int, default=5, help="Number of negative samples per positive edge")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate")
    parser.add_argument("--load_node_information", type=str, default=None, help="Optional .npy to load node information")
    parser.add_argument("--load_information_proximity", type=str, default=None, help="Optional .npy to load proximity matrix")

    # Base arguments
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Enable CUDA (default: on)")
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", help="Disable CUDA even if available")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./experiments/graphchase_psro1")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="graphchase", help="Weights & Biases project name")

    # PSRO arguments
    parser.add_argument("--num_psro_iteration", type=int, default=20)
    parser.add_argument("--rollouts_per_attacker_action", type=int, default=int(1e4), help="Number of rollouts for attacker BR during PSRO")
    parser.add_argument("--train_defender_batches", type=int, default=630, help="Number of PPO batches for defender BR during PSRO")
    parser.add_argument("--episodes_per_batch", type=int, default=8, help="Episodes collected per PPO batch for defender BR during PSRO")
    parser.add_argument("--vec_envs", type=int, default=16, help="Number of synchronous envs for BR rollouts and defender collection")
    parser.add_argument("--eval_episodes", type=int, default=1000)

    # Pretraining arguments
    parser.add_argument("--pretrain_iterations", type=int, default=0, help="Number of defender pretrain iterations before PSRO")
    parser.add_argument("--pretrain_tasks", type=int, default=30, help="Random attacker policies per pretrain iteration")
    parser.add_argument("--pretrain_episodes_per_task", type=int, default=20, help="Episodes per pretrain task")
    parser.add_argument("--load_pretrained_model", action="store_true", help="Load defender pretrain model before PSRO")
    parser.add_argument("--pretrain_model_path", type=str, default=None, help="Path to load/save defender pretrain model")

    # PPO hyperparameters
    parser.add_argument("--ppo_actor_lr", type=float, default=1e-3)
    parser.add_argument("--ppo_critic_lr", type=float, default=3e-3)
    parser.add_argument("--ppo_gamma", type=float, default=0.99)
    parser.add_argument("--ppo_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_batch_size", type=int, default=32)
    parser.add_argument("--ppo_hidden_dim", type=int, default=128)
    parser.add_argument("--entropy_coef", type=float, default=1e-2)

    # Attacker settings
    parser.add_argument("--action_type", type=str, choices=["exit_node", "all_path"], default="exit_node", help="Attacker action granularity")
    parser.add_argument("--attacker_path_type", type=str, choices=["shortest", "simple"], default="shortest", help="Path generation strategy for attacker")
    parser.add_argument("--strategy_type", type=str, choices=["mix", "greedy"], default="mix", help="Attacker selection strategy")
    parser.add_argument("--reward_mode", type=str, choices=["utility", "win_rate"], default="utility", help="Aggregate attacker BR rollout rewards as utilities or win_rate (map -1 to 0)")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.graph_metadata:
        args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    return args
