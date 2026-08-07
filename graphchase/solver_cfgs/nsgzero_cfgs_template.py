from __future__ import annotations

import argparse
import json


def comma_separated_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(item) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSGZero runner")
    parser.add_argument("--graph_gpickle_path", type=str, default="graphchase/graph/custom_graph/7_7_grid_graph.gpickle", help="Path to gpickle graph file")
    parser.add_argument("--attacker_init", type=comma_separated_ints, default=comma_separated_ints("25"), help="Comma-separated attacker start nodes")
    parser.add_argument("--defender_init", type=comma_separated_ints, default=comma_separated_ints("9,28,44,46"), help="Comma-separated defender start nodes")
    parser.add_argument("--exit_nodes", type=comma_separated_ints, default=comma_separated_ints("4,22,43,49"), help="Comma-separated exit nodes")
    parser.add_argument("--time_horizon", type=int, default=7)
    parser.add_argument("--graph_metadata", type=str, default=None, help="Optional JSON string for custom graph metadata")
    parser.add_argument("--use_weighted_graph", action="store_true", default=False, help="Use edge weights from gpickle graphs")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Enable CUDA (default: on)")
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", help="Disable CUDA even if available")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./experiments/nsgzero")
    parser.add_argument("--run_id", type=str, default="run_0")
    parser.add_argument("--use_tensorboard", action="store_true", default=False, help="Enable TensorBoard logging")
    parser.add_argument("--no_tensorboard", action="store_false", dest="use_tensorboard", help="Disable TensorBoard logging")
    parser.add_argument("--save_model", action="store_true", default=True, help="Save models periodically")
    parser.add_argument("--no_save_model", action="store_false", dest="save_model", help="Disable model saving")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--debug_single_process", action="store_true", default=False, help="Run rollouts in the main process for debugging")
    parser.add_argument("--max_episodes", type=int, default=100000)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0005)

    parser.add_argument("--train_every", type=int, default=16)
    parser.add_argument("--train_from", type=int, default=128)
    parser.add_argument("--test_every", type=int, default=250)
    parser.add_argument("--test_nepisodes", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=500)

    parser.add_argument("--num_sims", type=int, default=15, help="MCTS simulations per action")
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--cpuct", type=float, default=0.3)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward_mode", type=str, choices=["utility", "win_rate"], default="utility")

    parser.add_argument("--att_type", type=str, choices=["random", "nfsp"], default="nfsp")
    parser.add_argument("--ban_capacity", type=int, default=500)
    parser.add_argument("--cache_capacity", type=int, default=20)
    parser.add_argument("--br_rate", type=float, default=0.2)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.graph_metadata:
        args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    return args
