from __future__ import annotations

import argparse
import json


def comma_separated_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(item) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSGNFSP runner")
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

    parser.add_argument("--save_path", type=str, default="./experiments/nsgnfsp")
    parser.add_argument("--save_folder", type=str, default=None)

    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--relevant_v_size", type=int, default=64)
    parser.add_argument("--if_naivedrrn", type=bool, default=False)
    parser.add_argument("--br_buffer_capacity", type=int, default=int(5e5))
    parser.add_argument("--avg_buffer_capacity", type=int, default=int(1e7))
    parser.add_argument("--br_lr", type=float, default=0.0001)
    parser.add_argument("--avg_lr", type=float, default=0.0001)
    parser.add_argument("--d_expl", type=float, default=0.0)
    parser.add_argument("--a_expl", type=float, default=0.1)
    parser.add_argument("--br_prob", type=float, default=0.1)
    parser.add_argument("--seq_mode", type=str, default="cnn")
    parser.add_argument("--pre_embedding_path", type=str, default=None)
    parser.add_argument("--br_warmup_path", type=str, default=None)
    parser.add_argument("--defender_rl_mode", type=str, default="drrn")
    parser.add_argument("--defender_sl_mode", type=str, default="drrn")
    parser.add_argument("--attacker_mode", type=str, default="bandit")
    parser.add_argument("--reward_mode", type=str, choices=["utility", "win_rate"], default="utility")

    parser.add_argument("--br_idx", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=int(1e6))
    parser.add_argument("--train_br_freq", type=int, default=4)
    parser.add_argument("--train_avg_freq", type=int, default=32)
    parser.add_argument("--check_freq", type=int, default=int(1e5))
    parser.add_argument("--check_from", type=int, default=int(1e5))
    parser.add_argument("--display_freq", type=int, default=int(2e3))
    parser.add_argument("--min_to_train", type=int, default=1000)
    parser.add_argument("--br_batch_size", type=int, default=128)
    parser.add_argument("--avg_batch_size", type=int, default=256)
    parser.add_argument("--exact_br", type=bool, default=False)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.graph_metadata:
        args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    return args
