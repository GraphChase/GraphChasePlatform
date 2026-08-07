from __future__ import annotations

import argparse
import json


def comma_separated_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(item) for item in value.split(",")]


def comma_separated_strings(value: str) -> list[str]:
    if value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CFRMix runner")
    parser.add_argument("--graph_gpickle_path", type=str, default="graphchase/graph/custom_graph/7_7_grid_graph.gpickle", help="Path to gpickle graph file")
    parser.add_argument("--attacker_init", type=comma_separated_ints, default=comma_separated_ints("13"), help="Comma-separated attacker start nodes")
    parser.add_argument("--defender_init", type=comma_separated_ints, default=comma_separated_ints("3,11,15,23"), help="Comma-separated defender start nodes")
    parser.add_argument("--exit_nodes", type=comma_separated_ints, default=comma_separated_ints("1,5,7,9,17,19,21,25"), help="Comma-separated exit nodes")
    parser.add_argument("--time_horizon", type=int, default=4)
    parser.add_argument("--graph_metadata", type=str, default=None, help="Optional JSON string for custom graph metadata")
    parser.add_argument("--use_weighted_graph", action="store_true", default=False, help="Use edge weights from gpickle graphs")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Enable CUDA (default: on)")
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda", help="Disable CUDA even if available")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="graphchase/results/cfrmix")
    parser.add_argument("--run_id", type=str, default="run_0")

    parser.add_argument("--network_dim", type=int, default=32)
    parser.add_argument("--sample_number", type=int, default=20)
    parser.add_argument("--action_number", type=int, default=1000)
    parser.add_argument("--train_epoch", type=int, default=1000)
    parser.add_argument("--attacker_regret_batch_size", type=int, default=32)
    parser.add_argument("--defender_regret_batch_size", type=int, default=512)
    parser.add_argument("--defender_strategy_batch_size", type=int, default=32)
    parser.add_argument("--attacker_regret_lr", type=float, default=0.0015)
    parser.add_argument("--defender_regret_lr", type=float, default=0.0015)
    parser.add_argument("--defender_strategy_lr", type=float, default=0.0015)
    parser.add_argument("--iteration", type=int, default=31)

    parser.add_argument("--regret_file_names", type=comma_separated_strings, default=["cfrmix_regret_attacker.dat", "cfrmix_regret_defender.dat"])
    parser.add_argument("--strategy_file_name", type=str, default="cfrmix_defender_strategy_{:d}.dat")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.graph_metadata:
        args.graph_metadata = json.loads(args.graph_metadata)
    else:
        args.graph_metadata = {}
    return args
