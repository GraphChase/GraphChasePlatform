from __future__ import annotations

import argparse
import os
import pickle
import random
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def build_graph_from_json(json_path: str):
    pass


def build_grid_graph(
    grid_width: int,
    grid_height: int,
    axis_edge_prob: float,
    diag_edge_prob: float,
    seed: int = 0,
):
    rng = random.Random(seed)
    graph = nx.grid_2d_graph(grid_height, grid_width)

    positions = {node: (node[1], -node[0]) for node in graph.nodes()}

    edges_to_remove = []
    for edge in list(graph.edges()):
        if rng.random() > axis_edge_prob:
            edges_to_remove.append(edge)
    if edges_to_remove:
        graph.remove_edges_from(edges_to_remove)

    def neighbors(row: int, col: int):
        deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_height and 0 <= nc < grid_width:
                yield (nr, nc)

    for row in range(grid_height):
        for col in range(grid_width):
            for diag in neighbors(row, col):
                if diag <= (row, col):
                    continue
                if rng.random() <= diag_edge_prob:
                    graph.add_edge((row, col), diag)

    labeled = nx.convert_node_labels_to_integers(
        graph,
        ordering="sorted",
        first_label=1,
        label_attribute="original_label",
    )

    for node, data in labeled.nodes(data=True):
        original = data.pop("original_label")
        labeled.nodes[node]["pos"] = positions[original]

    return labeled


def parse_args():
    parser = argparse.ArgumentParser(description="Generate custom graph structures")
    parser.add_argument("--json_path", type=str, default=None, help="Path to custom graph json file")
    parser.add_argument("--grid_width", type=int, default=None, help="Width of default grid when adjacency not provided")
    parser.add_argument("--grid_height", type=int, default=None, help="Height of default grid when adjacency not provided")
    parser.add_argument("--axis_edge_prob", type=float, default=1.0, help="Probability to keep horizontal/vertical edges")
    parser.add_argument("--diag_edge_prob", type=float, default=0.0, help="Probability to add diagonal edges")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for graph generation")
    parser.add_argument("--output_path", type=str, default="custom_graph.gpickle", help="Output gpickle path")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.json_path is None and (args.grid_width is None or args.grid_height is None):
        raise ValueError("Either provide json_path or both grid_width and grid_height")

    if args.json_path:
        graph = build_graph_from_json(args.json_path)
    else:
        graph = build_grid_graph(
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            axis_edge_prob=args.axis_edge_prob,
            diag_edge_prob=args.diag_edge_prob,
            seed=args.seed,
        )

    with open(args.output_path, "wb") as fp:
        pickle.dump(graph, fp)
    logger.info("Graph saved to %s", os.path.abspath(args.output_path))


if __name__ == "__main__":
    main()
