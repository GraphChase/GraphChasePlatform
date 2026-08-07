from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

import networkx as nx


def load_graph_file(file_path: str) -> Any:
    if file_path.endswith(".pkl") and not file_path.endswith(".gpickle"):
        try:
            import torch
            return torch.load(file_path, map_location="cpu")
        except Exception:
            with open(file_path, "rb") as fp:
                return pickle.load(fp)
    with open(file_path, "rb") as fp:
        return pickle.load(fp)


def map_info_to_graph(obj: Any) -> nx.Graph | None:
    if isinstance(obj, nx.Graph):
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        edges = obj[0]
        adjlist = obj[1]
        if isinstance(adjlist, (list, tuple)):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(1, len(adjlist) + 1))
            if isinstance(edges, (list, tuple)):
                graph.add_edges_from(edges)
            return graph
    if isinstance(obj, dict):
        graph = nx.DiGraph()
        graph.add_nodes_from(obj.keys())
        for node, neighbors in obj.items():
            if isinstance(neighbors, (list, tuple, set)):
                for neighbor in neighbors:
                    graph.add_edge(node, neighbor)
        if graph.number_of_nodes() > 0:
            return graph
    return None


def load_graph(graph_path: str) -> nx.Graph:
    obj = load_graph_file(graph_path)
    graph_obj = map_info_to_graph(obj)
    if graph_obj is None:
        graph_obj = obj
    if not isinstance(graph_obj, nx.Graph):
        raise TypeError("Loaded object is not a NetworkX graph.")
    return nx.convert_node_labels_to_integers(graph_obj, first_label=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute unweighted shortest path length between two nodes.")
    parser.add_argument("--graph_path", required=True, help="Path to the gpickle/pkl graph file.")
    parser.add_argument("--node1", type=int, required=True, help="Start node id after relabeling from 1.")
    parser.add_argument("--node2", type=int, required=True, help="End node id after relabeling from 1.")
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        print(f"File not found: {args.graph_path}")
        return

    try:
        graph_obj = load_graph(args.graph_path)
    except (TypeError, pickle.UnpicklingError) as exc:
        print(f"Failed to load graph: {exc}")
        return

    if args.node1 not in graph_obj or args.node2 not in graph_obj:
        print("Node id not in graph after relabeling.")
        print(f"Available nodes: 1..{graph_obj.number_of_nodes()}")
        return

    try:
        length = nx.shortest_path_length(graph_obj, source=args.node1, target=args.node2)
    except nx.NetworkXNoPath:
        print(f"No path between node {args.node1} and node {args.node2}.")
        return

    print(f"Shortest path length (unweighted) from {args.node1} to {args.node2}: {length}")


if __name__ == "__main__":
    main()
