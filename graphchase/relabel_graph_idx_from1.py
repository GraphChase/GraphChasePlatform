from __future__ import annotations

import argparse
import math
import pickle
import random

import networkx as nx


def relabel_graph(graph: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(graph, first_label=1, ordering="sorted")


def nudge_close_nodes(
    pos: dict,
    min_dist: float,
    max_iter: int = 80,
    max_step: float | None = None,
    seed: int = 0,
) -> dict:
    rnd = random.Random(seed)
    if max_step is None:
        max_step = min_dist * 0.25

    nodes = list(pos.keys())
    xy = {n: [float(pos[n][0]), float(pos[n][1])] for n in nodes}

    for _ in range(max_iter):
        moved_any = False
        for i in range(len(nodes)):
            ni = nodes[i]
            xi, yi = xy[ni]
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                xj, yj = xy[nj]
                dx, dy = (xi - xj), (yi - yj)
                dist = math.hypot(dx, dy)

                if dist >= min_dist:
                    continue

                moved_any = True

                if dist < 1e-12:
                    ang = rnd.random() * 2 * math.pi
                    ux, uy = math.cos(ang), math.sin(ang)
                    dist = 1e-12
                else:
                    ux, uy = dx / dist, dy / dist

                push = min((min_dist - dist) * 0.5, max_step)
                xy[ni][0] += ux * push
                xy[ni][1] += uy * push
                xy[nj][0] -= ux * push
                xy[nj][1] -= uy * push
                xi, yi = xy[ni]

        if not moved_any:
            break

    return {n: (xy[n][0], xy[n][1]) for n in nodes}


def extract_positions(graph: nx.Graph) -> dict:
    positions: dict = {}
    for node, data in graph.nodes(data=True):
        if "pos" in data and len(data["pos"]) == 2:
            positions[node] = tuple(data["pos"])
        elif "x" in data and "y" in data:
            positions[node] = (data["x"], data["y"])
    if len(positions) == graph.number_of_nodes():
        return positions
    if positions:
        missing = [node for node in graph.nodes if node not in positions]
        spring = nx.spring_layout(graph, seed=0)
        for node in missing:
            positions[node] = spring[node]
        return positions
    return nx.spring_layout(graph, seed=0)


def update_positions(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        return
    pos = extract_positions(graph)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    if scale <= 0:
        scale = 1.0
    min_dist = 0.02 * scale
    pos = nudge_close_nodes(pos, min_dist=min_dist, max_iter=80, seed=0)
    for node, (x, y) in pos.items():
        graph.nodes[node]["pos"] = (float(x), float(y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel gpickle graph nodes to start at 1.")
    parser.add_argument("--graph_path", required=True, help="Path to the gpickle file to rewrite.")
    args = parser.parse_args()

    with open(args.graph_path, "rb") as file_obj:
        graph = pickle.load(file_obj)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Object in {args.graph_path} is not a networkx Graph.")

    relabeled = relabel_graph(graph)
    update_positions(relabeled)

    with open(args.graph_path, "wb") as file_obj:
        pickle.dump(relabeled, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Relabeled, updated positions, and saved graph to {args.graph_path}")


if __name__ == "__main__":
    main()
