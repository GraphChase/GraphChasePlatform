from __future__ import annotations

import argparse
import os
import pickle
from typing import Any
import random
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

def nudge_close_nodes(pos: dict, min_dist: float, max_iter: int = 80, max_step: float | None = None, seed: int = 0):
    """
    pos: {node: (x,y)}
    min_dist: 认为“重叠/过近”的距离阈值（坐标系单位）
    max_step: 每次推开的最大步长（防止一下子推太远），默认 = min_dist * 0.25
    """
    rnd = random.Random(seed)
    if max_step is None:
        max_step = min_dist * 0.25

    nodes = list(pos.keys())
    xy = {n: [float(pos[n][0]), float(pos[n][1])] for n in nodes}

    for _ in range(max_iter):
        moved_any = False
        # 每一轮只移动发生“过近”的节点；不参与碰撞的节点不会被改动
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

                # 方向：两点重合时随机一个方向
                if dist < 1e-12:
                    ang = rnd.random() * 2 * math.pi
                    ux, uy = math.cos(ang), math.sin(ang)
                    dist = 1e-12
                else:
                    ux, uy = dx / dist, dy / dist

                # 推开量：只推到“刚好不重叠”为止，并限制单次最大步长
                push = min((min_dist - dist) * 0.5, max_step)

                # 只移动这对里“过近”的点（其实两者都过近），其它点不动
                xy[ni][0] += ux * push
                xy[ni][1] += uy * push
                xy[nj][0] -= ux * push
                xy[nj][1] -= uy * push

                # 更新本地缓存，减少误差累积
                xi, yi = xy[ni]

        if not moved_any:
            break

    return {n: (xy[n][0], xy[n][1]) for n in nodes}

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


def describe_object(obj: Any, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Type: {type(obj)}")
    if isinstance(obj, dict):
        print("Dictionary keys:")
        for key in obj.keys():
            print(f"  - {key}")
    else:
        if hasattr(obj, "number_of_nodes") and hasattr(obj, "number_of_edges"):
            print(f"Nodes: {obj.number_of_nodes()}  Edges: {obj.number_of_edges()}")
        else:
            print(obj)


def render_graph(obj: Any, output_name: str, title: str | None = None) -> None:
    if not isinstance(obj, nx.Graph):
        return

    def extract_positions(graph: nx.Graph):
        positions = {}
        for node, data in graph.nodes(data=True):
            if "pos" in data and len(data["pos"]) == 2:
                positions[node] = tuple(data["pos"])
            elif "x" in data and "y" in data:
                positions[node] = (data["x"], data["y"])
        if len(positions) == graph.number_of_nodes():
            return positions
        if positions:
            # fill missing nodes using spring layout but keep existing ones
            missing = [node for node in graph.nodes if node not in positions]
            spring = nx.spring_layout(graph, seed=0)
            for node in missing:
                positions[node] = spring[node]
            return positions
        return nx.spring_layout(graph, seed=0)

    pos = extract_positions(obj)

    # 自动按图的尺度给个 min_dist（你也可以手动调）
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    min_dist = 0.02 * scale   # 0.01~0.04 都可以试试
    pos = nudge_close_nodes(pos, min_dist=min_dist, max_iter=80, seed=0)

    plt.figure(figsize=(10, 10))
    labels = {node: str(node) for node in obj.nodes()}
    nx.draw(
        obj,
        pos=pos,
        node_size=60,
        edge_color="gray",
        node_color="steelblue",
        labels=labels,
        font_size=6,
    )
    out_path = output_name
    if not os.path.splitext(out_path)[1]:
        out_path = f"{out_path}.png"
    if title is None:
        title = os.path.splitext(os.path.basename(out_path))[0]
    plt.title(title)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Graph visualization saved to {out_path}")


def report_node_start(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        print("Node labels start from: (empty graph)")
        return
    numeric_nodes = []
    for node in graph.nodes():
        if isinstance(node, int):
            numeric_nodes.append(node)
        elif isinstance(node, str) and node.isdigit():
            numeric_nodes.append(int(node))
    if len(numeric_nodes) == graph.number_of_nodes():
        print(f"Node labels start from: {min(numeric_nodes)}")
    else:
        print("Node labels start from: (non-numeric labels)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a graph pickle to an image.")
    parser.add_argument("--graph_path", required=True, help="Path to the gpickle/pkl graph file.")
    parser.add_argument("--output_name", required=True, help="Output image filename (png).")
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        print(f"File not found: {args.graph_path}")
        return
    obj = load_graph_file(args.graph_path)
    describe_object(obj, args.graph_path)
    graph_obj = map_info_to_graph(obj)
    if graph_obj is None:
        graph_obj = obj
    if isinstance(graph_obj, nx.Graph):
        report_node_start(graph_obj)
    render_graph(graph_obj, args.output_name)


if __name__ == "__main__":
    main()
