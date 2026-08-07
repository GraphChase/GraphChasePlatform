"""Train node embeddings for custom graphs.

Example:

    python graphchase/graph/embedding_graph.py \
        --graph_path graphchase/graph/custom_graph/7_7_grid_graph.gpickle \
        --exit_nodes 1 49 \
        --emb_size 16 --line_order all --epochs 200 \
        --save_dir graphchase/graph/embeddings

The script stays self-contained inside the graphchase folder and only
depends on standard libraries plus networkx, numpy, and torch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import logging
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train node embeddings for a gpickle graph")
    parser.add_argument("--graph_path", type=str, default="graphchase/graph/custom_graph/7_7_grid_graph.gpickle", help="Path to .gpickle graph")
    parser.add_argument("--exit_nodes", type=int, nargs="+", default=[4, 22, 43, 49], help="List of exit node ids")
    parser.add_argument("--emb_size", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--node_information_type", choices=["all", "min"], default="all", help="Use all exit distances or minimum only")
    parser.add_argument("--no_normalize_info", action="store_true", help="Disable normalization of node information rows")
    parser.add_argument("--similarity", choices=["cosine", "dot"], default="cosine", help="Similarity type for information proximity")
    parser.add_argument("--line_order", choices=["first", "second", "all"], default="all", help="LINE order to optimize")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per order")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--neg_samples", type=int, default=5, help="Number of negative samples per positive edge")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=os.path.join("graphchase", "graph", "custom_graph", "graph_embeddings"), help="Directory to store artifacts")
    parser.add_argument("--load_node_information", type=str, default=None, help="Optional .npy to load node information")
    parser.add_argument("--load_information_proximity", type=str, default=None, help="Optional .npy to load proximity matrix")
    parser.add_argument("--load_embeddings", type=str, default=None, help="Optional .pkl to load precomputed embeddings and skip training")
    return parser.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graph(path: str) -> nx.Graph:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as fp:
        graph = pickle.load(fp)
    if not isinstance(graph, nx.Graph):
        raise ValueError("Loaded object is not a networkx graph")
    return graph


def validate_exit_nodes(graph: nx.Graph, exit_nodes: Iterable[int]) -> list[int]:
    exits = list(exit_nodes)
    missing = [node for node in exits if node not in graph.nodes()]
    if missing:
        raise ValueError(f"Exit nodes not present in graph: {missing}")
    return exits


def create_alias_table(area_ratio: list[float]) -> tuple[list[float], list[int]]:
    size = len(area_ratio)
    accept, alias = [0.0] * size, [0] * size
    small, large = [], []
    area_ratio_scaled = np.array(area_ratio) * size
    for i, prob in enumerate(area_ratio_scaled):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = float(area_ratio_scaled[small_idx])
        alias[small_idx] = large_idx
        area_ratio_scaled[large_idx] = area_ratio_scaled[large_idx] - (1 - area_ratio_scaled[small_idx])
        if area_ratio_scaled[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1.0
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1.0

    return accept, alias


def alias_sample(accept: list[float], alias: list[int]) -> int:
    size = len(accept)
    if size == 0:
        raise ValueError("Alias table is empty.")
    i = int(np.random.random() * size)
    r = np.random.random()
    return i if r < accept[i] else alias[i]


def compute_node_information(graph: nx.Graph, exit_nodes: list[int], info_type: str, normalize: bool) -> np.ndarray:
    nodes = list(sorted(graph.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    info = np.zeros((num_nodes, len(exit_nodes)), dtype=float)

    for node in nodes:
        idx = node_to_idx[node]
        for j, exit_node in enumerate(exit_nodes):
            if nx.has_path(graph, node, exit_node):
                length = nx.shortest_path_length(graph, source=node, target=exit_node)
            else:
                length = math.inf
            info[idx, j] = float(length if math.isfinite(length) else 0.0)

    if info_type == "min":
        min_info = np.min(info, axis=1, keepdims=True)
        info = min_info

    if normalize:
        for i in range(info.shape[0]):
            row = info[i]
            total = np.sum(row)
            if total > 0:
                info[i] = row / total

    return info


def compute_similarity_matrix(info: np.ndarray, similarity: str) -> np.ndarray:
    num_nodes = info.shape[0]
    sim_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    if similarity == "cosine":
        norms = np.linalg.norm(info, axis=1)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if similarity == "cosine":
                denom = norms[i] * norms[j]
                if denom == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(info[i], info[j]) / denom)
            else:
                sim = float(np.dot(info[i], info[j]))
            sim_matrix[i, j] = sim

        row_sum = np.sum(sim_matrix[i])
        if row_sum != 0:
            sim_matrix[i] = (sim_matrix[i] / row_sum) * num_nodes

    return sim_matrix


@dataclass
class LineConfig:
    emb_size: int
    batch_size: int
    epochs: int
    negative_ratio: int
    lr: float
    order: str
    seed: int


class LineEmbeddingTrainer:
    def __init__(self, graph: nx.Graph, proximity: np.ndarray | None, config: LineConfig):
        self.graph = graph.to_directed() if not graph.is_directed() else graph
        self.proximity = proximity
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1e-9

        self.nodes = list(sorted(self.graph.nodes()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)

        self.edges, self.edge_weights = self._build_edges()
        self.node_accept, self.node_alias, self.edge_accept, self.edge_alias = self._prepare_alias_tables()
        self._init_embeddings()

    def _build_edges(self):
        edges = []
        weights = []
        for u, v, data in self.graph.edges(data=True):
            weight = float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0
            edges.append((self.node_to_idx[u], self.node_to_idx[v]))
            weights.append(weight)
        return edges, weights

    def _prepare_alias_tables(self):
        node_degree = np.zeros(self.num_nodes, dtype=float)
        for (u_idx, _), weight in zip(self.edges, self.edge_weights):
            node_degree[u_idx] += weight

        power = 0.75
        degree_sum = sum(math.pow(node_degree[i], power) for i in range(self.num_nodes))
        if degree_sum == 0:
            node_prob = [1.0 / self.num_nodes] * self.num_nodes
        else:
            node_prob = [float(math.pow(node_degree[i], power)) / degree_sum for i in range(self.num_nodes)]
        node_accept, node_alias = create_alias_table(node_prob)

        num_edges = len(self.edge_weights)
        if num_edges == 0:
            return node_accept, node_alias, [], []

        total_weight = float(sum(self.edge_weights))
        if total_weight == 0:
            edge_prob = [1.0 / num_edges] * num_edges
        else:
            edge_prob = [weight * num_edges / total_weight for weight in self.edge_weights]
        edge_accept, edge_alias = create_alias_table(edge_prob)
        return node_accept, node_alias, edge_accept, edge_alias

    def _init_embeddings(self):
        self.first_embeddings = torch.nn.Embedding(self.num_nodes, self.config.emb_size).to(self.device)
        self.second_embeddings = torch.nn.Embedding(self.num_nodes, self.config.emb_size).to(self.device)
        self.context_embeddings = torch.nn.Embedding(self.num_nodes, self.config.emb_size).to(self.device)
        torch.nn.init.xavier_uniform_(self.first_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.second_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.context_embeddings.weight)

        if self.config.order == "first":
            params = list(self.first_embeddings.parameters())
        elif self.config.order == "second":
            params = list(self.second_embeddings.parameters()) + list(self.context_embeddings.parameters())
        else:
            params = list(self.first_embeddings.parameters()) + list(self.second_embeddings.parameters()) + list(
                self.context_embeddings.parameters()
            )
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)

    def _line_loss(self, src_idx, dst_idx, sign):
        first_loss = torch.tensor(0.0, device=self.device)
        second_loss = torch.tensor(0.0, device=self.device)
        if self.config.order in ("first", "all"):
            first_loss = self._pair_loss(
                self.first_embeddings(src_idx),
                self.first_embeddings(dst_idx),
                sign,
            )
        if self.config.order in ("second", "all"):
            second_loss = self._pair_loss(
                self.second_embeddings(src_idx),
                self.context_embeddings(dst_idx),
                sign,
            )
        total_loss = first_loss + second_loss
        return total_loss, first_loss, second_loss

    def _pair_loss(self, src_emb, dst_emb, sign):
        score = torch.sum(src_emb * dst_emb, dim=1)
        return -torch.mean(torch.log(torch.sigmoid(sign * score) + self.eps))

    def _optimize_batch(self, src_idx, dst_idx, sign):
        src = torch.tensor(src_idx, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst_idx, dtype=torch.long, device=self.device)
        sign_tensor = torch.tensor(sign, dtype=torch.float, device=self.device)
        total_loss, first_loss, second_loss = self._line_loss(src, dst, sign_tensor)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return float(total_loss.item()), float(first_loss.item()), float(second_loss.item())

    def _export_embeddings(self):
        first = self.first_embeddings.weight.detach().cpu().numpy()
        second = self.second_embeddings.weight.detach().cpu().numpy()
        if self.config.order == "first":
            weights = first
        elif self.config.order == "second":
            weights = second
        else:
            weights = np.hstack((first, second))
        return {node: weights[idx] for node, idx in self.node_to_idx.items()}

    def train_graph_embedding(self):
        data_size = len(self.edges)
        if data_size == 0:
            return self._export_embeddings()

        batch_size = self.config.batch_size
        negative_ratio = self.config.negative_ratio
        proximity = self.proximity

        steps_per_epoch = math.ceil(data_size / batch_size) * (1 + negative_ratio)
        for epoch in range(self.config.epochs):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            start_time = time.time()
            epoch_loss = 0.0
            epoch_first_loss = 0.0
            epoch_second_loss = 0.0
            step_count = 0
            start_index = 0
            while start_index < data_size:
                end_index = min(start_index + batch_size, data_size)
                batch_indices = shuffle_indices[start_index:end_index].tolist()
                for i, edge_idx in enumerate(batch_indices):
                    if random.random() >= self.edge_accept[edge_idx]:
                        batch_indices[i] = self.edge_alias[edge_idx]
                src_idx = [self.edges[idx][0] for idx in batch_indices]
                dst_idx = [self.edges[idx][1] for idx in batch_indices]
                if proximity is None:
                    weights = np.zeros(len(src_idx), dtype=float)
                else:
                    weights = proximity[np.array(src_idx), np.array(dst_idx)]
                sign = 1.0 + weights
                total_loss, first_loss, second_loss = self._optimize_batch(src_idx, dst_idx, sign)
                epoch_loss += total_loss
                epoch_first_loss += first_loss
                epoch_second_loss += second_loss
                step_count += 1

                for _ in range(negative_ratio):
                    neg_dst = [alias_sample(self.node_accept, self.node_alias) for _ in src_idx]
                    neg_sign = np.full(len(src_idx), -1.0, dtype=float)
                    total_loss, first_loss, second_loss = self._optimize_batch(src_idx, neg_dst, neg_sign)
                    epoch_loss += total_loss
                    epoch_first_loss += first_loss
                    epoch_second_loss += second_loss
                    step_count += 1

                start_index = end_index

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(step_count, 1)
                avg_first_loss = epoch_first_loss / max(step_count, 1)
                avg_second_loss = epoch_second_loss / max(step_count, 1)
                elapsed = int(time.time() - start_time)
                logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)
                logger.info(
                    "%d/%d - %ds - loss: %.4f - first_order_loss: %.4f - second_order_loss: %.4f",
                    steps_per_epoch,
                    steps_per_epoch,
                    elapsed,
                    avg_loss,
                    avg_first_loss,
                    avg_second_loss,
                )

        return self._export_embeddings()


def maybe_train_graph_embeddings(args, logger_override: logging.Logger | None = None):
    logger_to_use = logger_override or logger
    if not args.graph_embeddings:
        return None

    graph = load_graph(args.graph_gpickle_path)
    exit_nodes = validate_exit_nodes(graph, args.exit_nodes)
    if args.load_embeddings is not None:
        try:
            if not os.path.isfile(args.load_embeddings):
                raise FileNotFoundError(f"Embedding file not found at {args.load_embeddings}")
            with open(args.load_embeddings, "rb") as fp:
                graph_embeddings = pickle.load(fp)
        except Exception as exc:
            logger_to_use.warning(
                "Failed to load embeddings from %s (%s). Retraining embeddings.",
                args.load_embeddings,
                exc,
            )
        else:
            logger_to_use.info("Loaded node embeddings from %s", args.load_embeddings)
            return graph_embeddings

    graph_name = os.path.splitext(os.path.basename(args.graph_gpickle_path))[0]
    exit_nodes_label = "_".join(str(node) for node in exit_nodes)
    output_dir = os.path.join(args.save_dir, f"{graph_name}_{exit_nodes_label}")
    os.makedirs(output_dir, exist_ok=True)

    if args.load_node_information:
        node_information = np.load(args.load_node_information)
    else:
        node_information = compute_node_information(
            graph=graph,
            exit_nodes=exit_nodes,
            info_type=args.node_information_type,
            normalize=not args.no_normalize_info,
        )
    save_numpy(node_information, os.path.join(output_dir, "node_information.npy"))

    if args.load_information_proximity:
        proximity = np.load(args.load_information_proximity)
    else:
        proximity = compute_similarity_matrix(node_information, similarity=args.similarity)
    save_numpy(proximity, os.path.join(output_dir, "information_proximity_matrix.npy"))

    line_config = LineConfig(
        emb_size=args.emb_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        negative_ratio=args.neg_samples,
        lr=args.lr,
        order=args.line_order,
        seed=args.seed,
    )

    trainer = LineEmbeddingTrainer(graph=graph, proximity=proximity, config=line_config)
    graph_embeddings = trainer.train_graph_embedding()

    emb_path = os.path.join(output_dir, "node_embeddings.pkl")
    save_embeddings(graph_embeddings, emb_path)
    logger_to_use.info("Saved embeddings to %s", emb_path)

    config_path = os.path.join(output_dir, "embedding_config.json")
    config_dict = asdict(line_config)
    config_dict.update(
        {
            "graph_path": args.graph_gpickle_path,
            "exit_nodes": exit_nodes,
            "node_information_type": args.node_information_type,
            "normalize_info": not args.no_normalize_info,
            "similarity": args.similarity,
        }
    )
    save_config(config_dict, config_path)
    logger_to_use.info("Saved embedding config to %s", config_path)

    return graph_embeddings


def save_numpy(array: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def save_embeddings(embeddings: dict[int, np.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(embeddings, fp)


def save_config(config: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    set_seeds(args.seed)

    graph = load_graph(args.graph_path)
    exit_nodes = validate_exit_nodes(graph, args.exit_nodes)

    graph_name = os.path.splitext(os.path.basename(args.graph_path))[0]
    exit_nodes_label = "_".join(str(node) for node in exit_nodes)
    output_dir = os.path.join(args.save_dir, f"{graph_name}_{exit_nodes_label}")
    os.makedirs(output_dir, exist_ok=True)

    if args.load_embeddings:
        with open(args.load_embeddings, "rb") as fp:
            loaded = pickle.load(fp)
        print(f"Loaded embeddings from {args.load_embeddings}, skipping training.")
        save_embeddings(loaded, os.path.join(output_dir, "node_embeddings.pkl"))
        return

    if args.load_node_information:
        node_information = np.load(args.load_node_information)
    else:
        node_information = compute_node_information(
            graph=graph,
            exit_nodes=exit_nodes,
            info_type=args.node_information_type,
            normalize=not args.no_normalize_info,
        )
    save_numpy(node_information, os.path.join(output_dir, "node_information.npy"))

    if args.load_information_proximity:
        proximity = np.load(args.load_information_proximity)
    else:
        proximity = compute_similarity_matrix(node_information, similarity=args.similarity)
    save_numpy(proximity, os.path.join(output_dir, "information_proximity_matrix.npy"))

    line_config = LineConfig(
        emb_size=args.emb_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        negative_ratio=args.neg_samples,
        lr=args.lr,
        order=args.line_order,
        seed=args.seed,
    )

    trainer = LineEmbeddingTrainer(graph=graph, proximity=proximity, config=line_config)
    embeddings = trainer.train_graph_embedding()

    emb_path = os.path.join(output_dir, "node_embeddings.pkl")
    save_embeddings(embeddings, emb_path)
    print(f"Saved embeddings to {emb_path}")

    config_path = os.path.join(output_dir, "embedding_config.json")
    config_dict = asdict(line_config)
    config_dict.update(
        {
            "graph_path": args.graph_path,
            "exit_nodes": exit_nodes,
            "node_information_type": args.node_information_type,
            "normalize_info": not args.no_normalize_info,
            "similarity": args.similarity,
        }
    )
    save_config(config_dict, config_path)
    print(f"Saved config to {config_path}")


# if __name__ == "__main__":
#     main()
