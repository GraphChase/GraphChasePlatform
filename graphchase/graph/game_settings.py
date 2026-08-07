from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class GameSettings:
    graph: nx.Graph
    attacker_init: list[int]
    defender_init: list[int]
    exit_nodes: list[int]
    time_horizon: int
    metadata: dict[str, Any]
    use_weighted_graph: bool = False
    edge_weights: dict[tuple[int, int], float] | None = None
    neighbor_map: dict[int, set[int]] | None = None

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata or {})
        self.graph = self._override_graph_from_metadata(self.graph)
        nodes = set(self.graph.nodes())
        for idx in self.attacker_init + self.defender_init + self.exit_nodes:
            if idx not in nodes:
                raise ValueError(f"Node index {idx} not present in graph")
        if self.edge_weights is None:
            self.edge_weights = self._extract_edge_weights()
        if self.neighbor_map is None:
            self.neighbor_map = {node: set(self.graph.neighbors(node)) for node in self.graph.nodes()}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def neighbors(self, node: int) -> list[int]:
        if self.neighbor_map is not None and node in self.neighbor_map:
            return list(self.neighbor_map[node])
        return list(self.graph.neighbors(node))

    def edge_weight(self, u: int, v: int) -> float:
        if self.edge_weights is None:
            raise ValueError("Edge weights have not been initialized")
        key = (u, v)
        if key not in self.edge_weights:
            raise ValueError(f"Edge ({u}, {v}) does not exist")
        return self.edge_weights[key]

    def to_networkx(self) -> nx.Graph:
        return self.graph

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def _override_graph_from_metadata(self, graph: nx.Graph) -> nx.Graph:
        if not isinstance(self.metadata, dict):
            return graph
        if "adjacency_matrix" not in self.metadata:
            return graph
        adjacency = self.metadata.get("adjacency_matrix")
        node_ids = self.metadata.get("node_ids")
        return self._build_graph_from_adjacency(adjacency, node_ids, graph)

    def _build_graph_from_adjacency(self, adjacency, node_ids, base_graph: nx.Graph) -> nx.Graph:
        if node_ids is None:
            raise ValueError("metadata must include node_ids when adjacency_matrix is provided")
        if not isinstance(node_ids, (list, tuple)) or len(node_ids) == 0:
            raise ValueError("metadata node_ids must be a non-empty list or tuple")
        if adjacency is None or not isinstance(adjacency, (list, tuple)):
            raise ValueError("metadata adjacency_matrix must be a list or tuple")
        n = len(node_ids)
        if len(adjacency) != n:
            raise ValueError("adjacency_matrix dimension does not match node_ids length")
        for row in adjacency:
            if not isinstance(row, (list, tuple)) or len(row) != n:
                raise ValueError("adjacency_matrix must be square and match node_ids length")
        node_set = set(node_ids)
        base_nodes = set(base_graph.nodes())
        if node_set != base_nodes:
            raise ValueError("node_ids must match the nodes present in the provided graph")

        symmetric = True
        for i in range(n):
            for j in range(n):
                if adjacency[i][j] != adjacency[j][i]:
                    symmetric = False
                    break
            if not symmetric:
                break

        new_graph: nx.Graph = nx.Graph() if symmetric else nx.DiGraph()
        base_attrs = dict(base_graph.nodes(data=True))
        for node in node_ids:
            attrs = base_attrs.get(node, {})
            new_graph.add_node(node, **attrs)

        for i, u in enumerate(node_ids):
            row = adjacency[i]
            for j, val in enumerate(row):
                try:
                    weight = float(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"adjacency_matrix entry ({i}, {j}) is not numeric") from exc
                if weight == 0:
                    continue
                if symmetric and j < i:
                    continue
                v = node_ids[j]
                new_graph.add_edge(u, v, weight=weight)
        return new_graph

    def _extract_edge_weights(self, default_weight: float = 1.0) -> dict[tuple[int, int], float]:
        if not self.use_weighted_graph:
            if self.graph.is_directed():
                self.graph = self.graph.to_undirected()
            for u, v, data in self.graph.edges(data=True):
                if isinstance(data, dict):
                    data["weight"] = float(default_weight)
                else:
                    self.graph[u][v]["weight"] = float(default_weight)
        is_directed = self.graph.is_directed()
        weights: dict[tuple[int, int], float] = {}
        for u, v, data in self.graph.edges(data=True):
            if self.use_weighted_graph:
                weight = data.get("weight", default_weight) if isinstance(data, dict) else default_weight
            else:
                weight = default_weight
            w = float(weight)
            weights[(u, v)] = w
            if not is_directed:
                weights[(v, u)] = w
        return weights


def build_game_settings(args) -> GameSettings:
    if args.graph_gpickle_path is None:
        raise ValueError("Please provide a gpickle graph path via --graph_gpickle_path")
    with open(args.graph_gpickle_path, "rb") as fp:
        graph = pickle.load(fp)

    metadata = dict(args.graph_metadata or {})
    metadata.setdefault("source_gpickle", args.graph_gpickle_path)

    return GameSettings(
        graph=graph,
        attacker_init=list(args.attacker_init),
        defender_init=list(args.defender_init),
        exit_nodes=list(args.exit_nodes),
        time_horizon=args.time_horizon,
        metadata=metadata,
        use_weighted_graph=bool(getattr(args, "use_weighted_graph", False)),
    )
