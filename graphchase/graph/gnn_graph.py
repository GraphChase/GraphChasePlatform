from __future__ import annotations

import os
import pickle
import random
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from graphchase.graph.game_settings import GameSettings


MAX_DEGREES = 30
MAX_PATHS_PER_EXIT = 200
DEFAULT_SAVE_DIR = os.path.join("graphchase", "graph", "pretrain_models", "graph_level")
logger = logging.getLogger(__name__)


@dataclass
class GraphSample:
    adjacency: torch.Tensor
    features: torch.Tensor


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def create_activation(name: str | None) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    if name is None:
        return nn.Identity()
    if name == "elu":
        return nn.ELU()
    raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name: str | None):
    if name == "layernorm":
        return nn.LayerNorm
    if name == "batchnorm":
        return nn.BatchNorm1d
    return nn.Identity


def create_optimizer(opt: str, model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    opt_lower = opt.lower()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        return torch.optim.Adam(model.parameters(), **opt_args)
    if opt_lower == "adamw":
        return torch.optim.AdamW(model.parameters(), **opt_args)
    if opt_lower == "adadelta":
        return torch.optim.Adadelta(model.parameters(), **opt_args)
    if opt_lower == "radam":
        return torch.optim.RAdam(model.parameters(), **opt_args)
    if opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return torch.optim.SGD(model.parameters(), **opt_args)
    raise ValueError(f"Invalid optimizer {opt}")


def _device_from_args(device_id: int) -> torch.device:
    if torch.cuda.is_available() and device_id >= 0:
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm=None, activation=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = norm(out_dim) if norm not in (None, nn.Identity) else None
        self.activation = activation

    def forward(self, adj: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        adj = adj.coalesce()
        deg_out = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
        deg_in = torch.sparse.sum(adj, dim=0).to_dense().clamp(min=1)
        norm_out = deg_out.pow(-0.5).unsqueeze(1)
        norm_in = deg_in.pow(-0.5).unsqueeze(1)
        feat_src = feat * norm_out
        agg = torch.sparse.mm(adj.transpose(0, 1), feat_src)
        out = self.fc(agg)
        out = out * norm_in
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class GCN(nn.Module):
    def __init__(self, in_dim: int, num_hidden: int, out_dim: int, num_layers: int, dropout: float,
                 activation: str = "prelu", norm: str = "layernorm", encoding: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList()
        last_activation = create_activation(activation) if encoding else None
        last_norm = create_norm(norm) if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(in_dim, out_dim, norm=last_norm, activation=last_activation))
        else:
            self.gcn_layers.append(GraphConv(in_dim, num_hidden, norm=create_norm(norm),
                                             activation=create_activation(activation)))
            for _ in range(1, num_layers - 1):
                self.gcn_layers.append(GraphConv(num_hidden, num_hidden, norm=create_norm(norm),
                                                 activation=create_activation(activation)))
            self.gcn_layers.append(GraphConv(num_hidden, out_dim, norm=last_norm, activation=last_activation))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, adj: torch.Tensor, inputs: torch.Tensor, return_hidden: bool = False):
        h = inputs
        hidden_list = []
        for layer in self.gcn_layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(adj, h)
            if self.norms is not None:
                h = self.norms(h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)

    def reset_classifier(self, num_classes: int) -> None:
        self.head = nn.Linear(self.gcn_layers[-1].fc.out_features, num_classes)


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        out_dim: int,
        num_layers: int,
        feat_drop: float = 0.5,
        mask_rate: float = 0.5,
        encoder_type: str = "gcn",
        decoder_type: str = "gcn",
        loss_fn: str = "sce",
        drop_edge_rate: float = 0.0,
        replace_rate: float = 0.0,
        alpha_l: float = 2.0,
        concat_hidden: bool = False,
    ):
        super().__init__()
        self.mask_rate = mask_rate
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.drop_edge_rate = drop_edge_rate
        self.output_hidden_size = num_hidden
        self.concat_hidden = concat_hidden
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate

        self.encoder = GCN(in_dim, num_hidden, out_dim, num_layers, feat_drop, encoding=True)
        if decoder_type == "linear":
            self.decoder = nn.Linear(out_dim, in_dim)
        elif decoder_type == "mlp":
            self.decoder = nn.Sequential(nn.Linear(out_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, in_dim))
        else:
            self.decoder = GCN(out_dim, num_hidden, in_dim, 1, feat_drop, encoding=False)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(out_dim * num_layers, out_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(out_dim, out_dim, bias=False)

        self.criterion = self._setup_loss_fn(loss_fn, alpha_l)

    def _setup_loss_fn(self, loss_fn: str, alpha_l: float):
        if loss_fn == "mse":
            return nn.MSELoss()
        if loss_fn == "sce":
            return lambda x, y: sce_loss(x, y, alpha=alpha_l)
        raise NotImplementedError

    def encoding_mask_noise(self, x: torch.Tensor, mask_rate: float):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        return out_x, (mask_nodes, keep_nodes)

    def forward(self, adj: torch.Tensor, x: torch.Tensor):
        loss = self.mask_attr_prediction(adj, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        use_x, (mask_nodes, _) = self.encoding_mask_noise(x, self.mask_rate)
        if self.drop_edge_rate > 0:
            use_adj = drop_edge(adj, self.drop_edge_rate)
        else:
            use_adj = adj

        enc_rep, all_hidden = self.encoder(use_adj, use_x, return_hidden=True)
        if self.concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        if self.decoder_type not in {"mlp", "linear"}:
            rep = rep.clone()
            rep[mask_nodes] = 0

        if self.decoder_type in {"mlp", "linear"}:
            recon = self.decoder(rep)
        else:
            recon = self.decoder(adj, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        if x_init.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        return self.criterion(x_rec, x_init)

    def embed(self, adj: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rep = self.encoder(adj, x)
        rep_pooled = rep.mean(dim=0)
        return rep, rep_pooled

    def save(self, file_name: str) -> None:
        torch.save(self.encoder.state_dict(), file_name)


def drop_edge(adj: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return adj
    adj = adj.coalesce()
    values = adj.values()
    keep_mask = torch.rand_like(values) >= drop_rate
    indices = adj.indices()[:, keep_mask]
    values = values[keep_mask]
    return torch.sparse_coo_tensor(indices, values, adj.size()).coalesce()


def _block_diag_sparse(adjs: list[torch.Tensor]) -> torch.Tensor:
    if not adjs:
        raise ValueError("No adjacency matrices to combine")
    indices_list = []
    values_list = []
    offset = 0
    for adj in adjs:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        indices_list.append(indices + offset)
        values_list.append(values)
        offset += adj.size(0)
    indices = torch.cat(indices_list, dim=1)
    values = torch.cat(values_list)
    return torch.sparse_coo_tensor(indices, values, (offset, offset)).coalesce()


class GraphPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[GraphSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]


def _collate_graphs(batch: list[GraphSample]) -> tuple[torch.Tensor, torch.Tensor]:
    adjs = [sample.adjacency for sample in batch]
    feats = [sample.features for sample in batch]
    batch_adj = _block_diag_sparse(adjs)
    batch_feat = torch.cat(feats, dim=0)
    return batch_adj, batch_feat


def _load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as fp:
        graph = pickle.load(fp)
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"Graph at {path} is not a networkx graph")
    return graph


def _ensure_directed(graph: nx.Graph) -> nx.DiGraph:
    return graph.to_directed() if not graph.is_directed() else graph


def _build_grid_graph(row: int, column: int, edge_probability: float) -> nx.DiGraph:
    total_nodes = row * column
    while True:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(1, total_nodes + 1))
        for r in range(row):
            for c in range(column):
                node = r * column + c + 1
                if r + 1 < row:
                    neighbor = node + column
                    if c == 0 or c == column - 1 or random.random() <= edge_probability:
                        graph.add_edge(node, neighbor)
                        graph.add_edge(neighbor, node)
                if c + 1 < column:
                    neighbor = node + 1
                    if r == 0 or r == row - 1 or random.random() <= edge_probability:
                        graph.add_edge(node, neighbor)
                        graph.add_edge(neighbor, node)
        if nx.is_connected(graph.to_undirected()):
            return graph


def _build_map_graph(edge_probability: float, graph_path: str) -> nx.DiGraph:
    graph = _load_graph(graph_path)
    if edge_probability >= 1.0:
        return _ensure_directed(graph)
    while True:
        new_graph = graph.copy()
        for edge in list(graph.edges):
            deg_0 = graph.degree[edge[0]]
            deg_1 = graph.degree[edge[1]]
            if random.random() > edge_probability and deg_0 > 3 and deg_1 > 3:
                new_graph.remove_edge(*edge)
        if nx.is_connected(new_graph.to_undirected()):
            return _ensure_directed(new_graph)


def _load_or_build_random_graph(graph_path: str, build_fn) -> nx.DiGraph:
    if os.path.exists(graph_path):
        graph = _load_graph(graph_path)
    else:
        graph = build_fn()
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, "wb") as fp:
            pickle.dump(graph, fp)
    return _ensure_directed(graph)


def _build_sf_graph(node_number: int, seed: int, cache_dir: str) -> nx.DiGraph:
    graph_path = os.path.join(cache_dir, f"sf_graph_node_num{node_number}.gpickle")

    def _build():
        graph = nx.barabasi_albert_graph(node_number, 2, seed=seed)
        if not nx.is_connected(graph):
            raise ValueError(f"Graph is not connected. Change seed {seed} to generate again.")
        mapping = {node: node + 1 for node in graph.nodes}
        return nx.relabel_nodes(graph, mapping)

    return _load_or_build_random_graph(graph_path, _build)


def _build_er_graph(node_number: int, edge_probability: float, seed: int, cache_dir: str) -> nx.DiGraph:
    graph_path = os.path.join(cache_dir, f"er_graph_node_num{node_number}_prob{edge_probability}.gpickle")

    def _build():
        graph = nx.erdos_renyi_graph(node_number, edge_probability, seed=seed)
        if not nx.is_connected(graph):
            raise ValueError(f"Graph is not connected. Change seed {seed} to generate again.")
        mapping = {node: node + 1 for node in graph.nodes}
        return nx.relabel_nodes(graph, mapping)

    return _load_or_build_random_graph(graph_path, _build)


def _build_sw_graph(node_number: int, edge_probability: float, seed: int, small_world_k: int,
                    cache_dir: str) -> nx.DiGraph:
    graph_path = os.path.join(cache_dir, f"sw_graph_node_num{node_number}_k{small_world_k}_prob{edge_probability}.gpickle")

    def _build():
        graph = nx.connected_watts_strogatz_graph(node_number, small_world_k, edge_probability, 100, seed=seed)
        if not nx.is_connected(graph):
            raise ValueError(f"Graph is not connected. Change seed {seed} to generate again.")
        mapping = {node: node + 1 for node in graph.nodes}
        return nx.relabel_nodes(graph, mapping)

    return _load_or_build_random_graph(graph_path, _build)


def _build_sy_graph(graph_path: str) -> nx.DiGraph:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"SY graph file not found: {graph_path}")
    graph = _load_graph(graph_path)
    if not nx.is_connected(graph.to_undirected()):
        raise ValueError("SY graph is not connected.")
    return _ensure_directed(graph)


def _sample_time_horizon(args) -> int:
    if args.min_time_horizon < args.max_time_horizon:
        return int(np.random.randint(args.min_time_horizon, args.max_time_horizon))
    return int(args.max_time_horizon)


def _sample_num_defenders(args) -> int:
    if args.min_num_defender < args.max_num_defender:
        return int(np.random.randint(args.min_num_defender, args.max_num_defender))
    return int(args.max_num_defender)


def _sample_num_exits(args, max_exits: int) -> int:
    if args.min_num_exit < args.max_num_exit:
        return int(np.random.randint(args.min_num_exit, args.max_num_exit))
    return int(args.max_num_exit)


def _sample_exit_nodes(args, graph: nx.Graph, row: int | None, column: int | None) -> list[int]:
    graph_type = args.graph_type
    node_list = list(sorted(graph.nodes()))
    if graph_type == "Grid_Graph":
        if row is None or column is None:
            raise ValueError("row/column required for Grid_Graph")
        exit_candidates = (
            [i + 1 for i in range(column)]
            + [(row - 1) * column + i + 1 for i in range(column)]
            + [i * column + 1 for i in range(1, row - 1)]
            + [i * column + column for i in range(1, row - 1)]
        )
        max_exits = row * 2 + column * 2 - 4
        num_exit = _sample_num_exits(args, max_exits)
        exit_nodes = list(np.random.choice(exit_candidates, num_exit, replace=False))
        return sorted(exit_nodes)
    if graph_type == "SY_Graph":
        candidate_start_nodes = [103, 112, 34, 155, 94, 117, 132, 53, 174, 198, 50, 91, 26, 29, 141, 13, 138, 197]
        exit_candidates = [node for node in node_list if node not in candidate_start_nodes]
        num_exit = _sample_num_exits(args, len(exit_candidates))
        exit_nodes = list(np.random.choice(exit_candidates, num_exit, replace=False))
        return sorted(exit_nodes)
    num_exit = _sample_num_exits(args, len(node_list))
    exit_nodes = list(np.random.choice(node_list, num_exit, replace=False))
    return sorted(exit_nodes)


def _sample_initial_locations(args, graph: nx.Graph, exit_nodes: list[int], row: int | None, column: int | None,
                              num_defender: int) -> tuple[int, list[int]]:
    graph_type = args.graph_type
    node_list = list(sorted(graph.nodes()))
    if graph_type == "Grid_Graph":
        feasible_locations = [node for node in node_list if node not in exit_nodes]
        perm = np.random.permutation(feasible_locations)
        attacker = int(perm[0])
        defenders = list(np.random.choice(list(perm[1:]), num_defender, replace=False))
        return attacker, [int(x) for x in defenders]
    if graph_type == "SY_Graph":
        candidate_start_nodes = [103, 112, 34, 155, 94, 117, 132, 53, 174, 198, 50, 91, 26, 29, 141, 13, 138, 197]
        rnd_locations = np.random.permutation(candidate_start_nodes)
        attacker = int(rnd_locations[0])
        defenders = [int(x) for x in rnd_locations[1:num_defender + 1]]
        return attacker, defenders
    if graph_type in {"SF_Graph", "SW_Graph", "ER_Graph"}:
        feasible_locations = [node for node in node_list if node not in exit_nodes]
        rnd_locations = np.random.permutation(feasible_locations)
        attacker = int(rnd_locations[0])
        defenders = [int(x) for x in rnd_locations[1:num_defender + 1]]
        return attacker, defenders
    feasible_locations = [node for node in node_list if node not in exit_nodes]
    attacker = int(feasible_locations[0])
    defenders = list(np.random.choice(list(feasible_locations[1:]), num_defender, replace=False))
    return attacker, [int(x) for x in defenders]


def _compute_attacker_paths(graph: nx.Graph, start_node: int, exit_nodes: list[int], max_length: int) -> tuple[list[list[int]], dict[int, list[list[int]]]]:
    paths: list[list[int]] = []
    path_list: dict[int, list[list[int]]] = {}
    exit_set = set(exit_nodes)
    for exit_node in exit_nodes:
        node_paths: list[list[int]] = []
        if nx.has_path(graph, source=start_node, target=exit_node):
            for path in nx.all_shortest_paths(graph, source=start_node, target=exit_node):
                if len(path) > max_length + 1 or len(node_paths) >= MAX_PATHS_PER_EXIT:
                    break
                if list(set(path) & exit_set) == [exit_node]:
                    node_paths.append(list(path))
        path_list[exit_node] = node_paths
        paths.extend(node_paths)
    return paths, path_list


def _valid_exit_paths(path_list: dict[int, list[list[int]]], min_attacker_pth_len: int) -> bool:
    lengths = np.array([len(paths[0]) if len(paths) > 0 else 0 for paths in path_list.values()])
    if np.sum(lengths > 0) == 0:
        return False
    return int(np.min(lengths[lengths > 0])) >= min_attacker_pth_len


def _build_graph_for_args(args, row: int | None, column: int | None) -> nx.DiGraph:
    graph_type = args.graph_type
    if graph_type == "Grid_Graph":
        if row is None or column is None:
            raise ValueError("row and column are required for Grid_Graph")
        return _build_grid_graph(row, column, args.edge_probability)
    if graph_type == "Map_Graph":
        graph_path = args.graph_gpickle_path
        return _build_map_graph(args.edge_probability, graph_path)
    if graph_type == "SY_Graph":
        graph_path = args.graph_gpickle_path
        return _build_sy_graph(graph_path)
    cache_dir = os.path.join("graphchase", "graph", "generated_graphs")
    if graph_type == "SF_Graph":
        return _build_sf_graph(args.sf_sw_node_num, args.seed_to_generate_graph, cache_dir)
    if graph_type == "SW_Graph":
        return _build_sw_graph(args.sf_sw_node_num, args.edge_probability, args.seed_to_generate_graph,
                               args.small_world_k, cache_dir)
    if graph_type == "ER_Graph":
        return _build_er_graph(args.sf_sw_node_num, args.edge_probability, args.seed_to_generate_graph, cache_dir)
    raise ValueError(f"Unknown graph_type {graph_type}")


def build_game_settings(args) -> GameSettings:
    if args.graph_type == "Grid_Graph" and args.differ_size:
        row = int(np.random.randint(args.row_min, args.row_max + 1))
        column = int(np.random.randint(args.column_min, args.column_max + 1))
    else:
        row = int(args.row) if args.graph_type == "Grid_Graph" else None
        column = int(args.column) if args.graph_type == "Grid_Graph" else None

    graph = _build_graph_for_args(args, row, column)
    num_defender = _sample_num_defenders(args)
    exit_nodes = _sample_exit_nodes(args, graph, row, column)
    attacker_init, defender_init = _sample_initial_locations(args, graph, exit_nodes, row, column, num_defender)
    time_horizon = _sample_time_horizon(args)
    metadata = {
        "graph_type": args.graph_type,
        "row": row,
        "column": column,
        "edge_probability": args.edge_probability,
    }
    return GameSettings(
        graph=graph,
        attacker_init=[attacker_init],
        defender_init=defender_init,
        exit_nodes=exit_nodes,
        time_horizon=time_horizon,
        metadata=metadata,
        use_weighted_graph=bool(getattr(args, "use_weighted_graph", False)),
    )


def _node_features(settings: GameSettings, node_feat_dim: int) -> np.ndarray:
    graph = settings.graph
    nodes = list(sorted(graph.nodes()))
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    feat = np.zeros((num_nodes, node_feat_dim), dtype=float)

    if node_feat_dim >= 1:
        for node in settings.exit_nodes:
            feat[node_idx[node], 0] = 1
    if node_feat_dim >= 2 and settings.attacker_init:
        feat[node_idx[settings.attacker_init[0]], 1] = 1
    if node_feat_dim >= 3:
        for node in settings.defender_init:
            feat[node_idx[node], 2] += 1

    degrees = np.array([deg for _, deg in graph.degree()], dtype=int)
    degrees = np.minimum(degrees, MAX_DEGREES)
    degree_one_hot = np.eye(MAX_DEGREES + 1)[degrees]
    return np.concatenate((feat, degree_one_hot), axis=1)


def _adjacency_matrix(settings: GameSettings) -> torch.Tensor:
    graph = settings.graph
    nodes = list(sorted(graph.nodes()))
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    indices = []
    values = []
    for u, v, data in graph.edges(data=True):
        indices.append([node_idx[u], node_idx[v]])
        weight = data.get("weight", 1.0) if isinstance(data, dict) else 1.0
        values.append(float(weight))
    if not indices:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float)
    else:
        indices = torch.tensor(indices, dtype=torch.long).t()
        values = torch.tensor(values, dtype=torch.float)
    size = (len(nodes), len(nodes))
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


def _settings_to_sample(settings: GameSettings, node_feat_dim: int) -> GraphSample:
    features = torch.tensor(_node_features(settings, node_feat_dim), dtype=torch.float)
    adjacency = _adjacency_matrix(settings)
    return GraphSample(adjacency=adjacency, features=features)


def _game_pool_path(args) -> str:
    base_dir = args.game_pool_dir
    file_name = (
        f"{args.graph_type}_size{args.pool_size}_dnum{args.min_num_defender}_{args.max_num_defender}_"
        f"enum{args.min_num_exit}_{args.max_num_exit}_T{args.min_time_horizon}_{args.max_time_horizon}_"
        f"map{args.min_attacker_pth_len}.pik"
    )
    return os.path.join(base_dir, file_name)


def _normalize_game_pool_item(item: Any) -> GameSettings:
    if isinstance(item, GameSettings):
        return item
    if isinstance(item, dict):
        graph = item.get("graph")
        if graph is None and "graph_path" in item:
            graph = _load_graph(item["graph_path"])
        if not isinstance(graph, nx.Graph):
            raise ValueError("game_pool item missing graph")
        return GameSettings(
            graph=graph,
            attacker_init=list(item.get("attacker_init", [])),
            defender_init=list(item.get("defender_init", [])),
            exit_nodes=list(item.get("exit_nodes", [])),
            time_horizon=int(item.get("time_horizon", 0)),
            metadata=dict(item.get("metadata", {})),
            use_weighted_graph=bool(item.get("use_weighted_graph", False)),
        )
    raise TypeError("Unsupported game_pool item type")


def load_game_pool(args) -> list[GameSettings]:
    file_path = _game_pool_path(args)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"game_pool file not found: {file_path}")
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)
    raw_pool = data.get("game_pool") if isinstance(data, dict) else data
    if not isinstance(raw_pool, list):
        raise ValueError("game_pool must be a list or dict with key 'game_pool'")
    return [_normalize_game_pool_item(item) for item in raw_pool]


def generate_game_pool(args) -> list[GameSettings]:
    pool: list[GameSettings] = []
    for i in range(args.pool_size):
        if i % 1000 == 0:
            print(f"Generate graph {i}")
        while True:
            settings = build_game_settings(args)
            attacker_start = settings.attacker_init[0] if settings.attacker_init else None
            if attacker_start is None:
                continue
            _, path_list = _compute_attacker_paths(settings.graph, attacker_start, settings.exit_nodes, settings.time_horizon)
            if args.action_type == "exit_node":
                if _valid_exit_paths(path_list, args.min_attacker_pth_len):
                    break
            else:
                if sum(len(paths) for paths in path_list.values()) > 0:
                    break
        pool.append(settings)
    file_path = _game_pool_path(args)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as fp:
        pickle.dump({"game_pool": pool}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return pool


class GNNGraphPretrainer:
    def __init__(self, args):
        self.args = args
        self.device = _device_from_args(args.device)
        self.save_path = DEFAULT_SAVE_DIR

    def _prepare_samples(self, settings_list: list[GameSettings]) -> list[GraphSample]:
        node_feat_dim = self.args.node_feat_dim
        samples = [_settings_to_sample(settings, node_feat_dim) for settings in settings_list]
        return samples

    def train(self, settings_list: list[GameSettings]) -> None:
        samples = self._prepare_samples(settings_list)
        if not samples:
            raise ValueError("No graph samples available for training")
        node_feat_dim = samples[0].features.shape[1]
        print(f"******** # Num Graphs: {len(samples)}, # Num Feat: {node_feat_dim} ********")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        lr = 0.00015
        weight_decay = 1e-5
        max_epoch = self.args.max_epoch

        differ_size_str = "_ds" if self.args.graph_type == "Grid_Graph" and self.args.differ_size else ""
        game_pool_str = f"_gp{self.args.pool_size}" if self.args.load_game_pool_file else ""
        logger.info(
            "Start pretrain: graph_type=%s%s ep=%s%s layer=%s hidden=%s out=%s dnum=%s_%s enum=%s_%s map=%s",
            self.args.graph_type,
            differ_size_str,
            self.args.edge_probability,
            game_pool_str,
            self.args.gnn_num_layer,
            self.args.gnn_hidden_dim,
            self.args.gnn_output_dim,
            self.args.min_num_defender,
            self.args.max_num_defender,
            self.args.min_num_exit,
            self.args.max_num_exit,
            self.args.min_attacker_pth_len,
        )

        train_idx = torch.arange(len(samples))
        train_sampler = SubsetRandomSampler(train_idx)
        dataset = GraphPretrainDataset(samples)
        train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            collate_fn=_collate_graphs,
            batch_size=32,
            pin_memory=False,
        )

        set_random_seed(self.args.seed)
        model = PreModel(node_feat_dim, self.args.gnn_hidden_dim, self.args.gnn_output_dim, self.args.gnn_num_layer,
                         self.args.gnn_dropout)
        model.to(self.device)

        optimizer = create_optimizer("adam", model, lr, weight_decay)
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

        start_time = time.time()
        start_stamp = datetime.now().replace(microsecond=0)
        train_loss = []
        for epoch in range(max_epoch):
            model.train()
            loss_list = []
            for batch_adj, batch_feat in train_loader:
                batch_adj = batch_adj.to(self.device)
                batch_feat = batch_feat.to(self.device)
                loss, _ = model(batch_adj, batch_feat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            scheduler.step()
            mean_loss = float(np.mean(loss_list)) if loss_list else 0.0
            train_loss.append(mean_loss)
            logger.info("Epoch %s | train_loss: %.4f", epoch + 1, mean_loss)
            if (epoch + 1) % 200 == 0:
                file_name = (
                    f"checkpoint_epoch{epoch + 1}_type_{self.args.graph_type}{differ_size_str}_ep{self.args.edge_probability}"
                    f"{game_pool_str}_layer{self.args.gnn_num_layer}_hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}"
                    f"_dnum{self.args.min_num_defender}_{self.args.max_num_defender}_enum{self.args.min_num_exit}_{self.args.max_num_exit}"
                    f"_map{self.args.min_attacker_pth_len}.pt"
                )
                model.save(os.path.join(self.save_path, file_name))

        train_time = time.time() - start_time
        record_name = (
            f"train_record_type_{self.args.graph_type}{differ_size_str}_ep{self.args.edge_probability}{game_pool_str}"
            f"_layer{self.args.gnn_num_layer}_hidden{self.args.gnn_hidden_dim}_out{self.args.gnn_output_dim}"
            f"_dnum{self.args.min_num_defender}_{self.args.max_num_defender}_enum{self.args.min_num_exit}_{self.args.max_num_exit}"
            f"_map{self.args.min_attacker_pth_len}.pkl"
        )
        record_path = os.path.join(self.save_path, record_name)
        with open(record_path, "wb") as fp:
            pickle.dump({"train_time": train_time, "train_loss": train_loss}, fp, protocol=pickle.HIGHEST_PROTOCOL)

        end_stamp = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_stamp)
        print("Finished training at (GMT) : ", end_stamp)
        print("Total training time  : ", end_stamp - start_stamp)
        print("============================================================================================")


def build_training_settings(args) -> list[GameSettings]:
    if args.load_game_pool_file:
        return load_game_pool(args)
    return generate_game_pool(args)
