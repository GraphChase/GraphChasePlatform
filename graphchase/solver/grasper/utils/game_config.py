from __future__ import annotations

import os
import pickle
from typing import Any

from graphchase.graph.game_settings import GameSettings
from graphchase.solver.grasper.grasper_game import GrasperGame


def _load_graph(graph_path: str):
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    with open(graph_path, "rb") as fp:
        return pickle.load(fp)


def _settings_from_config(config: dict[str, Any], args) -> GameSettings:
    graph = config.get("graph")
    if graph is None:
        graph_path = config.get("graph_gpickle_path") or getattr(args, "graph_gpickle_path", None)
        if graph_path is None:
            raise ValueError("graph_gpickle_path must be provided when graph is not in config")
        graph = _load_graph(graph_path)
    attacker_init = list(config.get("attacker_init", getattr(args, "attacker_init", [])))
    defender_init = list(config.get("defender_init", getattr(args, "defender_init", [])))
    exit_nodes = list(config.get("exit_nodes", getattr(args, "exit_nodes", [])))
    time_horizon = int(config.get("time_horizon", getattr(args, "time_horizon", 0)))
    metadata = dict(config.get("metadata", getattr(args, "graph_metadata", {})))
    use_weighted_graph = bool(config.get("use_weighted_graph", getattr(args, "use_weighted_graph", False)))
    return GameSettings(
        graph=graph,
        attacker_init=attacker_init,
        defender_init=defender_init,
        exit_nodes=exit_nodes,
        time_horizon=time_horizon,
        metadata=metadata,
        use_weighted_graph=use_weighted_graph,
    )


def get_game(args, given_config: dict[str, Any] | None = None, compute_path: bool = True):
    if given_config is None:
        graph_path = getattr(args, "graph_gpickle_path", None)
        if graph_path is None:
            raise ValueError("graph_gpickle_path must be provided")
        with open(graph_path, "rb") as fp:
            base_graph = pickle.load(fp)
        settings = GameSettings(
            graph=base_graph,
            attacker_init=list(args.attacker_init),
            defender_init=list(args.defender_init),
            exit_nodes=list(args.exit_nodes),
            time_horizon=args.time_horizon,
            metadata={"source": graph_path},
            use_weighted_graph=bool(getattr(args, "use_weighted_graph", False)),
        )
    else:
        settings = _settings_from_config(given_config, args)
    action_type = args.action_type
    game = GrasperGame(settings, args, action_type=action_type, compute_path=compute_path)
    config = {
        "graph_gpickle_path": getattr(args, "graph_gpickle_path", None),
        "attacker_init": list(settings.attacker_init),
        "defender_init": list(settings.defender_init),
        "exit_nodes": list(settings.exit_nodes),
        "time_horizon": int(settings.time_horizon),
        "graph_metadata": dict(settings.metadata),
        "use_weighted_graph": bool(settings.use_weighted_graph),
    }
    return game, action_type, config
