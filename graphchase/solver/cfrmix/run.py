from __future__ import annotations

import logging
import os

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import build_game_settings
from graphchase.solver.cfrmix.algorithm.mix_deep_probe_cfr import deep_mix_probe_cfr
from graphchase.solver.cfrmix.game_setting.graph_module import graph as CFRMixGraph

logger = logging.getLogger(__name__)


def _normalize_regret_file_names(value) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
    else:
        raise ValueError("regret_file_names must be a list or comma-separated string")
    if len(names) != 2:
        raise ValueError("regret_file_names must contain exactly two file names")
    return names


def _resolve_path(root: str, name: str) -> str:
    if os.path.isabs(name):
        return name
    return os.path.join(root, name)


def _normalize_strategy_template(value: str) -> str:
    template = str(value)
    if "{" not in template:
        template = f"{template}_{{}}"
    return template


def run(args) -> None:
    if not hasattr(args, "ex_results_path"):
        args.ex_results_path = args.save_path
    settings = build_game_settings(args)
    env = UNSGEnv(settings)
    game_graph = CFRMixGraph(settings, env)

    attacker_init = list(args.attacker_init)
    defender_init = list(args.defender_init)
    if len(attacker_init) != 1:
        raise ValueError("CFRMix currently supports a single attacker start position")
    init_location = [attacker_init[0], tuple(defender_init)]

    regret_names = _normalize_regret_file_names(args.regret_file_names)
    regret_file_name = [_resolve_path(args.ex_results_path, name) for name in regret_names]
    strategy_template = _normalize_strategy_template(args.strategy_file_name)
    if not os.path.isabs(strategy_template):
        strategy_template = os.path.join(args.ex_results_path, strategy_template)

    logger.info("Starting CFRMix with attacker=%s defender=%s", attacker_init, defender_init)

    deep_mix_probe_cfr(
        game_graph,
        init_location,
        args.time_horizon,
        args.network_dim,
        args.sample_number,
        args.action_number,
        args.attacker_regret_batch_size,
        args.defender_regret_batch_size,
        args.defender_strategy_batch_size,
        args.train_epoch,
        args.attacker_regret_lr,
        args.defender_regret_lr,
        args.defender_strategy_lr,
        regret_file_name,
        strategy_template,
        args.iteration,
    )
