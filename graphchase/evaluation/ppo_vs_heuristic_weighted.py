from __future__ import annotations

import argparse
import itertools
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graphchase.agents.evader_risk_aware_exit_switching_agent import EvaderRiskAwareExitSwitchingAgent
from graphchase.agents.evader_shortest_path_to_exit_agent import EvaderShortestPathToExitAgent
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.agents.pursuer_nearest_threatened_exit_guard_agent import PursuerNearestThreatenedExitGuardAgent
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.game_settings import GameSettings
from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.utils import load_yaml_config

logger = logging.getLogger(__name__)


@dataclass
class PolicyPopulation:
    name: str
    attacker_policies: list[Any]
    defender_policies: list[Any]
    attacker_meta_strategy: np.ndarray
    defender_meta_strategy: np.ndarray
    meta_iteration: int


class WeightedPpoPolicy:
    def __init__(
        self,
        env_builder,
        role: str,
        hidden_dim: int,
        device: torch.device,
        time_horizon: int,
    ) -> None:
        if role not in {"attacker", "defender"}:
            raise ValueError("role must be attacker or defender")
        self.env_builder = env_builder
        self.role = role
        self.device = device
        self.time_horizon = int(time_horizon)

        env = self.env_builder()
        try:
            self.max_branching = max(env.graph.degree[node] + 1 for node in env.graph.nodes())
            input_dim = (env.num_attackers + env.num_defenders) * 3 + 1
            if self.role == "attacker":
                controlled_agents = env.num_attackers
            else:
                controlled_agents = env.num_defenders
            action_dim = self.max_branching ** controlled_agents
        finally:
            env.close()

        self.agent = PPOAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )
        self._action_map = self._build_action_map(controlled_agents)

    def _build_action_map(self, controlled_agents: int) -> list[list[int]]:
        action_map: list[list[int]] = []
        branches = [list(range(self.max_branching)) for _ in range(controlled_agents)]
        for combo in itertools.product(*branches):
            action_map.append([int(item) for item in combo])
        return action_map

    def _trim_legal_actions(self, legal_actions: list[list[int]]) -> list[list[int]]:
        trimmed: list[list[int]] = []
        for actions in legal_actions:
            trimmed.append(list(actions)[: self.max_branching])
        return trimmed

    def _legal_mask_from_info(self, legal_actions: list[list[int]]) -> torch.Tensor:
        trimmed_actions = self._trim_legal_actions(legal_actions)
        branch_masks: list[list[bool]] = []
        for actions in trimmed_actions:
            valid_len = len(actions)
            branch_masks.append([True] * valid_len + [False] * max(0, self.max_branching - valid_len))

        mask: list[bool] = [True]
        for branch_mask in branch_masks:
            next_mask: list[bool] = []
            for prefix in mask:
                for value in branch_mask:
                    next_mask.append(prefix and value)
            mask = next_mask
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def _map_branch_to_node_actions(self, legal_actions: list[list[int]], branch_actions: list[int]) -> list[int]:
        trimmed_actions = self._trim_legal_actions(legal_actions)
        if len(trimmed_actions) != len(branch_actions):
            raise ValueError("Branch action size does not match legal action size")
        node_actions: list[int] = []
        for branch_index, actions in zip(branch_actions, trimmed_actions):
            if branch_index < 0 or branch_index >= len(actions):
                raise ValueError("Branch action index out of range")
            node_actions.append(int(actions[branch_index]))
        return node_actions

    def _encode_obs(self, obs: dict, cur_time: float | None, remaining_time: float | None) -> torch.Tensor:
        attacker_state = obs.get("attacker_state", [])
        defender_state = obs.get("defender_state", [])

        parts: list[float] = []
        for position in attacker_state:
            parts.extend([float(position[0]), float(position[1]), float(position[2])])
        for position in defender_state:
            parts.extend([float(position[0]), float(position[1]), float(position[2])])

        elapsed_time = 0.0
        if cur_time is not None:
            elapsed_time = float(cur_time)
        if remaining_time is not None:
            elapsed_time += float(1.0 - remaining_time)
        left_time = max(float(self.time_horizon) - elapsed_time, 0.0) / float(self.time_horizon)
        parts.append(left_time)
        return torch.tensor(parts, dtype=torch.float32, device=self.device)

    def load(self, path: str) -> None:
        self.agent.load(path)
        self.agent.eval()

    def reset(self) -> None:
        self.agent.reset_state()

    def policy_action(self, obs: dict, info: dict) -> list[int]:
        if self.role == "attacker":
            legal_actions = info.get("attacker_legal_action", [])
        else:
            legal_actions = info.get("defender_legal_action", [])
        encoded_obs = self._encode_obs(obs, info.get("cur_time"), info.get("remaining_time"))
        legal_mask = self._legal_mask_from_info(legal_actions)
        with torch.no_grad():
            branch_action, _, _, _, _ = self.agent.get_action_and_value(
                encoded_obs=encoded_obs,
                legal_mask=legal_mask,
                action_map=self._action_map,
            )
        return self._map_branch_to_node_actions(legal_actions, branch_action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate weighted PPO policies against heuristic attacker/defender policies.")
    parser.add_argument(
        "--ppo_exp_dir",
        type=str,
        default="experiments/both_ppo_weighted/mumbai",
        help="Directory containing weighted PPO attacker/defender populations.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1000,
        help="Number of simulations for each attacker/defender pairing.",
    )
    parser.add_argument("--seed", type=int, default=77, help="Random seed for numpy, torch, and python.")
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_experiment_config(experiment_dir: str) -> argparse.Namespace:
    config_path = Path(experiment_dir) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return load_yaml_config(str(config_path), build_parser, overrides=[])


def normalize_graph_edge_weights(graph) -> None:
    weights: list[float] = []
    for _, _, data in graph.edges(data=True):
        if not isinstance(data, dict) or "weight" not in data:
            raise ValueError("Weighted graph must provide a numeric 'weight' attribute on every edge")
        weights.append(float(data["weight"]))
    if not weights:
        raise ValueError("Weighted graph must contain at least one edge")

    mean_weight = float(sum(weights) / len(weights))
    if mean_weight <= 0:
        raise ValueError("Mean edge weight must be positive")

    for _, _, data in graph.edges(data=True):
        data["weight"] = float(data["weight"]) / mean_weight


def build_env_builder(exp_args: argparse.Namespace):
    with open(exp_args.graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)
    normalize_graph_edge_weights(base_graph)
    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(exp_args.attacker_init),
        defender_init=list(exp_args.defender_init),
        exit_nodes=list(exp_args.exit_nodes),
        time_horizon=exp_args.time_horizon,
        metadata={"source": exp_args.graph_gpickle_path},
        use_weighted_graph=True,
    )

    def _make_env():
        return UNSGEnv(settings)

    return _make_env


def build_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and args.use_cuda:
        return torch.device(f"cuda:{args.device_id}")
    return torch.device("cpu")


def list_meta_iterations(strategy_dir: Path) -> list[int]:
    iterations: list[int] = []
    for path in strategy_dir.glob("meta_strategy_iter_*.npy"):
        try:
            iterations.append(int(path.stem.split("_")[-1]))
        except ValueError as exc:
            raise ValueError(f"Invalid meta strategy filename: {path.name}") from exc
    return sorted(iterations)


def choose_latest_common_iteration(attacker_dir: Path, defender_dir: Path) -> int:
    attacker_iterations = set(list_meta_iterations(attacker_dir))
    defender_iterations = set(list_meta_iterations(defender_dir))
    common_iterations = sorted(attacker_iterations & defender_iterations)
    if not common_iterations:
        raise ValueError(f"No common meta-strategy iterations in {attacker_dir} and {defender_dir}")
    return common_iterations[-1]


def list_strategy_files(strategy_dir: Path) -> list[Path]:
    entries: list[tuple[int, Path]] = []
    for path in strategy_dir.glob("strategy_*.pt"):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid strategy filename: {path.name}") from exc
        entries.append((index, path))
    if not entries:
        raise ValueError(f"No strategy files found in {strategy_dir}")
    return [path for _, path in sorted(entries, key=lambda item: item[0])]


def load_meta_strategy(strategy_dir: Path, iteration: int, expected_size: int) -> np.ndarray:
    path = strategy_dir / f"meta_strategy_iter_{iteration}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meta strategy file: {path}")
    meta_strategy = np.asarray(np.load(path), dtype=float).flatten()
    if meta_strategy.shape[0] != expected_size:
        raise ValueError(
            f"Meta strategy length mismatch for {path}: expected {expected_size}, got {meta_strategy.shape[0]}"
        )
    total = float(meta_strategy.sum())
    if total <= 0:
        raise ValueError(f"Meta strategy sum must be positive: {path}")
    return meta_strategy / total


def load_weighted_ppo_population(exp_dir: str, env_builder, exp_args: argparse.Namespace) -> PolicyPopulation:
    device = build_device(exp_args)
    attacker_dir = Path(exp_dir) / "attacker"
    defender_dir = Path(exp_dir) / "defender"

    attacker_policies: list[WeightedPpoPolicy] = []
    for strategy_path in list_strategy_files(attacker_dir):
        policy = WeightedPpoPolicy(
            env_builder=env_builder,
            role="attacker",
            hidden_dim=exp_args.ppo_hidden_dim,
            device=device,
            time_horizon=exp_args.time_horizon,
        )
        policy.load(str(strategy_path))
        attacker_policies.append(policy)

    defender_policies: list[WeightedPpoPolicy] = []
    for strategy_path in list_strategy_files(defender_dir):
        policy = WeightedPpoPolicy(
            env_builder=env_builder,
            role="defender",
            hidden_dim=exp_args.ppo_hidden_dim,
            device=device,
            time_horizon=exp_args.time_horizon,
        )
        policy.load(str(strategy_path))
        defender_policies.append(policy)

    iteration = choose_latest_common_iteration(attacker_dir, defender_dir)
    attacker_meta_strategy = load_meta_strategy(attacker_dir, iteration, len(attacker_policies))
    defender_meta_strategy = load_meta_strategy(defender_dir, iteration, len(defender_policies))
    return PolicyPopulation(
        name="ppo",
        attacker_policies=attacker_policies,
        defender_policies=defender_policies,
        attacker_meta_strategy=attacker_meta_strategy,
        defender_meta_strategy=defender_meta_strategy,
        meta_iteration=iteration,
    )


def sample_policy(policies: list[Any], meta_strategy: np.ndarray) -> Any:
    if len(policies) != meta_strategy.shape[0]:
        raise ValueError("Policy count does not match meta strategy length")
    index = int(np.random.choice(len(policies), p=meta_strategy))
    return policies[index]


def evaluate_pursuer_capture_rate(
    env_builder,
    attacker_policies: list[Any],
    attacker_meta_strategy: np.ndarray,
    defender_policies: list[Any],
    defender_meta_strategy: np.ndarray,
    num_simulations: int,
) -> float:
    env = env_builder()
    captures = 0
    try:
        for _ in range(num_simulations):
            attacker_policy = sample_policy(attacker_policies, attacker_meta_strategy)
            defender_policy = sample_policy(defender_policies, defender_meta_strategy)
            if hasattr(attacker_policy, "reset"):
                attacker_policy.reset()
            if hasattr(defender_policy, "reset"):
                defender_policy.reset()

            obs, info = env.reset()
            done = False
            while not done:
                attacker_action = attacker_policy.policy_action(obs, info)
                defender_action = defender_policy.policy_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(
                    {"attacker_action": attacker_action, "defender_action": defender_action}
                )
                done = terminated or truncated
                if terminated and float(reward["defender"]) > float(reward["attacker"]):
                    captures += 1
    finally:
        env.close()
    return float(captures) / float(num_simulations)


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    return "\n".join(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) for row in rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_random_seeds(args.seed)

    exp_args = load_experiment_config(args.ppo_exp_dir)
    if not bool(exp_args.use_weighted_graph):
        raise ValueError("The provided experiment config is not a weighted-graph PPO experiment")
    env_builder = build_env_builder(exp_args)

    ppo_population = load_weighted_ppo_population(args.ppo_exp_dir, env_builder, exp_args)
    logger.info("Loaded weighted PPO population from %s (meta_strategy_iter=%s)", args.ppo_exp_dir, ppo_population.meta_iteration)

    env = env_builder()
    try:
        graph = env.graph
        exit_nodes = list(env.exit_nodes)
    finally:
        env.close()

    attacker_policies = {
        "ppo": (ppo_population.attacker_policies, ppo_population.attacker_meta_strategy),
        "shortest_path_to_exit": ([EvaderShortestPathToExitAgent(graph=graph, exit_nodes=exit_nodes)], np.array([1.0])),
        "risk_aware_exit_switching": (
            [EvaderRiskAwareExitSwitchingAgent(graph=graph, exit_nodes=exit_nodes)],
            np.array([1.0]),
        ),
    }
    defender_policies = {
        "ppo": (ppo_population.defender_policies, ppo_population.defender_meta_strategy),
        "nearest_threatened_exit_guard": (
            [PursuerNearestThreatenedExitGuardAgent(graph=graph, exit_nodes=exit_nodes)],
            np.array([1.0]),
        ),
    }

    results: dict[tuple[str, str], float] = {}
    for attacker_name, (attacker_policy_list, attacker_meta) in attacker_policies.items():
        for defender_name, (defender_policy_list, defender_meta) in defender_policies.items():
            logger.info("Evaluating %s attacker vs %s defender", attacker_name, defender_name)
            results[(attacker_name, defender_name)] = evaluate_pursuer_capture_rate(
                env_builder=env_builder,
                attacker_policies=attacker_policy_list,
                attacker_meta_strategy=attacker_meta,
                defender_policies=defender_policy_list,
                defender_meta_strategy=defender_meta,
                num_simulations=args.num_simulations,
            )

    rows = [
        ["attacker\\defender", "ppo", "nearest_threatened_exit_guard"],
        [
            "ppo",
            f"{results[('ppo', 'ppo')]:.4f}",
            f"{results[('ppo', 'nearest_threatened_exit_guard')]:.4f}",
        ],
        [
            "shortest_path_to_exit",
            f"{results[('shortest_path_to_exit', 'ppo')]:.4f}",
            f"{results[('shortest_path_to_exit', 'nearest_threatened_exit_guard')]:.4f}",
        ],
        [
            "risk_aware_exit_switching",
            f"{results[('risk_aware_exit_switching', 'ppo')]:.4f}",
            f"{results[('risk_aware_exit_switching', 'nearest_threatened_exit_guard')]:.4f}",
        ],
    ]
    print(format_table(rows))


if __name__ == "__main__":
    main()
