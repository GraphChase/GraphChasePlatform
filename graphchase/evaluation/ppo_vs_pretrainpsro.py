from __future__ import annotations

import argparse
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.graph.embedding_graph import maybe_train_graph_embeddings
from graphchase.graph.game_settings import GameSettings
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.runners.attacker_ppo_psro_runner import AttackerPpoPsroRunner
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner
from graphchase.solver_cfgs.pretrain_psro_cfgs_template import build_parser
from graphchase.utils import load_yaml_config

logger = logging.getLogger(__name__)


@dataclass
class PolicyPopulation:
    name: str
    attacker_runners: list[Any]
    defender_runners: list[Any]
    attacker_meta_strategy: np.ndarray
    defender_meta_strategy: np.ndarray
    meta_iteration: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO and pretrain-PSRO policy populations on unweighted Mumbai.")
    parser.add_argument(
        "--pretrain_exp_dir",
        type=str,
        default="experiments/pretrain_psro/mumbai",
        help="Directory containing pretrain-PSRO attacker/defender strategy folders.",
    )
    parser.add_argument(
        "--ppo_exp_dir",
        type=str,
        default="experiments/both_ppo_unweighted/mumbai",
        help="Directory containing PPO attacker/defender strategy folders.",
    )
    parser.add_argument(
        "--graph_gpickle_path",
        type=str,
        default="graphchase/graph/custom_graph/mumbai.gpickle",
        help="Unweighted graph used for evaluation.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1000,
        help="Number of simulations for each attacker/defender population pairing.",
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


def validate_shared_game_setup(
    pretrain_args: argparse.Namespace,
    ppo_args: argparse.Namespace,
    graph_gpickle_path: str,
) -> None:
    target_graph_path = Path(graph_gpickle_path).resolve()
    required_fields = [
        "attacker_init",
        "defender_init",
        "exit_nodes",
        "time_horizon",
    ]
    for field in required_fields:
        if getattr(pretrain_args, field) != getattr(ppo_args, field):
            raise ValueError(f"Mismatched field '{field}' between experiment configs")
    if Path(pretrain_args.graph_gpickle_path).resolve() != target_graph_path:
        raise ValueError("Pretrain-PSRO config graph does not match requested evaluation graph")
    if Path(ppo_args.graph_gpickle_path).resolve() != target_graph_path:
        raise ValueError("PPO config graph does not match requested evaluation graph")


def build_env_builder(graph_gpickle_path: str, reference_args: argparse.Namespace):
    with open(graph_gpickle_path, "rb") as fp:
        base_graph = pickle.load(fp)

    settings = GameSettings(
        graph=base_graph,
        attacker_init=list(reference_args.attacker_init),
        defender_init=list(reference_args.defender_init),
        exit_nodes=list(reference_args.exit_nodes),
        time_horizon=reference_args.time_horizon,
        metadata={"source": graph_gpickle_path},
        use_weighted_graph=False,
    )

    def _make_env():
        return UNSGEnv(settings)

    return _make_env


def infer_embedding_size(args: argparse.Namespace, graph_embeddings: dict[int, np.ndarray] | None) -> int:
    if graph_embeddings:
        first = next(iter(graph_embeddings.values()))
        return int(len(first))
    if not args.graph_embeddings:
        return 1
    return args.emb_size * 2 if args.line_order == "all" else args.emb_size


def compute_input_dim(env: UNSGEnv, embedding_size: int) -> int:
    return (env.num_attackers + env.num_defenders) * embedding_size + 1


def compute_action_dim(env: UNSGEnv, agent_count: int) -> int:
    max_branch = 0
    for node in env.graph.nodes():
        max_branch = max(max_branch, env.graph.degree[node] + 1)
    return max_branch ** agent_count


def build_ppo_algorithm(args: argparse.Namespace, device: torch.device) -> PPOAlgorithm:
    return PPOAlgorithm(
        learning_rate=args.ppo_actor_lr,
        critic_learning_rate=args.ppo_critic_lr,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_lambda,
        clip_coef=args.ppo_clip,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_batch_size,
        entropy_coef=args.entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_vloss=True,
        device=device,
    )


def list_meta_iterations(strategy_dir: Path) -> list[int]:
    iterations = []
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


def load_meta_strategy(strategy_dir: Path, iteration: int, expected_size: int) -> np.ndarray:
    path = strategy_dir / f"meta_strategy_iter_{iteration}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meta strategy file: {path}")
    meta_strategy = np.asarray(np.load(path), dtype=float).flatten()
    if meta_strategy.shape[0] != expected_size:
        raise ValueError(
            f"Meta strategy length mismatch for {path}: expected {expected_size}, got {meta_strategy.shape[0]}"
        )
    if np.any(meta_strategy < 0):
        raise ValueError(f"Meta strategy contains negative probabilities: {path}")
    total = float(meta_strategy.sum())
    if total <= 0:
        raise ValueError(f"Meta strategy sum must be positive: {path}")
    return meta_strategy / total


def list_strategy_files(strategy_dir: Path, suffix: str) -> list[Path]:
    entries = []
    for path in strategy_dir.glob(f"strategy_*.{suffix}"):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid strategy filename: {path.name}") from exc
        entries.append((index, path))
    if not entries:
        raise ValueError(f"No strategy files found in {strategy_dir}")
    return [path for _, path in sorted(entries, key=lambda item: item[0])]


def build_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and args.use_cuda:
        return torch.device(f"cuda:{args.device_id}")
    return torch.device("cpu")


def load_pretrain_attackers(
    exp_args: argparse.Namespace,
    env_builder,
    attacker_dir: Path,
) -> list[AttackerPathRunner]:
    runners = []
    strategy_files = list_strategy_files(attacker_dir, "pkl")
    for strategy_path in strategy_files:
        runner = AttackerPathRunner(
            env_builder=env_builder,
            action_type=exp_args.action_type,
            strategy_type=exp_args.strategy_type,
            max_path_length=exp_args.time_horizon,
            attacker_path_type=exp_args.attacker_path_type,
            metrics_logger=None,
        )
        runner.load(str(strategy_path))
        runners.append(runner)
    return runners


def load_ppo_attackers(
    exp_args: argparse.Namespace,
    env_builder,
    attacker_dir: Path,
    graph_embeddings: dict[int, np.ndarray] | None,
    embedding_size: int,
    device: torch.device,
) -> list[AttackerPpoPsroRunner]:
    temp_env = env_builder()
    input_dim = compute_input_dim(temp_env, embedding_size)
    action_dim = compute_action_dim(temp_env, temp_env.num_attackers)
    temp_env.close()

    runners = []
    strategy_files = list_strategy_files(attacker_dir, "pt")
    for strategy_path in strategy_files:
        agent = PPOAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=exp_args.ppo_hidden_dim,
            device=device,
        )
        runner = AttackerPpoPsroRunner(
            env_builder=env_builder,
            agent=agent,
            algorithm=build_ppo_algorithm(exp_args, device),
            metrics_logger=None,
            graph_embeddings=graph_embeddings,
            time_horizon=exp_args.time_horizon,
            embedding_size=embedding_size,
        )
        runner.load(str(strategy_path))
        runner.agent.eval()
        runners.append(runner)
    return runners


def load_defenders(
    exp_args: argparse.Namespace,
    env_builder,
    defender_dir: Path,
    graph_embeddings: dict[int, np.ndarray] | None,
    embedding_size: int,
    device: torch.device,
) -> list[DefenderPretrainPsroRunner]:
    temp_env = env_builder()
    input_dim = compute_input_dim(temp_env, embedding_size)
    action_dim = compute_action_dim(temp_env, temp_env.num_defenders)
    temp_env.close()

    runners = []
    strategy_files = list_strategy_files(defender_dir, "pt")
    for strategy_path in strategy_files:
        agent = PPOAgent(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=exp_args.ppo_hidden_dim,
            device=device,
        )
        runner = DefenderPretrainPsroRunner(
            env_builder=env_builder,
            agent=agent,
            algorithm=build_ppo_algorithm(exp_args, device),
            metrics_logger=None,
            graph_embeddings=graph_embeddings,
            time_horizon=exp_args.time_horizon,
            embedding_size=embedding_size,
        )
        runner.load(str(strategy_path))
        runner.agent.eval()
        runners.append(runner)
    return runners


def load_pretrain_population(exp_dir: str, env_builder) -> PolicyPopulation:
    exp_args = load_experiment_config(exp_dir)
    graph_embeddings = maybe_train_graph_embeddings(exp_args, logger)
    embedding_size = infer_embedding_size(exp_args, graph_embeddings)
    device = build_device(exp_args)

    attacker_dir = Path(exp_dir) / "attacker"
    defender_dir = Path(exp_dir) / "defender"
    attacker_runners = load_pretrain_attackers(exp_args, env_builder, attacker_dir)
    defender_runners = load_defenders(
        exp_args,
        env_builder,
        defender_dir,
        graph_embeddings=graph_embeddings,
        embedding_size=embedding_size,
        device=device,
    )
    iteration = choose_latest_common_iteration(attacker_dir, defender_dir)
    attacker_meta_strategy = load_meta_strategy(attacker_dir, iteration, len(attacker_runners))
    defender_meta_strategy = load_meta_strategy(defender_dir, iteration, len(defender_runners))
    return PolicyPopulation(
        name="pretrain_psro",
        attacker_runners=attacker_runners,
        defender_runners=defender_runners,
        attacker_meta_strategy=attacker_meta_strategy,
        defender_meta_strategy=defender_meta_strategy,
        meta_iteration=iteration,
    )


def load_ppo_population(exp_dir: str, env_builder) -> PolicyPopulation:
    exp_args = load_experiment_config(exp_dir)
    graph_embeddings = maybe_train_graph_embeddings(exp_args, logger)
    embedding_size = infer_embedding_size(exp_args, graph_embeddings)
    device = build_device(exp_args)

    attacker_dir = Path(exp_dir) / "attacker"
    defender_dir = Path(exp_dir) / "defender"
    attacker_runners = load_ppo_attackers(
        exp_args,
        env_builder,
        attacker_dir,
        graph_embeddings=graph_embeddings,
        embedding_size=embedding_size,
        device=device,
    )
    defender_runners = load_defenders(
        exp_args,
        env_builder,
        defender_dir,
        graph_embeddings=graph_embeddings,
        embedding_size=embedding_size,
        device=device,
    )
    iteration = choose_latest_common_iteration(attacker_dir, defender_dir)
    attacker_meta_strategy = load_meta_strategy(attacker_dir, iteration, len(attacker_runners))
    defender_meta_strategy = load_meta_strategy(defender_dir, iteration, len(defender_runners))
    return PolicyPopulation(
        name="ppo",
        attacker_runners=attacker_runners,
        defender_runners=defender_runners,
        attacker_meta_strategy=attacker_meta_strategy,
        defender_meta_strategy=defender_meta_strategy,
        meta_iteration=iteration,
    )


def sample_runner(runners: list[Any], meta_strategy: np.ndarray) -> Any:
    if len(runners) != meta_strategy.shape[0]:
        raise ValueError("Runner count does not match meta strategy length")
    index = int(np.random.choice(len(runners), p=meta_strategy))
    return runners[index]


def evaluate_defender_capture_rate(
    env_builder,
    attacker_runners: list[Any],
    attacker_meta_strategy: np.ndarray,
    defender_runners: list[Any],
    defender_meta_strategy: np.ndarray,
    num_simulations: int,
) -> float:
    env = env_builder()
    defender_wins = 0
    try:
        for _ in range(num_simulations):
            attacker_runner = sample_runner(attacker_runners, attacker_meta_strategy)
            defender_runner = sample_runner(defender_runners, defender_meta_strategy)
            if hasattr(attacker_runner, "reset"):
                attacker_runner.reset()
            if hasattr(defender_runner, "reset"):
                defender_runner.reset()

            obs, info = env.reset()
            done = False
            while not done:
                attacker_action = attacker_runner.policy_action(obs, info)
                defender_action = defender_runner.policy_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(
                    {"attacker_action": attacker_action, "defender_action": defender_action}
                )
                done = terminated or truncated
                if done and float(reward["defender"]) > float(reward["attacker"]):
                    defender_wins += 1
    finally:
        env.close()
    return float(defender_wins) / float(num_simulations)


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    return "\n".join(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) for row in rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_random_seeds(args.seed)

    pretrain_args = load_experiment_config(args.pretrain_exp_dir)
    ppo_args = load_experiment_config(args.ppo_exp_dir)
    validate_shared_game_setup(pretrain_args, ppo_args, args.graph_gpickle_path)

    env_builder = build_env_builder(args.graph_gpickle_path, pretrain_args)
    pretrain_population = load_pretrain_population(args.pretrain_exp_dir, env_builder)
    ppo_population = load_ppo_population(args.ppo_exp_dir, env_builder)

    logger.info("Loaded pretrain-PSRO population from %s (meta_strategy_iter=%s)", args.pretrain_exp_dir, pretrain_population.meta_iteration)
    logger.info("Loaded PPO population from %s (meta_strategy_iter=%s)", args.ppo_exp_dir, ppo_population.meta_iteration)

    capture_rates = {
        ("pretrain_psro", "pretrain_psro"): evaluate_defender_capture_rate(
            env_builder,
            pretrain_population.attacker_runners,
            pretrain_population.attacker_meta_strategy,
            pretrain_population.defender_runners,
            pretrain_population.defender_meta_strategy,
            args.num_simulations,
        ),
        ("pretrain_psro", "ppo"): evaluate_defender_capture_rate(
            env_builder,
            pretrain_population.attacker_runners,
            pretrain_population.attacker_meta_strategy,
            ppo_population.defender_runners,
            ppo_population.defender_meta_strategy,
            args.num_simulations,
        ),
        ("ppo", "pretrain_psro"): evaluate_defender_capture_rate(
            env_builder,
            ppo_population.attacker_runners,
            ppo_population.attacker_meta_strategy,
            pretrain_population.defender_runners,
            pretrain_population.defender_meta_strategy,
            args.num_simulations,
        ),
        ("ppo", "ppo"): evaluate_defender_capture_rate(
            env_builder,
            ppo_population.attacker_runners,
            ppo_population.attacker_meta_strategy,
            ppo_population.defender_runners,
            ppo_population.defender_meta_strategy,
            args.num_simulations,
        ),
    }

    rows = [
        ["attacker\\defender", "pretrain_psro", "ppo"],
        [
            "pretrain_psro",
            f"{capture_rates[('pretrain_psro', 'pretrain_psro')]:.4f}",
            f"{capture_rates[('pretrain_psro', 'ppo')]:.4f}",
        ],
        [
            "ppo",
            f"{capture_rates[('ppo', 'pretrain_psro')]:.4f}",
            f"{capture_rates[('ppo', 'ppo')]:.4f}",
        ],
    ]
    print(format_table(rows))


if __name__ == "__main__":
    main()
