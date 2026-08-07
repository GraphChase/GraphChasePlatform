from __future__ import annotations

import os
import pickle
import random
import logging
import numpy as np
import torch
from graphchase.graph.game_settings import GameSettings
from graphchase.envs.unsg_env import UNSGEnv
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.runners.defender_pretrain_psro_runner import DefenderPretrainPsroRunner
from graphchase.agents.ppo_agent import PPOAgent
from graphchase.algorithms.ppo_algorithm import PPOAlgorithm
from graphchase.utils import convert2nodeidx_unweighted_graph

logger = logging.getLogger(__name__)


def build_env_builder():
    graph_path = "graphchase/graph/custom_graph/7_7_grid_graph.gpickle"
    with open(graph_path, "rb") as fp:
        base_graph = pickle.load(fp)
    settings = GameSettings(
        graph=base_graph,
        attacker_init=[25],
        defender_init=[9, 28, 44, 46],
        exit_nodes=[4, 22, 43, 49],
        time_horizon=8,
        metadata={"source": graph_path},
    )

    def _make_env():
        return UNSGEnv(settings)

    return _make_env


def compute_input_dim(env: UNSGEnv) -> int:
    obs, _ = env.reset()
    attacker_nodes = [convert2nodeidx_unweighted_graph((int(p[0]), int(p[1]), float(p[2]))) for p in obs["attacker_state"]]
    defender_nodes = [convert2nodeidx_unweighted_graph((int(p[0]), int(p[1]), float(p[2]))) for p in obs["defender_state"]]
    return len(attacker_nodes) + len(defender_nodes) + 1


def compute_action_dim(env: UNSGEnv) -> int:
    max_branch = 0
    for node in env.graph.nodes():
        degree = env.graph.degree[node]
        max_branch = max(max_branch, degree + 1)  # neighbors + stay
    return max_branch ** env.num_defenders


def main():
    logging.basicConfig(level=logging.INFO)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cpu"

    env_builder = build_env_builder()
    temp_env = env_builder()
    input_dim = compute_input_dim(temp_env)
    action_dim = compute_action_dim(temp_env)
    temp_env.close()

    defender_agent = PPOAgent(input_dim=input_dim, action_dim=action_dim, hidden_dim=64, device=device)
    ppo_algo = PPOAlgorithm(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=2,
        minibatch_size=8,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )
    defender_runner = DefenderPretrainPsroRunner(
        env_builder=env_builder,
        agent=defender_agent,
        algorithm=ppo_algo,
    )

    attacker_runner = AttackerPathRunner(
        env_builder=env_builder,
        max_path_length=temp_env.time_horizon,
    )

    num_iterations = 2
    for itr in range(num_iterations):
        logger.info("=== Iteration %s ===", itr)
        attacker_runner.compute_best_response(defender_runner, config={"rollouts_per_path": 1})
        defender_runner.compute_best_response([attacker_runner], config={"train_batches": 1, "episodes_per_batch": 1})
        eval_metrics = defender_runner.evaluate(attacker_runner, episodes=2)
        logger.info("Defender win_rate: %.3f", eval_metrics["win_rate"])

    save_path = os.path.join("experiments", "graphchase", "defender_br.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    defender_agent.save(save_path)
    logger.info("Saved defender policy to %s", save_path)


if __name__ == "__main__":
    main()
