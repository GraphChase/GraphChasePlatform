from __future__ import annotations

import logging
import os
import time

import numpy as np

from graphchase.solver.prd_solver import projected_replicator_dynamics

logger = logging.getLogger(__name__)


class PpoPsroSolver:
    """
    Symmetric PSRO solver where both attacker and defender best responses are
    trained with PPO runners.
    """

    def __init__(self, env_builder, attacker_runner, defender_runner, args, meta_solver=projected_replicator_dynamics):
        self.env_builder = env_builder
        self.base_attacker_runner = attacker_runner
        self.base_defender_runner = defender_runner
        self.args = args
        self.meta_solver = meta_solver

        self.attacker_runners: list = []
        self.defender_runners: list = []
        self.meta_games: list[np.ndarray] = []
        self.meta_strategies: list[np.ndarray] = []

        self.save_root = self.args.save_path
        self.attacker_save_dir = os.path.join(self.save_root, "attacker")
        self.defender_save_dir = os.path.join(self.save_root, "defender")
        os.makedirs(self.attacker_save_dir, exist_ok=True)
        os.makedirs(self.defender_save_dir, exist_ok=True)

    def _save_attacker_strategy(self, idx: int) -> None:
        save_path = os.path.join(self.attacker_save_dir, f"strategy_{idx}.pt")
        if hasattr(self.attacker_runners[idx], "save"):
            self.attacker_runners[idx].save(save_path)
        else:
            raise NotImplementedError("Attacker runner does not implement save method")

    def _save_defender_strategy(self, idx: int) -> None:
        save_path = os.path.join(self.defender_save_dir, f"strategy_{idx}.pt")
        if hasattr(self.defender_runners[idx], "save"):
            self.defender_runners[idx].save(save_path)
        else:
            raise NotImplementedError("Defender runner does not implement save method")

    def _save_meta_strategies(self, iteration_idx: int) -> None:
        if self.meta_strategies is None or len(self.meta_strategies) == 0:
            return
        attacker_meta_path = os.path.join(self.attacker_save_dir, f"meta_strategy_iter_{iteration_idx}.npy")
        defender_meta_path = os.path.join(self.defender_save_dir, f"meta_strategy_iter_{iteration_idx}.npy")
        np.save(attacker_meta_path, np.asarray(self.meta_strategies[0], dtype=float))
        np.save(defender_meta_path, np.asarray(self.meta_strategies[1], dtype=float))

    def _save_iteration_artifacts(
        self,
        iteration_idx: int,
        attacker_idx: int | None = None,
        defender_idx: int | None = None,
    ) -> None:
        if attacker_idx is not None:
            self._save_attacker_strategy(attacker_idx)
        if defender_idx is not None:
            self._save_defender_strategy(defender_idx)
        self._save_meta_strategies(iteration_idx)

    def _clone_attacker_runner(self, agent_state_dict=None):
        if hasattr(self.base_attacker_runner, "clone"):
            return self.base_attacker_runner.clone(agent_state_dict=agent_state_dict)
        raise NotImplementedError("Base attacker runner does not implement clone")

    def _clone_defender_runner(self, agent_state_dict=None):
        if hasattr(self.base_defender_runner, "clone"):
            return self.base_defender_runner.clone(agent_state_dict=agent_state_dict)
        raise NotImplementedError("Base defender runner does not implement clone")

    def _evaluate_pair(self, attacker_runner, defender_runner, episodes=None):
        env = self.env_builder()
        num_eps = episodes if episodes is not None else self.args.eval_episodes
        attacker_reward = 0.0
        defender_reward = 0.0
        for _ in range(num_eps):
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
                if done:
                    attacker_reward += float(reward["attacker"])
                    defender_reward += float(reward["defender"])
        env.close()
        attacker_reward /= float(num_eps)
        defender_reward /= float(num_eps)
        return attacker_reward, defender_reward

    def _update_meta_game(self):
        r = len(self.attacker_runners)
        c = len(self.defender_runners)
        new_meta = [
            np.full((r, c), fill_value=np.nan, dtype=float),
            np.full((r, c), fill_value=np.nan, dtype=float),
        ]
        if self.meta_games:
            old_r, old_c = self.meta_games[0].shape
            for i in range(min(r, old_r)):
                for j in range(min(c, old_c)):
                    new_meta[0][i, j] = self.meta_games[0][i, j]
                    new_meta[1][i, j] = self.meta_games[1][i, j]
        for i in range(r):
            for j in range(c):
                if np.isnan(new_meta[0][i, j]):
                    a_reward, d_reward = self._evaluate_pair(self.attacker_runners[i], self.defender_runners[j])
                    new_meta[0][i, j] = a_reward
                    new_meta[1][i, j] = d_reward
        self.meta_games = new_meta

    def _compute_meta_strategies(self):
        self.meta_strategies = self.meta_solver(self.meta_games)

    def _sample_runner(self, runners: list, meta_strategy: np.ndarray):
        probs = np.asarray(meta_strategy, dtype=float)
        if probs.shape[0] != len(runners):
            raise ValueError("meta_strategy length must match strategy library size")
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(runners), dtype=float) / float(len(runners))
        else:
            probs = probs / total
        idx = int(np.random.choice(len(runners), p=probs))
        runner = runners[idx]
        if hasattr(runner, "clone"):
            return runner.clone()
        return runner

    def _evaluate_meta_strategy_defender_win_rate(self, episodes: int = 1000) -> float:
        if (
            not self.attacker_runners
            or not self.defender_runners
            or self.meta_strategies is None
            or len(self.meta_strategies) == 0
        ):
            raise ValueError("Strategy libraries and meta strategies must be initialized before evaluation")

        env = self.env_builder()
        defender_wins = 0
        for _ in range(episodes):
            attacker_runner = self._sample_runner(self.attacker_runners, self.meta_strategies[0])
            defender_runner = self._sample_runner(self.defender_runners, self.meta_strategies[1])
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
        env.close()
        return float(defender_wins) / float(episodes)

    def _add_best_responses(self, iteration_idx: int | None = None):
        logger.info("Starting attacker best response computation")
        start_time = time.time()
        attacker_br_runner = self._clone_attacker_runner(
            agent_state_dict=self.base_attacker_runner.agent.state_dict()
        )
        attacker_br_runner.compute_best_response(
            self.defender_runners,
            config={
                "train_batches": self.args.train_defender_batches,
                "episodes_per_batch": self.args.episodes_per_batch,
                "br_iter": iteration_idx,
                "vec_envs": self.args.vec_envs,
            },
            meta_strategy=self.meta_strategies[1],
        )
        logger.info("Attacker best response computation time: %.2f seconds", time.time() - start_time)

        logger.info("Starting defender best response computation")
        start_time = time.time()
        defender_br_runner = self._clone_defender_runner(
            agent_state_dict=self.base_defender_runner.agent.state_dict()
        )
        defender_br_runner.compute_best_response(
            self.attacker_runners,
            config={
                "train_batches": self.args.train_defender_batches,
                "episodes_per_batch": self.args.episodes_per_batch,
                "br_iter": iteration_idx,
                "vec_envs": self.args.vec_envs,
            },
            meta_strategy=self.meta_strategies[0],
        )
        logger.info("Defender best response computation time: %.2f seconds", time.time() - start_time)

        self.attacker_runners.append(attacker_br_runner)
        self.defender_runners.append(defender_br_runner)
        return len(self.attacker_runners) - 1, len(self.defender_runners) - 1

    def init(self):
        base_att_runner = self._clone_attacker_runner(
            agent_state_dict=self.base_attacker_runner.agent.state_dict()
        )
        base_def_runner = self._clone_defender_runner(
            agent_state_dict=self.base_defender_runner.agent.state_dict()
        )
        self.attacker_runners = [base_att_runner]
        self.defender_runners = [base_def_runner]
        self._update_meta_game()
        self.meta_strategies = [np.array([1.0]), np.array([1.0])]
        self._save_iteration_artifacts(iteration_idx=0, attacker_idx=0, defender_idx=0)

    def solve(self):
        self.init()
        start_time = time.time()
        iterations = self.args.num_psro_iteration
        for iteration in range(1, iterations + 1):
            attacker_idx, defender_idx = self._add_best_responses(iteration_idx=iteration)
            self._update_meta_game()
            self._compute_meta_strategies()
            self._save_iteration_artifacts(
                iteration_idx=iteration,
                attacker_idx=attacker_idx,
                defender_idx=defender_idx,
            )
            defender_win_rate = self._evaluate_meta_strategy_defender_win_rate(episodes=1000)
            logger.info("Iteration %s meta-strategy defender win rate: %.4f", iteration, defender_win_rate)
            logger.info("Elapsed time since init: %.2f seconds", time.time() - start_time)

        return {
            "meta_games": self.meta_games,
            "meta_strategies": self.meta_strategies,
            "attacker_runners": self.attacker_runners,
            "defender_runners": self.defender_runners,
        }
