from __future__ import annotations

import copy
import logging
import os
import random
import re
from typing import Any

import numpy as np
import torch

from graphchase.envs.unsg_env import UNSGEnv
from graphchase.runners.attacker_path_runner import AttackerPathRunner
from graphchase.solver.psro_solver import PSRO
from graphchase.solver.grasper.utils.game_config import get_game
from graphchase.solver.grasper.grasper_mappo.grasper_mappo_mtl import grasper_mappo_mtl
from graphchase.solver.grasper.graph_learning.graph_pretrain import build_pretrain_graphs, graph_pretrain
from graphchase.solver.grasper.grasper_mappo.config import get_mtl_model_results_dir
from graphchase.solver.grasper.grasper_mappo_psro_runner import GrasperMappoPsroRunner

logger = logging.getLogger(__name__)


class GrasperMappoRunner:
    def __init__(self, args) -> None:
        self.args = args
        self._prepare_device()

    def _prepare_device(self) -> None:
        use_cuda = bool(self.args.use_cuda)
        device_id = int(self.args.device_id)
        if torch.cuda.is_available() and use_cuda:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        self.args.device = device
        self.args.cuda = torch.cuda.is_available()

    def _seed_everything(self) -> None:
        seed = int(self.args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_env_builder(self, game):
        settings = game.settings

        def _make_env():
            return UNSGEnv(settings)

        return _make_env

    def run_pre_pretrain(self) -> None:
        self._seed_everything()
        self.args.load_game_pool_file = True
        try:
            graphs, game_pool_str, differ_size_str = build_pretrain_graphs(self.args)
        except FileNotFoundError as exc:
            logger.warning("Game pool file not found, generating a new pool instead: %s", exc)
            self.args.load_game_pool_file = False
            graphs, game_pool_str, differ_size_str = build_pretrain_graphs(self.args)
            self.args.load_game_pool_file = True
        graph_pretrain(self.args, graphs, game_pool_str, differ_size_str)

    def run_pretrain(self) -> None:
        self._seed_everything()
        grasper_mappo_mtl(self.args)

    def run_psro(self) -> dict[str, Any]:
        self._seed_everything()
        game, action_type, _ = get_game(self.args)
        self._prepare_psro_models(game, action_type)

        env_builder = self._build_env_builder(game)
        attacker_runner = AttackerPathRunner(
            env_builder=env_builder,
            action_type=self.args.action_type,
            strategy_type=self.args.strategy_type,
            max_path_length=game.settings.time_horizon,
            attacker_path_type=self.args.attacker_path_type,
        )

        mappo_args = copy.deepcopy(self.args)
        mappo_args.num_defender = game._defender_num
        mappo_args.device = self.args.device

        defender_runner = GrasperMappoPsroRunner(mappo_args, self.args, game)
        self._save_shared_graph_emb(defender_runner)

        psro = PSRO(env_builder=env_builder, attacker_runner=attacker_runner, defender_runner=defender_runner, args=self.args)
        return psro.solve()

    def run_end_to_end(self) -> dict[str, Any]:
        args_copy = copy.deepcopy(self.args)
        args_copy.use_end_to_end = True
        args_copy.use_emb_layer = True
        args_copy.use_node_emb = False
        self.args = args_copy
        self._prepare_device()
        self.run_pretrain()
        return self.run_psro()

    def _prepare_psro_models(self, game, action_type: str) -> None:
        if not self.args.load_pretrain_model:
            self.args.load_pretrain_model = True
        pretrain_iter = self.args.pretrain_model_iteration
        if pretrain_iter is None:
            model_prefix = get_mtl_model_results_dir(game, action_type, self.args, 0)
            selected = self._select_latest_hypernet_checkpoint(model_prefix)
            if selected is None:
                self.args.actor_model = f"{model_prefix}_actor.pt"
                self.args.critic_model = f"{model_prefix}_critic.pt"
                logger.warning("No matching hypernet checkpoints found for %s", model_prefix)
            else:
                self.args.actor_model, self.args.critic_model = selected
            return

        model_prefix = get_mtl_model_results_dir(game, action_type, self.args, int(pretrain_iter))
        actor_model = f"{model_prefix}_actor.pt"
        critic_model = f"{model_prefix}_critic.pt"
        if not (os.path.isfile(actor_model) and os.path.isfile(critic_model)):
            selected = self._select_latest_hypernet_checkpoint(model_prefix)
            if selected is None:
                logger.warning("Hypernet checkpoints not found for %s", model_prefix)
                self.args.actor_model = actor_model
                self.args.critic_model = critic_model
            else:
                self.args.actor_model, self.args.critic_model = selected
        else:
            self.args.actor_model = actor_model
            self.args.critic_model = critic_model

    def _save_shared_graph_emb(self, defender_runner: GrasperMappoPsroRunner) -> None:
        if self.args.use_end_to_end:
            return
        graph_emb_model = defender_runner.graph_emb_model
        if graph_emb_model is None:
            return
        defender_dir = os.path.join(self.args.save_path, "defender")
        os.makedirs(defender_dir, exist_ok=True)
        shared_path = os.path.join(defender_dir, "shared_graph_emb.pt")
        if not os.path.isfile(shared_path):
            graph_emb_model.save(shared_path)

    def _select_latest_hypernet_checkpoint(self, model_prefix: str) -> tuple[str, str] | None:
        directory = os.path.dirname(model_prefix)
        if not os.path.isdir(directory):
            return None
        base_name = os.path.basename(model_prefix)
        match = re.search(r"iter(\d+)", base_name)
        if not match:
            return None
        iter_token = match.group(1)
        base_pattern = re.escape(base_name).replace(f"iter{iter_token}", r"iter(?P<iter>\d+)")
        actor_pattern = re.compile(rf"^{base_pattern}_actor\.pt$")
        best_iter = -1
        best_actor = None
        for name in os.listdir(directory):
            match = actor_pattern.match(name)
            if not match:
                continue
            critic_name = name.replace("_actor.pt", "_critic.pt")
            if not os.path.isfile(os.path.join(directory, critic_name)):
                continue
            iter_value = int(match.group("iter"))
            if iter_value > best_iter:
                best_iter = iter_value
                best_actor = name
        if best_actor is None:
            return None
        actor_path = os.path.join(directory, best_actor)
        critic_path = os.path.join(directory, best_actor.replace("_actor.pt", "_critic.pt"))
        logger.info("Auto-selected hypernet checkpoint: %s", actor_path)
        return actor_path, critic_path
