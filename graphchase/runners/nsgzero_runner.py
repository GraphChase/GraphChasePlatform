from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from graphchase.solver.nsgzero.run import run
from graphchase.utils import save_experiment_config

logger = logging.getLogger(__name__)


class NSGZeroRunner:
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

    def run(self) -> None:
        self._seed_everything()
        save_root = self.args.save_path
        os.makedirs(save_root, exist_ok=True)
        self.args.ex_results_path = os.path.join(save_root, self.args.run_id)
        os.makedirs(self.args.ex_results_path, exist_ok=True)
        config_payload = vars(self.args).copy()
        device_value = config_payload.get("device")
        if isinstance(device_value, torch.device):
            config_payload["device"] = str(device_value)
        config_path = save_experiment_config(config_payload, self.args.ex_results_path)
        logger.info("Saved experiment config to %s", config_path)
        run(self.args)
