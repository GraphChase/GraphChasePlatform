from __future__ import annotations

import argparse
import logging
import time
from graphchase.utils import load_yaml_config
from graphchase.solver_cfgs.grasper_mappo_cfgs_template import build_parser
from graphchase.runners.grasper_mappo_runner import GrasperMappoRunner


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Grasper MAPPO with YAML config")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (use None for defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    parser.add_argument("--mode", type=str, default="psro", choices=["three_step", "end_to_end", "pre_pretrain", "pretrain", "psro"], help="Execution mode")
    cli_args = parser.parse_args()

    args = load_yaml_config(cli_args.config, build_parser, cli_args.overrides)
    runner = GrasperMappoRunner(args)

    if cli_args.mode == "pre_pretrain":
        runner.run_pre_pretrain()
    elif cli_args.mode == "pretrain":
        runner.run_pretrain()
    elif cli_args.mode == "psro":
        runner.run_psro()
    elif cli_args.mode == "end_to_end":
        runner.run_end_to_end()
    else:
        runner.run_pre_pretrain()
        time.sleep(30)
        runner.run_pretrain()
        time.sleep(30)
        runner.run_psro()


if __name__ == "__main__":
    main()
