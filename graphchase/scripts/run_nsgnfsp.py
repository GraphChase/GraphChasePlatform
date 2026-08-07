from __future__ import annotations

import argparse
import logging

from graphchase.runners.nfsp_runner import NFSPRunner
from graphchase.solver_cfgs.nsgnfsp_cfgs_template import build_parser
from graphchase.utils import load_yaml_config


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run NSGNFSP with YAML config")
    parser.add_argument("--config", type=str, default="graphchase/solver_cfgs/nsgnfsp_cfgs.yaml", help="Path to YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. --set seed=123")
    cli_args = parser.parse_args()

    args = load_yaml_config(cli_args.config, build_parser, cli_args.overrides)
    runner = NFSPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
