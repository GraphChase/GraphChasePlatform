from __future__ import annotations

import argparse
import json
import os
import shutil
import logging
from typing import Any
from collections.abc import Callable
import yaml
import imageio.v2 as imageio
from graphchase.envs.unsg_env import UNSGEnv

logger = logging.getLogger(__name__)


def _serialize_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_config_value(item) for key, item in value.items()}
    return value


def _flatten_yaml_config(config_data: dict) -> dict:
    flattened = {}
    if not config_data:
        return flattened
    is_grouped = all(isinstance(value, dict) or value is None for value in config_data.values())
    if not is_grouped:
        return dict(config_data)
    for section_name, section_values in config_data.items():
        if section_values is None:
            continue
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping of values")
        for key, value in section_values.items():
            if key in flattened:
                raise ValueError(f"Config key '{key}' is duplicated across sections")
            flattened[key] = value
    return flattened


def _normalize_list_fields(config_values: dict) -> dict:
    int_list_fields = {"attacker_init", "defender_init", "exit_nodes"}
    for field in int_list_fields:
        if field not in config_values:
            continue
        value = config_values[field]
        if isinstance(value, str):
            config_values[field] = [int(item) for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            config_values[field] = [int(item) for item in value]
    return config_values


def _parse_override_values(overrides: list[str]) -> dict:
    if not overrides:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to parse override values. Install with `pip install pyyaml`.") from exc

    parsed = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in key=value format")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ValueError(f"Override '{override}' must include a key before '='")
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


def load_yaml_config(
    config_path: str | None,
    build_parser: Callable[[], argparse.ArgumentParser],
    overrides: list[str] | None = None,
) -> argparse.Namespace:

    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    if config_path is None:
        config_data = {}
    else:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            config_data = yaml.safe_load(file_obj) or {}
    if not isinstance(config_data, dict):
        raise ValueError("YAML config must be a mapping of section names to values")
    flattened = _flatten_yaml_config(config_data)
    _normalize_list_fields(flattened)
    override_values = _parse_override_values(overrides or [])
    _normalize_list_fields(override_values)
    merged_values = {**flattened, **override_values}
    unknown_keys = set(merged_values) - set(defaults)
    if unknown_keys:
        unknown_list = ", ".join(sorted(unknown_keys))
        raise KeyError(f"Unknown config keys in {config_path}: {unknown_list}")
    defaults.update(merged_values)
    args = argparse.Namespace(**defaults)
    if args.graph_metadata is None:
        args.graph_metadata = {}
    elif isinstance(args.graph_metadata, str):
        args.graph_metadata = json.loads(args.graph_metadata)
    return args


def save_experiment_config(args: Any, save_path: str, filename: str = "config.yaml") -> str:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save YAML configs. Install with `pip install pyyaml`.") from exc

    os.makedirs(save_path, exist_ok=True)
    if isinstance(args, dict):
        raw_config = args
    elif hasattr(args, "__dict__"):
        raw_config = vars(args)
    else:
        raise TypeError("args must be a dict or namespace-like object")
    config_data = _serialize_config_value(raw_config)
    file_path = os.path.join(save_path, filename)
    with open(file_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config_data, file_obj, sort_keys=False)
    return file_path


def run_episode(
    env: UNSGEnv,
    num_episodes: int,
    attacker_policy_fn: Callable[[dict, dict], list[int]],
    defender_policy_fn: Callable[[dict, dict], list[int]],
    video_root: str | None = None,
    save_video: bool = False,
) -> list[dict]:
    """
    Run multiple episodes in the given UNSG environment, log every step, and optionally render frames into an mp4 (1 FPS).

    Args:
    - env: UNSG environment instance.
    - num_episodes: Number of episodes to run.
    - attacker_policy_fn: Callable taking (obs, info) and returning attacker action list.
    - defender_policy_fn: Callable taking (obs, info) and returning defender action list.
    - video_root: Directory to store mp4 and temporary frames when save_video=True.
    - save_video: Whether to save rendered frames and compose `episode_<id>.mp4`.

    Returns:
    - episodes_data: List of logs per episode (obs, actions, rewards, termination flags). When save_video=True, mp4s are written under video_root and temporary frame directories are removed after writing.
    """

    def _as_action_list(action, expected_len: int, role: str) -> list[int]:
        if not isinstance(action, (list, tuple)):
            raise ValueError(f"{role} action must be a list or tuple of length {expected_len}")
        action_list = list(action)
        if len(action_list) != expected_len:
            raise ValueError(f"{role} action length {len(action_list)} != expected {expected_len}")
        return [int(a) for a in action_list]

    episodes_data: list[dict] = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_over = False
        frames = []
        frames_dir = None
        frame_idx = 0
        step_log = {
            "obs": [],
            "attacker_action": [],
            "defender_action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "info": [],
        }

        # log initial observation/info before any action
        step_log["obs"].append(obs)
        step_log["info"].append(info)

        if save_video:
            if video_root is None:
                raise ValueError("video_root must be provided when save_video is True")
            os.makedirs(video_root, exist_ok=True)
            frames_dir = os.path.join(video_root, f"episode_{episode}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            init_frame = env.render(mode="rgb_array")
            init_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            imageio.imwrite(init_path, init_frame)
            frames.append(init_frame.copy())
            frame_idx += 1

        while not episode_over:
            attacker_action = _as_action_list(attacker_policy_fn(obs, info), env.num_attackers, "attacker")
            defender_action = _as_action_list(defender_policy_fn(obs, info), env.num_defenders, "defender")

            step_log["attacker_action"].append(attacker_action)
            step_log["defender_action"].append(defender_action)

            obs, reward, terminated, truncated, info = env.step(
                {
                    "attacker_action": attacker_action,
                    "defender_action": defender_action,
                }
            )

            step_log["reward"].append(reward)
            step_log["terminated"].append(terminated)
            step_log["truncated"].append(truncated)
            step_log["info"].append(info)
            step_log["obs"].append(obs)

            if save_video:
                frame = env.render(mode="rgb_array")
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                imageio.imwrite(frame_path, frame)
                frames.append(frame.copy())
                frame_idx += 1

            episode_over = terminated or truncated

        if save_video:
            video_path = os.path.join(video_root, f"episode_{episode}.mp4")
            if frames:
                imageio.mimsave(video_path, frames, fps=1)
                logger.info(f"[Episode {episode}] Video saved to {video_path} with {len(frames)} frames")
            else:
                logger.info(f"[Episode {episode}] No frames recorded, skipping video save.")
            if frames_dir and os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir)

        episodes_data.append(step_log)

    return episodes_data


def convert2nodeidx_unweighted_graph(position: tuple[int, int, float]) -> int:
    """
    Convert a position tuple (start, end, dist) from an unweighted graph to the node index.
    Assumes start == end and dist == 0.0 for positions returned by the environment.
    """
    if not isinstance(position, tuple) or len(position) != 3:
        raise ValueError("Position must be a tuple of (start, end, dist).")
    start, end, dist = position
    if start != end or dist != 0.0:
        raise ValueError(f"Expected start==end and dist==0.0, got {position}.")
    return int(start)


class WandbLogger:
    """
    Thin wrapper to send metrics to Weights & Biases without coupling runners
    to wandb import paths or metric names.
    """

    def __init__(self, project: str, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError("wandb is required for WandbLogger but is not installed.") from exc
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config)

    def __call__(self, metrics: dict[str, Any], context: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {}
        if context:
            payload.update(context)
        if metrics:
            payload.update(metrics)
        if payload:
            self.wandb.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.wandb.finish()
