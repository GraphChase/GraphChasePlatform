import copy

import numpy as np

from .game_config import get_game


def _valid_game(args, game, min_attacker_pth_len: int) -> bool:
    if args.action_type == "exit_node":
        path_length = np.array([len(paths[0]) if len(paths) > 0 else 0 for paths in game.attacker_path.values()])
        return sum(path_length > 0) > 0 and min(path_length[path_length > 0]) >= min_attacker_pth_len
    return len(game.attacker_path) > 0


def sample_game(args, default_game=None, min_attacker_pth_len: int = 0):
    game, action_type, config = get_game(args)
    if _valid_game(args, game, min_attacker_pth_len):
        return game, action_type, True, config
    if default_game is None:
        raise ValueError("Fixed graph settings do not yield valid attacker paths")
    return copy.deepcopy(default_game), action_type, False, config
