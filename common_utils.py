from os.path import join, exists
from datetime import datetime
import os
import time
import torch
import numpy as np
import random

def directory_config(path, fold_name=None):
    if not exists(path):
        os.makedirs(path)
    if fold_name is None:
        fold_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = join(path, str(fold_name))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def store_args(args, save_path):
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        for arg in vars(args):
            if arg=='device_id':
                continue
            value = getattr(args, arg)
            file.write(f'{arg}: {value}\n') 

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string

def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)

def arg_max(state_action):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)