import os
import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def read_file(file_name):
    data = np.load(file_name, allow_pickle=True)
    return data


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_line(data, save_name, color, smoothing_window=1, legend=None):
    fig = plt.figure(figsize=(10, 5))
    plt.style.use('seaborn')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    if legend is None:
        legend = [None for i in range(len(data))]
    for i in range(len(data)):
        x, y, std = zip(*data[i])
        x = x[smoothing_window-1:]
        y = moving_average(y, smoothing_window)
        std = moving_average(std, smoothing_window)

        plt.plot(x, y, linewidth=1., label=legend[i], color=color[i])
        plt.fill_between(x, y+std, y-std, alpha=0.3, facecolor=color[i])
    plt.legend()
    plt.savefig(save_name)



  