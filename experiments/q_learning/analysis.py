# ARGS PARSING
import argparse
import os

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from algorithms.utils.printing import ctext

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=123)
parser.set_defaults(debug=False)
args = parser.parse_args()


def results_exist(*, n, sigma, alpha, epsilon, gamma) -> bool:
    """ Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    experiment_name = "n={},sigma={},alpha={},epsilon={},gamma={},episodes={},K={},seed={}"
    experiment_name = experiment_name.format(
        n,
        sigma,
        alpha,
        epsilon,
        gamma,
        args.episodes,
        args.K,
        args.seed
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


Ns = (1, 2, 4, 8, 16, 32, 64, 128, )
ALPHAs = np.round(np.linspace(0.01, 1, num=19), 3)
EPSILON = 0.1
GAMMA = 0.9


def plot_rms_against_alpha(dirpath):

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    for n in Ns[:4]:
        RMS_s = np.empty((len(ALPHAs)))
        for j, alpha in enumerate(ALPHAs):
            sarsa_data, _ = load_data(n, 1.0, alpha, EPSILON, GAMMA)
            if sarsa_data is not None:
                RMS_s[j] = sarsa_data["rms"][0]
            else:
                RMS_s[j] = np.nan
        ax[0].plot(ALPHAs, RMS_s, label=f"n={n}")
    ax[0].set_title(f"Average RMS error over {args.episodes} episodes for Sarsa")
    ax[0].legend()

    for n in Ns:
        RMS_t = np.empty((len(ALPHAs)))
        for j, alpha in enumerate(ALPHAs):
            tree_backup_data, _ = load_data(n, 0.0, alpha, EPSILON, GAMMA)
            if tree_backup_data is not None:
                RMS_t[j] = tree_backup_data["rms"][0]
            else:
                RMS_t[j] = np.nan
        ax[1].plot(ALPHAs, RMS_t, label=f"n={n}")
    ax[1].set_title(f"Average RMS error over {args.episodes} episodes for Tree Backup")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(f"{dirpath}/rms.png", dpi=200)
    plt.close()


def load_data(n, sigma, alpha, epsilon, gamma):
    experiment_name = "n={},sigma={},alpha={},epsilon={},gamma={},episodes={},K={},seed={}"
    experiment_name = experiment_name.format(
        n,
        sigma,
        alpha,
        epsilon,
        gamma,
        args.episodes,
        args.K,
        args.seed
    )
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        return None, dirpath

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath

# plot 
plot_rms_against_alpha("results")

