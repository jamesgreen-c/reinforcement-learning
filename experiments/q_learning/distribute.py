# ARGS PARSING
import argparse
import os

from itertools import product
import numpy as np
from algorithms.utils.printing import ctext

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, default=-1)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--episodes", default=10)
parser.add_argument("--seed", type=int, default=123)
parser.set_defaults(debug=False)
args = parser.parse_args()


def results_exist(*, N, sigma, alpha, epsilon, gamma) -> bool:
    """ Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    experiment_name = "n={},sigma={},alpha={},epsilon={},gamma={},episodes={},K={},seed={}"
    experiment_name = experiment_name.format(
        N,
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
SIGMAs = (0, 1.0, )
ALPHAs = np.round(np.linspace(0.01, 1, num=19), 3)
EPISLONs = (0.1, )
GAMMAs = (0.9, )


combination = list(product(Ns, SIGMAs, ALPHAs, EPISLONs, GAMMAs))
print(f"Number of experiments: {len(combination)}")

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices[:3]:
    N, sigma, alpha, eps, gamma = combination[j]

    if results_exist(N=N, sigma=sigma, alpha=alpha, epsilon=eps, gamma=gamma):
        print(ctext(
            f"Skipping (already run): N={N}, sigma={sigma}, alpha={alpha}, epsilon={eps}, gamma={gamma}", "yellow"
        ))
        continue

    exec_str = "python3 experiment.py --N {} --sigma {} --alpha {} --gamma {} --epsilon {} --K {} --episodes {} --seed {}"
    exec_str = exec_str.format(N, sigma, alpha, gamma, eps, args.K, args.episodes, args.seed)
    print("\nExecuting:", ctext(exec_str, "green"))
    os.system(exec_str)
