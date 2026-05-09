import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from algorithms.utils.bellman import true_q_star
from experiments.q_learning.kernels import build_kernel
from experiments.q_learning.environment import get_random_walk_model

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--K", type=int, default=1)

parser.add_argument("--sigma", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--gamma", type=float)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--N", type=int, default=1)

parser.add_argument("--learn-pi", dest="learn_pi", action="store_true")
parser.set_defaults(learn_pi=True)

parser.add_argument("--episodes", default=10)
parser.add_argument("--max-iter", dest="max_iter", default=1000)

parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

args = parser.parse_args()

# MODEL
MODEL, STATES, TERMINALS, ACTIONS = get_random_walk_model()
GAMMA = jnp.clip(args.gamma, 0.0, 1.0)
SIGMA = jnp.clip(args.sigma, 0.0, 1.0)
ALPHA = jnp.clip(args.alpha, 0.0, 1.0)
EPSILON = jnp.clip(args.epsilon, 0.0, 1.0)

Ns = STATES.shape[0]
Na = ACTIONS.shape[0]
TRUE_Q = true_q_star(MODEL, Ns, Na, GAMMA)

print(f"""
######################################
#    N-STEP Q LEARNING EXPERIMENT    #
######################################
Configuration:
    - N Steps:   {args.N}
    - sigma:     {SIGMA}
    - alpha:     {ALPHA}
    - Epsilon    {EPSILON}
    - Gamma      {GAMMA}
    
    - Episodes   {args.episodes}
    - Repeats    {args.K}
""")

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):

    kernel, inits, experiment_loop = build_kernel(
        model=MODEL, states=STATES, terminals=TERMINALS, actions=ACTIONS, 
        n=args.N, gamma=GAMMA, sigma=SIGMA, alpha=ALPHA, epsilon=EPSILON,
        max_iter=args.max_iter, learn_pi=args.learn_pi  
    )

    kernel = jax.jit(kernel)
    init_pi, init_Q = inits

    def get_samples(sampling_key, n_samples, get_all_samples):
        return experiment_loop(sampling_key, init_pi, init_Q, kernel, n_samples, get_all_samples)

    @jax.vmap
    def rms(q):
        return jnp.sqrt((TRUE_Q - q)**2)


    samples = get_samples(key, args.episodes, True)
    all_qs = samples[-1]
    rms_all = rms(all_qs)
    avg_rms = rms_all.mean()
    return avg_rms

rms_all = np.empty((args.K, ))
for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    rms_k = one_experiment(key_k)
    
    rms_all[k] = rms_k

average_rms = np.mean(rms_all)
print(f"""
   Average RMS error: {average_rms}   
""")

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "n={},sigma={},alpha={},epsilon={},gamma={},episodes={},K={},seed={}"
experiment_name = experiment_name.format(
    args.N,
    SIGMA,
    ALPHA,
    EPSILON,
    GAMMA,
    args.episodes,
    args.K,
    args.seed
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath,
    rms=np.array([average_rms])
)