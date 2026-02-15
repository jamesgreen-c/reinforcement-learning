import argparse
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.bandit_testbed.kernels import KernelType
from experiments.bandit_testbed.model import get_model

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--D", dest="D", type=int, default=5)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)

parser.add_argument("--T", dest="T", type=int, default=20)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=None)
parser.add_argument("--delta", dest="delta", type=int, default=None)
parser.add_argument("--epsilon", dest="epsilon", type=int, default=None)

parser.add_argument("--style", type=str, default="stationary")

parser.add_argument("--optimistic-init", dest="optimistic_init", action='store_true')
parser.set_defaults(optimistic_init=False)

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

parser.set_defaults(plot=False)

args = parser.parse_args()

# BACKEND CONFIG
NOW = time.time()

# PARAMETERS
EPSILON = args.epsilon / 100 if args.epsilon is not None else None
DELTA = args.delta / 100 if args.delta is not None else None

# KEYS
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

kernel_type = KernelType(args.kernel)

print(f"""
###############################################
#           BANDIT TESTBED EXPERIMENT         #
###############################################
Configuration:
    Time: {NOW}
    - T: {args.T}
    - kernel: {KernelType(args.kernel).name}
    - epsilon: {EPSILON}
    - step-size: {DELTA}
    - D: {args.D}  (# of Levers)
    - T: {args.T}  (# of Steps)
    - M: {args.M}  (# of Independent Chains)
""")


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    """
    """

    data_key, init_key, sample_key = jax.random.split(key, 3)
    true_q, R = get_model(data_key, args.D)

    kernel, init, experiment_loop = kernel_type.kernel_maker(
        reward_func=R,
        epsilon=EPSILON,
        delta=DELTA,
        T=args.T,
        style=args.style
    )

    kernel = jax.jit(kernel)
    experiment_loop = jax.jit(experiment_loop, static_argnums=(2, 3, 4, 5))

    # This looks like it's using the true data, but it's not (see, the conditional=False above)
    # We only pass it for the shape of the data.
    init_keys = jax.random.split(init_key, args.M)
    sample_keys = jax.random.split(sample_key, args.M)

    init_qs = jax.vmap(init, in_axes=[None, None, 0])(true_q, args.optimistic_init, init_keys)

    def get_samples(sample_key_op, init_q_op, all_samples, n_samples):
        return experiment_loop(sample_key_op, init_q_op, kernel, n_samples, args.verbose, all_samples)

    Qs, As, Rs = jax.vmap(get_samples, in_axes=[0, 0, None, None], out_axes=1)(
        sample_keys, 
        init_qs, 
        True,
        args.T
    ) # Shapes Qs: (T, M, D), As: (T, M), Rs: (T, M)

    # jax.debug.print(
    #     "samples: Qs.shape={qs}, As.shape={as_}, Rs.shape={rs}",
    #     qs=jnp.asarray(Qs).shape,
    #     as_=jnp.asarray(As).shape,
    #     rs=jnp.asarray(Rs).shape,
    # )

    # only store the stats we need
    A_star = jnp.argmax(true_q)
    opt_pct    = jnp.mean(As == A_star, axis=1) * 100   # (T,)
    avg_reward = jnp.mean(Rs, axis=1)                   # (T,)
    avg_q_est  = jnp.mean(Qs[-1, :, :], axis=0)          # (D,)

    return opt_pct, avg_reward, avg_q_est, true_q, init_qs

final_pct_all = np.empty((args.K, args.T))
avg_Rs_all = np.empty((args.K, args.T))
avg_Q_all = np.empty((args.K, args.D))
true_q_all = np.empty((args.K, args.D))
init_qs_all = np.empty((args.K, args.M, args.D))

for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")
    opt_pct_k, avg_reward_k, avg_q_est_k, true_q_k, init_qs_k = one_experiment(key_k)

    final_pct_all[k, ...] = opt_pct_k
    avg_Rs_all[k, ...] = avg_reward_k
    avg_Q_all[k, ...] = avg_q_est_k
    true_q_all[k, ...] = true_q_k
    init_qs_all[k, ...] = init_qs_k

    print(f"""
Results:
    - Terminal optimal action (% over M): {opt_pct_k[-1]:.2f}
    - final Q:                            {avg_q_est_k}
""")
    print()

if not os.path.exists("results"):
    os.mkdir("results")

file_name = "results/kernel={},T={},D={},M={},K={},eps={},delta={}.npz"
file_name = file_name.format(kernel_type.name, args.T, args.D, args.M, args.K, args.epsilon, args.delta)

np.savez_compressed(
    file_name,
    final_pct=final_pct_all,
    avg_rs=avg_Rs_all, avg_Q=avg_Q_all,
    true_qs=true_q_all, init_qs=init_qs_all
)

