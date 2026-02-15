import jax
import jax.numpy as jnp


def sampling_routine(key,
                     init_q,
                     deltas,
                     kernel,
                     n_steps,
                     verbose=False,
                     get_samples=True):
    """
    Runs a generic sequential sampling routine using the provided kernel.

    At each step, the kernel (eg e-greedy Bandit) is applied to the current state
    which produces a new state (q_t, a_t, r_t). If get_samples is set, them we store
    all the samples and return them at the end, otherwise we just return the final state.
        
    :param key: rng
    :param init_q: Initial value function estimate (M, ...)
    :param deltas: The step-sizes at each time point
    :param kernel: Kernel function
    :param n_steps: Number of iterations
    :param verbose: Use a progress bar?  - currently broken with this JAX version
    :param get_samples: Whether to return all samples or just the final state
    """

    # if verbose:
    #     decorator = progress_bar_scan(n_steps, show=-1)
    # else:
    decorator = lambda x: x

    @decorator
    def body(carry, inp):
        i, key_op, delta = inp
        q = carry

        # Run kernel
        next_q, next_As, next_Rs, *_ = kernel(key_op, q, delta)
        carry_out = next_q

        return carry_out, (next_q, next_As, next_Rs) if get_samples else None

    inps = jnp.arange(n_steps), jax.random.split(key, n_steps), deltas
    final_q, all_samples = jax.lax.scan(body, init_q, inps)
    
    if get_samples:
        return all_samples
    else:
        return final_q
