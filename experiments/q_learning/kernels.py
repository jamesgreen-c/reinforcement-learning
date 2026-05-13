from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import norm

from algorithms.td import nqs


######################
# Kernel constructor #
######################

def sampling_routine(key,
                     init_pi,
                     init_q,
                     kernel,
                     n_steps,
                     get_samples=True):
    """
    
    """

    def body(carry, key):
        pi, q = carry

        # Run kernel
        next_pi, next_q = kernel(key, pi, q)
        carry_out = next_pi, next_q
        return carry_out, (next_pi, next_q) if get_samples else None

    keys = jax.random.split(key, n_steps)
    final_q, all_samples = jax.lax.scan(body, (init_pi, init_q), keys)
    
    if get_samples:
        return all_samples
    else:
        return final_q


def build_kernel(
        model,
        states, 
        terminals, 
        actions, 
        n,
        gamma,
        sigma,
        alpha, 
        epsilon, 
        max_iter,
        learn_pi,
        **kwargs
    ):
    """
    Constructor for the tabular off-policy n-step Q(sigma) kernel.

    Parameters
    ----------
    model:      Callable, the joint model for states and rewards given current state action pair
    states:     (Ns,) list of state indices.
    terminals:  (Nt,) list of terminal state indices.
    actions:    (Na,) list of action indices.
    n:          Number of steps used in the return.
    gamma:      Discount factor.
    sigma:      Interpolation parameter. sigma=1 gives sampled Sarsa-style updates; sigma=0 gives expectation/tree-backup updates.
    alpha:      Step size for the Q-update.
    epsilon:    Exploration rate used by the epsilon-greedy behaviour policy.
    max_iter:   Maximum number of scan iterations per episode.
    learn_pi:   If True, updates pi greedily wrt Q after each Q-update. If False, pi is fixed.
    **kwargs:   Additional keyword arguments, currently unused.

    Returns
    -------
    kernel: Callable
    """
    Ns = states.shape[0]
    Na = actions.shape[0]

    def B(key, state, pi, epsilon):
        """
        Epsilon-greedy behaviour policy.

        Parameters
        ----------
        key:     PRNG key.
        state:   Current state index.
        pi:      (Ns, Na) target policy matrix.
        epsilon: Exploration rate.

        Returns
        -------
        a:      Sampled action index.
        b_prob: Behaviour probability b(a|s).
        """
        
        greedy_probs = pi[state]
        uniform_probs = jnp.ones(Na) / Na
        probs = (1.0 - epsilon) * greedy_probs + epsilon * uniform_probs

        a = jax.random.categorical(key, jnp.log(probs))
        b_prob = probs[a]
        return a, b_prob
    
    _B = (B, epsilon)
    kernel = nqs.get_kernel(
        B=_B,
        J=model,
        states=states,
        terminals=terminals,
        gamma=gamma,
        alpha=alpha,
        n=n,
        sigma=sigma,
        max_iter=max_iter,
        learn_pi=learn_pi
    )

    Q0 = jnp.zeros((Ns, Na))
    pi0 = jnp.ones((Ns, Na)) / Na
    return kernel, (pi0, Q0), sampling_routine 

