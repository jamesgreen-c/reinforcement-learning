from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp

from algorithms.bandit import bandit as bdt
from algorithms.bandit.policy import epsilon_greedy
from algorithms.bandit.value import weighted_average

from algorithms.utils.sampling import sampling_routine

class KernelType(Enum):
    EPS_GREEDY = 0
    UCB = 1
    GRADIENT = 2

    @property
    def kernel_maker(self):
        if self == KernelType.EPS_GREEDY:
            return get_egreedy_kernel
        elif self == KernelType.UCB:
            return get_ucb_kernel
        elif self == KernelType.GRADIENT:
            return get_gradient_bandit_kernel
        else:
            raise NotImplementedError


#######################
# Kernel constructors #
#######################


def get_egreedy_kernel(reward_func, epsilon, delta, T, style="stationary", **_kwargs):

    @jax.jit
    def init(q, optimistic, key):
        # always start at 0 or 10
        q = jnp.zeros(q.shape[0])
        q += optimistic * 10
        return q

    if epsilon is None:
        raise ValueError("Argument[epsilon] cannot be None for e-greedy kernel. Set to 0 for greedy.")

    P_plus_params = (epsilon_greedy, epsilon)
    kernel = bdt.get_kernel(P_plus_params, reward_func, weighted_average)
    
    if style == "stationary":
        DELTAS = 1 / jnp.arange(1, T + 1)
    elif style == "non-stationary":
        if delta is None:
            raise ValueError("Argument[delta] cannot be None for Non-Stationary style")
        DELTAS = jnp.repeat(delta, T)

    def sampling_routine_fn(key, init_q, kernel_, n_steps, verbose, get_samples):
        samples = sampling_routine(key, init_q, DELTAS, kernel_, n_steps, verbose, get_samples)
        return samples

    return kernel, init, sampling_routine_fn


def get_ucb_kernel(reward_func, epsilon, delta, T, style="stationary", **_kwargs):
    raise NotImplementedError


def get_gradient_bandit_kernel(reward_func, epsilon, delta, T, style="stationary", **_kwargs):
    raise NotImplementedError



