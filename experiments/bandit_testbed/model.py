"""
experiments.bandit_testbed.model


Generate a true value function q
Generate a true reward distribution function
"""

from chex import PRNGKey, Array

import jax.random as jr
import jax.numpy as jnp
from jax import jit

def get_model(key, D):
    """
    Generate the true value function as a D-vector of independent standard Normal RVs  
        
    :param key: RNG
    :param D: Shape of q (ie number of levers in the Bandit problem)
    """

    q = jr.normal(key, shape=(D,))

    @jit
    def reward_func(key_: PRNGKey, action: Array, *_):
        """

        :param key_: RNG
        :param action: integer position of the action to take
        """
        action = jnp.asarray(action, dtype=jnp.int32)
        action = jnp.squeeze(action)  # ensure scalar
        noise = jr.normal(key_, ())
        
        return q[action] + noise
    
    return q, reward_func

