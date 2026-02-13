"""
Implements a generic kernel for a single-step of the bandit learning problem.
"""
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey


def get_kernel(
        P: Union[Callable, tuple[Callable, Any]],
        R: Union[Callable, tuple[Callable, Any]],
        V: Union[Callable, tuple[Callable, Any]]   
    ):
    """
    Constructor for the bandit kernel

    Parameters
    ----------
    P:     Policy (Action sampling function). Either a Callable or a tuple of Callable and its parameters
    R:     Reward sampling function. Either a Callable or a tuple of Callable and its parameters
    V:     Function to update value estimates. Should take as input, and return as output, the whole vector of value estimates
    
    Returns
    -------
    kernel: Callable - The Bandit Kernel

    """

    P, P_params = P if isinstance(P, tuple) else (P, None)
    R, R_params = R if isinstance(R, tuple) else (R, None)
    V, V_params = V if isinstance(V, tuple) else (V, None)

    def kernel(key: PRNGKey, q: Array, delta: Array):
        """
        Implements a single-step of the K-armed bandit solver, where K is the length of the vector q 
        
        Parameters
        ----------
        key:      JAX PRNGKey
        q:        Current value function estimate
        delta:    Step-size parameter for the value estimate update
        
        Returns
        ------
        q_next:  Next estimate of the value function

        """
        key_policy, key_reward = jax.random.split(key)

        a_t = P(key_policy, q, P_params)
        r_t = R(key_reward, a_t, R_params)
        q_next = V(r_t, q, delta, V_params)

        return q_next, a_t, r_t
    
    return kernel