"""
Implements a generic kernel for approximate policy iteration.
"""
from typing import Callable, Union, Any

import jax
from jax import vmap, numpy as jnp
from chex import Array, PRNGKey


def get_kernel(
        P: Union[Callable, tuple[Callable, Any]],
        J: Union[Callable, tuple[Callable, Any]],
        states: Array,
        actions: Array,
        gamma: Array
    ):
    """
    Constructor for the policy iteration kernel

    Parameters
    ----------
    P:       Policy (Action sampling function). Either a Callable or a tuple of Callable and its parameters
    J:       Joint model for states and rewards ie p(s', r | s, a). Either a Callable or a tuple of Callable and its parameters
    states:  List of all possible states the model can take on
    actions: List of all possible actions the policy can produce
    gamma:   Discount
    
    Returns
    -------
    kernel: Callable - The Policy Iteration Kernel

    """

    # Unpack functions
    _P, P_params = P if isinstance(P, tuple) else (P, None)
    _J, J_params = J if isinstance(J, tuple) else (J, None)
    P = lambda s, pi: _P(s, pi, P_params)    
    J = lambda s, a: _J(s, a, J_params)
    
    eval = vmap(lambda _s, _v, _pi: policy_evaluation(_s, gamma, P, _pi, J, _v), in_axes=(0, None, None))
    improve = vmap(lambda _s, _v: policy_improvement(J, actions, _s, gamma, _v), in_axes=(0, None))

    def kernel(pi: Array, V: Array, num_iter: int):
        """
        Implements a single-step of the policy iteration algorithm
        
        Parameters
        ----------
        pi:        Deterministic vector mapping states to actions for P
        V:         Initial value function estimate
        num_iter:  Number of iterations to run policy evaluation for
        Returns
        ------

        """

        # POLICY EVALUATION (E-step)
        def _body(v_k, inp):
            v_k_p_1 = eval(states, v_k, pi)
            return v_k_p_1, v_k_p_1
        V_T, V_hist = jax.lax.scan(_body, V, jnp.arange(num_iter))

        # POLICY IMPROVEMENT (M-step)
        pi_next = improve(states, V_T)

        return pi_next, V_T, V_hist
    
    return kernel


def policy_evaluation(s: Array, gamma: Array, P: Callable, pi: Array, J: Callable, V: Array):
    """
    For a given state, return the updated value for that state via direct application of the Bellman equation

    Parameters
    ----------
    s:     The state for which the policy is to be evaluated
    gamma: Discount
    P:     Policy (Action sampling function).
    pi:    Deterministic vector mapping states to actions for P
    J:     Joint model for states and rewards ie p(s', r | s, a).
    V:     Current value function estimate

    Returns
    -------
    V(s)   The updated value for state s
    """

    a = P(s, pi)
    s_nexts, rewards, probs = J(s, a)                               # probs: (Ns, Nr)
    targets = rewards[None, :] + gamma * V.take(s_nexts)[:, None]   # (Ns, Nr)
    return jnp.sum(probs * targets)
    

def policy_improvement(J: Callable, actions: Array, s: Array, gamma: Array, V: Array):
    """
    For a given state, greedily selection the action which has the highest estimated value

    Parameters
    ----------
    J:        Joint model for states and rewards ie p(s', r | s, a).
    actions:  List of all possible actions
    s:        The state for which to greedily improve the policy
    gamma:    Discount
    V:        Current value function estimate

    Returns
    -------
    pi(s)   The greedily updated action for state s under a new policy
    """

    @vmap
    def _f(a):
        s_nexts, rewards, probs = J(s, a)     # probs: (Ns, Nr)
        targets = rewards[None, :] + gamma * V.take(s_nexts)[:, None]   # (Ns, Nr)
        return jnp.sum(probs * targets)
    
    return actions[jnp.argmax(_f(actions))]
 