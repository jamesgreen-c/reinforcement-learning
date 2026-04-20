"""
Implement the algorithm for Monte Carlo off-policy control.
"""

from typing import Callable, Union, Any

import jax
from jax import numpy as jnp
from chex import Array, PRNGKey


def get_kernel(
    Q0: Array,
    B: Union[Callable, tuple[Callable, Any]],
    J: Union[Callable, tuple[Callable, Any]],
    states: Array,
    gamma: Array,
    T: Array,
):
    """
    Constructor for the Monte Carlo off-policy control kernel.

    Parameters
    ----------
    Q0:      (Ns, Na) initial Q-table, used only for shape inference
    B:       Behaviour policy. Either a Callable or (Callable, params). Returns action and action probability b(a|s)
    J:       Joint model p(s', r | s, a). Either a Callable or (Callable, params).
    states:  (Ns,) list of state indices
    gamma:   Discount factor
    T:       Episode length

    Returns
    -------
    kernel: Callable
    """
    Na = Q0.shape[1]

    # Unpack functions
    _B, B_params = B if isinstance(B, tuple) else (B, None)
    _J, J_params = J if isinstance(J, tuple) else (J, None)
    B = lambda key, s, pi: _B(key, s, pi, B_params)
    J = lambda s, a: _J(s, a, J_params)

    def kernel(key: PRNGKey, pi: Array, Q: Array, C: Array):
        """
        One episode of off-policy MC control.

        Parameters
        ----------
        pi:  (Ns, Na) deterministic target policy in one-hot form
        Q:   (Ns, Na) action-value table
        C:   (Ns, Na) cumulative sum of importance weights

        Returns
        -------
        pi_next, Q_next, C_next
        """
        key_init, key_ep = jax.random.split(key)

        # sample trajectory
        s0 = jax.random.choice(key_init, states)
        actions, action_probs, ep_states, rewards = episode(key_ep, s0, pi, B, J, T)

        # reverse for backward MC return updates
        rev_states = ep_states[::-1]
        rev_actions = actions[::-1]
        rev_rewards = rewards[::-1]
        rev_b_probs = action_probs[::-1]

        def active_step(carry, inp):
            """
            Apply one genuine backward update.
            """
            Q_k, C_k, G_k, W_k, done_k = carry
            s_k, a_k, r_kp1, b_prob_k = inp

            # Return update
            G_k = gamma * G_k + r_kp1

            # C[s,a] += W
            C_sa_new = C_k[s_k, a_k] + W_k
            C_k = C_k.at[s_k, a_k].set(C_sa_new)

            # Q[s,a] += (W/C[s,a]) * (G - Q[s,a])
            Q_sa_old = Q_k[s_k, a_k]
            Q_sa_new = Q_sa_old + (W_k / C_sa_new) * (G_k - Q_sa_old)
            Q_k = Q_k.at[s_k, a_k].set(Q_sa_new)

            # Stop if sampled action is no longer greedy
            greedy_a = jnp.argmax(Q_k[s_k])
            done_k = (a_k != greedy_a)

            # update W
            W_k = W_k / b_prob_k

            return (Q_k, C_k, G_k, W_k, done_k), None

        def noop_step(carry, inp):
            return carry, None

        def body(carry, inp):
            done_k = carry[-1]
            return jax.lax.cond(
                done_k,
                noop_step,
                active_step,
                carry,
                inp,
            )

        G_0, W_0, flag_0 = jnp.array(0.0, Q.dtype), jnp.array(1.0, Q.dtype), jnp.array(False)
        carry0 = (Q, C, G_0, W_0, flag_0)

        (Q_T, C_T, _, _, _), _ = jax.lax.scan(
            body,
            carry0,
            (rev_states, rev_actions, rev_rewards, rev_b_probs),
        )

        pi_next = greedy_policy_matrix(Q_T, Na)
        return pi_next, Q_T, C_T

    return kernel


def episode(
    key: PRNGKey,
    state_0: Array,
    pi: Array,
    B: Callable,
    J: Callable,
    T: Array,
):
    """
    Sample one episode under the behaviour policy.

    Returns
    -------
    actions:      (T,)
    action_probs: (T,)   with b(a_t | s_t)
    states:       (T,)   states S_0, ..., S_{T-1}
    rewards:      (T,)   rewards R_1, ..., R_T
    """

    def _body(s_k, key):
        key_b, key_j = jax.random.split(key)

        # sample from behaviour policy and also record b(a_k | s_k)
        a_k, a_prob_k = B(key_b, s_k, pi)

        # sample next state and reward from environment
        s_nexts, rewards, probs = J(s_k, a_k)
        choice = jax.random.categorical(key_j, jnp.log(probs))
        s_kp1 = s_nexts[choice]
        r_kp1 = rewards[choice]

        return s_kp1, (a_k, a_prob_k, s_k, r_kp1)

    keys = jax.random.split(key, T)
    _, (actions, action_probs, states, rewards) = jax.lax.scan(_body, state_0, keys)

    return actions, action_probs, states, rewards


def greedy_policy_matrix(Q: Array, Na: Array) -> Array:
        """
        Deterministic policy in matrix form: (Ns, Na),
        with one-hot rows corresponding to argmax_a Q(s,a).

        Parameters
        ----------
        Q:  (Ns, Na) Value matrix
        Na: Number of actions

        Returns
        -------
        Pi: One-hot policy (ie greedy policy wrt Q)
        """
        greedy_actions = jnp.argmax(Q, axis=1)
        return jax.nn.one_hot(greedy_actions, num_classes=Na, dtype=Q.dtype)