"""
Implementation of off-policy n-step Q(sigma).

This generalises Sarsa, Expected Sarsa, Tree Backup, and Q-learning-style
updates through the interpolation parameter sigma.
"""

from typing import Union, Callable, Any

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


def get_kernel(
    B: Union[Callable, tuple[Callable, Any]],
    J: Union[Callable, tuple[Callable, Any]],
    states: Array,
    terminals: Array,
    gamma: Array,
    alpha: float = 0.1,
    n: int = 4,
    sigma: Union[float, Callable] = 1.0,
    max_iter: int = 1000,
    learn_pi: bool = True,
):
    """
    Constructor for the off-policy n-step Q(sigma) kernel.

    Parameters
    ----------
    B:         Behaviour policy. Either a Callable or (Callable, params). Returns action and action probability b(a|s).
    J:         Joint model p(s', r | s, a). Either a Callable or (Callable, params).
    states:    (Ns,) list of state indices.
    terminals: (Nt,) list of terminal state indices.
    gamma:     Discount factor.
    alpha:     Step size for the Q-update.
    n:         Number of steps used in the return.
    sigma:     Interpolation parameter. sigma=1 gives sampled Sarsa-style updates; sigma=0 gives expectation/tree-backup updates.
    max_iter:  Maximum number of scan iterations per episode.
    learn_pi:  If True, updates pi greedily wrt Q after each Q-update. If False, pi is fixed.

    Returns
    -------
    kernel: Callable
    """

    # Unpack functions
    _B, B_params = B if isinstance(B, tuple) else (B, None)
    _J, J_params = J if isinstance(J, tuple) else (J, None)
    B = lambda _k, _s, _pi: _B(_k, _s, _pi, B_params)
    J = lambda _s, _a: _J(_s, _a, J_params)

    sigma_fn = sigma if callable(sigma) else lambda _s, _a, _t: jnp.asarray(sigma)

    buffer_size = n + 1
    terminal_time_init = max_iter + 1

    def is_terminal(s):
        return jnp.any(s == terminals)

    def sample_nonterminal_state(key):
        mask = ~jax.vmap(is_terminal)(states)
        probs = mask.astype(jnp.float32)
        probs = probs / probs.sum()
        return jax.random.choice(key, states, p=probs)
    
    def update_value(t, tau, T, S, A, R, rho, sigmas, pi, Q):
        """
        Applies the n-step Q(sigma) update for the state-action pair at time tau.

        Parameters
        ----------
        t:       Current time index.
        tau:     Time index whose estimate is being updated.
        T:       Terminal time of the episode.
        S:       (n + 1,) cyclic buffer of stored states.
        A:       (n + 1,) cyclic buffer of stored actions.
        R:       (n + 1,) cyclic buffer of stored rewards.
        rho:     (n + 1,) cyclic buffer of importance ratios pi(a|s) / b(a|s).
        sigmas:  (n + 1,) cyclic buffer of sigma values.
        pi:      (Ns, Na) target policy matrix.
        Q:       (Ns, Na) action-value table.

        Returns
        -------
        pi_next: (Ns, Na) updated target policy matrix.
        Q_next:  (Ns, Na) updated action-value table.
        """

        K = jnp.minimum(t + 1, T)
        num_backwards_steps = K - tau

        def body(i, G):
            k = K - i
            k_mod = k % buffer_size

            S_k = S[k_mod]
            A_k = A[k_mod]
            R_k = R[k_mod]
            rho_k = rho[k_mod]
            sigma_k = sigmas[k_mod]

            Vbar = jnp.sum(pi[S_k] * Q[S_k])
            interpolation = sigma_k * rho_k + (1.0 - sigma_k) * pi[S_k, A_k]
            G_new = R_k + gamma * interpolation * (G - Q[S_k, A_k]) + gamma * Vbar

            G = jax.lax.cond(
                k == T,
                lambda _: R_k,
                lambda _: G_new,
                operand=None,
            )

            return G

        # G0 = jnp.asarray(0.0)
        K_mod = K % buffer_size
        G0 = jax.lax.cond(
            K < T,
            lambda _: Q[S[K_mod], A[K_mod]],
            lambda _: jnp.asarray(0.0, dtype=Q.dtype),
            operand=None,
        )
        G = jax.lax.fori_loop(0, num_backwards_steps, body, G0)

        tau_mod = tau % buffer_size
        S_tau = S[tau_mod]
        A_tau = A[tau_mod]

        Q = Q.at[S_tau, A_tau].add(alpha * (G - Q[S_tau, A_tau]))

        pi = jax.lax.cond(
            learn_pi,
            lambda Q_: update_pi(Q_),
            lambda Q_: pi,
            Q,
        )

        return pi, Q

    def kernel(key: PRNGKey, pi: Array, Q: Array):
        """
        Runs one episode of off-policy n-step Q(sigma).

        Parameters
        ----------
        key: PRNG key.
        pi:  (Ns, Na) target policy matrix.
        Q:   (Ns, Na) action-value table.

        Returns
        -------
        pi_next: (Ns, Na) updated target policy matrix.
        Q_next:  (Ns, Na) updated action-value table.
        """

        key_init, key_a0, key_scan = jax.random.split(key, 3)

        # Initial state S_0, non-terminal
        S0 = sample_nonterminal_state(key_init)

        # Initial action A_0 ~ b(. | S_0)
        A0, b_prob0 = B(key_a0, S0, pi)
        rho0 = pi[S0, A0] / b_prob0
        sigma0 = sigma_fn(S0, A0, 0)

        S = jnp.zeros((buffer_size,), dtype=states.dtype)
        A = jnp.zeros((buffer_size,), dtype=jnp.int32)
        R = jnp.zeros((buffer_size,), dtype=Q.dtype)
        rho = jnp.ones((buffer_size,), dtype=Q.dtype)
        sigmas = jnp.ones((buffer_size,), dtype=Q.dtype)

        S = S.at[0].set(S0)
        A = A.at[0].set(A0)
        rho = rho.at[0].set(rho0)
        sigmas = sigmas.at[0].set(sigma0)

        T = jnp.asarray(terminal_time_init)
        done = jnp.asarray(False)

        carry = (pi, Q, S, A, R, rho, sigmas, T, done)

        keys = jax.random.split(key_scan, max_iter)

        def body(carry, inp):
            pi, Q, S, A, R, rho, sigmas, T, done = carry
            key_t, t = inp

            tau = t - n + 1

            def sample_step(args):
                pi, Q, S, A, R, rho, sigmas, T, done = args

                t_mod = t % buffer_size
                tp1_mod = (t + 1) % buffer_size

                S_t = S[t_mod]
                A_t = A[t_mod]

                # Environment step from stored S_t, A_t
                key_j = key_t
                s_nexts, rewards, probs = J(S_t, A_t)
                choice = jax.random.categorical(key_j, jnp.log(probs))

                S_tp1 = s_nexts[choice]
                R_tp1 = rewards[choice]

                terminal_tp1 = is_terminal(S_tp1)

                T = jax.lax.cond(
                    terminal_tp1,
                    lambda _: t + 1,
                    lambda _: T,
                    operand=None,
                )

                S = S.at[tp1_mod].set(S_tp1)
                R = R.at[tp1_mod].set(R_tp1)

                def choose_next_action(args):
                    pi, S, A, rho, sigmas = args

                    key_b = jax.random.fold_in(key_t, 123)
                    A_tp1, b_prob_tp1 = B(key_b, S_tp1, pi)

                    rho_tp1 = pi[S_tp1, A_tp1] / b_prob_tp1
                    sigma_tp1 = sigma_fn(S_tp1, A_tp1, t + 1)

                    A = A.at[tp1_mod].set(A_tp1)
                    rho = rho.at[tp1_mod].set(rho_tp1)
                    sigmas = sigmas.at[tp1_mod].set(sigma_tp1)

                    return pi, S, A, rho, sigmas

                pi, S, A, rho, sigmas = jax.lax.cond(
                    terminal_tp1,
                    lambda args: args,
                    choose_next_action,
                    (pi, S, A, rho, sigmas),
                )

                return pi, Q, S, A, R, rho, sigmas, T, done

            # Only sample while t < T
            pi, Q, S, A, R, rho, sigmas, T, done = jax.lax.cond(
                t < T,
                sample_step,
                lambda args: args,
                (pi, Q, S, A, R, rho, sigmas, T, done),
            )

            # Update only when tau >= 0
            def update_step(args):
                pi, Q, S, A, R, rho, sigmas, T, done = args

                pi, Q = update_value(
                    t=t,
                    tau=tau,
                    T=T,
                    S=S,
                    A=A,
                    R=R,
                    rho=rho,
                    sigmas=sigmas,
                    pi=pi,
                    Q=Q,
                )

                done = done | (tau == (T - 1))

                return pi, Q, S, A, R, rho, sigmas, T, done

            pi, Q, S, A, R, rho, sigmas, T, done = jax.lax.cond(
                (tau >= 0) & (tau < T) & (~done),
                update_step,
                lambda args: args,
                (pi, Q, S, A, R, rho, sigmas, T, done),
            )

            return (pi, Q, S, A, R, rho, sigmas, T, done), None

        ts = jnp.arange(max_iter)

        carry, _ = jax.lax.scan(
            body,
            carry,
            (keys, ts),
        )

        pi, Q, *_ = carry

        return pi, Q

    return kernel


# def sample(key, s_k, pi, B, J):
#     """
#     Samples A_k ~ b(. | S_k), then samples S_{k+1}, R_{k+1}.
#     Also returns rho_k = pi(A_k | S_k) / b(A_k | S_k).
#     """
#     key_b, key_j = jax.random.split(key)

#     a_k, b_prob_k = B(key_b, s_k, pi)

#     pi_prob_k = pi[s_k, a_k]
#     rho_k = pi_prob_k / b_prob_k

#     s_nexts, rewards, probs = J(s_k, a_k)
#     choice = jax.random.categorical(key_j, jnp.log(probs))

#     s_kp1 = s_nexts[choice]
#     r_kp1 = rewards[choice]

#     return a_k, s_kp1, r_kp1, rho_k


def update_pi(Q):
    """
    Constructs a greedy policy wrt Q.

    Ties are handled by assigning uniform probability across all greedy actions.

    Parameters
    ----------
    Q: (Ns, Na) action-value table.

    Returns
    -------
    pi: (Ns, Na) greedy policy matrix.
    """
    q_max = jnp.max(Q, axis=-1, keepdims=True)
    greedy = Q == q_max
    return greedy / greedy.sum(axis=-1, keepdims=True)