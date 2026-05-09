import jax
import jax.numpy as jnp


def tabular_PR(model, Ns, Na):
    """
    Constructs full tabular transition and reward tensors.

    Parameters
    ----------
    model: Callable transition model returning s_nexts, rewards, probs.
    Ns:    Number of states.
    Na:    Number of actions.

    Returns
    -------
    P: (Ns, Na, Ns) transition probability tensor.
    R: (Ns, Na, Ns) reward tensor.
    """

    P = jnp.zeros((Ns, Na, Ns))
    R = jnp.zeros((Ns, Na, Ns))

    for s in range(Ns):
        for a in range(Na):
            s_nexts, rewards, probs = model(s, a)

            P = P.at[s, a, s_nexts].add(probs)
            R = R.at[s, a, s_nexts].set(rewards)

    return P, R


def true_q_star(model, Ns, Na, gamma, tol=1e-10, max_iter=10_000):
    """
    Computes the true q_* using value iteration.

    Parameters
    ----------
    model:    Callable transition model returning s_nexts, rewards, probs.
    Ns:       Number of states.
    Na:       Number of actions.
    gamma:    Discount factor.
    tol:      Convergence tolerance.
    max_iter: Maximum number of value-iteration steps.

    Returns
    -------
    Q_star: (Ns, Na) optimal action-value table.
    """

    P, R = tabular_PR(model, Ns, Na)
    Q = jnp.zeros((Ns, Na))

    def body(carry):
        Q, i, err = carry

        V = jnp.max(Q, axis=1)  # (Ns,)

        Q_next = jnp.sum(
            P * (R + gamma * V[None, None, :]),
            axis=-1,
        )

        err = jnp.max(jnp.abs(Q_next - Q))

        return Q_next, i + 1, err

    def cond(carry):
        _, i, err = carry
        return (i < max_iter) & (err > tol)

    Q, _, _ = jax.lax.while_loop(
        cond,
        body,
        (Q, 0, jnp.inf),
    )

    return Q