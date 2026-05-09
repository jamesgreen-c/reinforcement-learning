import jax
import jax.numpy as jnp


def get_random_walk_model(
    Ns: int = 19,
    p_success: float = 0.9,
    left_reward: float = -1.0,
    right_reward: float = 1.0,
):
    """
    Constructor for a tabular left/right random-walk MDP.

    Parameters
    ----------
    Ns:           Number of states, including the two terminal boundary states.
    p_success:    Probability that the intended left/right action is executed.
    left_reward:  Reward received when transitioning into the left terminal state.
    right_reward: Reward received when transitioning into the right terminal state.

    Returns
    -------
    model:     Callable
    states:    (Ns,) list of state indices.
    terminals: (2,) list of terminal state indices.
    actions:   (2,) list of action indices.
    """

    left_terminal = 0
    right_terminal = Ns - 1

    states = jnp.arange(Ns)
    terminals = jnp.array([left_terminal, right_terminal])
    actions = jnp.array([0, 1])

    def model(s, a, params=None):
        """
        Random-walk transition model p(s', r | s, a).

        Parameters
        ----------
        s:      Current state index.
        a:      Current action index. Action 0 moves left; action 1 moves right.
        params: Additional parameters, currently unused.

        Returns
        -------
        s_nexts: (2,) possible next states.
        rewards: (2,) rewards associated with each next state.
        probs:   (2,) transition probabilities.
        """

        is_terminal = (s == left_terminal) | (s == right_terminal)

        intended_step = jnp.where(a == 0, -1, 1)
        slipped_step = -intended_step

        s_success = jnp.clip(s + intended_step, 0, Ns - 1)
        s_slip = jnp.clip(s + slipped_step, 0, Ns - 1)

        s_nexts = jnp.array([s_success, s_slip])
        probs = jnp.array([p_success, 1.0 - p_success])

        rewards = jnp.where(
            s_nexts == right_terminal,
            right_reward,
            jnp.where(
                s_nexts == left_terminal,
                left_reward,
                0.0,
            ),
        )

        terminal_s_nexts = jnp.array([s, s])
        terminal_rewards = jnp.array([0.0, 0.0])
        terminal_probs = jnp.array([1.0, 0.0])

        return jax.lax.cond(
            is_terminal,
            lambda _: (terminal_s_nexts, terminal_rewards, terminal_probs),
            lambda _: (s_nexts, rewards, probs),
            operand=None,
        )

    return model, states, terminals, actions