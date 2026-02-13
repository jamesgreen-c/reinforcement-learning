import jax 
import jax.random as jr
import jax.numpy as jnp
from chex import Array, PRNGKey


@jax.jit
def epsilon_greedy(key: PRNGKey, q: Array, epsilon: Array):
    """
    Epsilon-greedy action selection.

    With prob epsilon: pick random action in {0, ..., A-1}
    With prob 1-epsilon: pick argmax(q)

    Parameters
    ----------
    key:        JAX PRNGKey
    q:          Current value function estimate 
    epsilon:    Probability of selecting a random action
    
    Returns
    -------
    action:     Index corresponding to the action to take. (In bandits which lever to pull)

    """
    eps = jnp.clip(epsilon, 0.0, 1.0)

    k1, k2 = jr.split(key, 2)
    explore = jr.bernoulli(k1, p=eps)                  # bool scalar
    a_rand = jr.randint(k2, (), 0, q.shape[0])         # int32 scalar
    a_greedy = jnp.argmax(q).astype(jnp.int32)         # int32 scalar

    return jax.lax.select(explore, a_rand, a_greedy)


    