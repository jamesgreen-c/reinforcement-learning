import jax 
import jax.random as jr
import jax.numpy as jnp
from chex import Array, PRNGKey


def weighted_average(r: Array, a: Array, q: Array, delta: Array, _):
    """
    Calculates the incremental weighted average update step
    
    Parameters
    ----------
    r:      Current sampled reward
    q:      Current value function estimate
    delta:  Step-size
    _:      Unused parameters
    
    Returns
    -------
    q_next: The updated value function estimate

    """
    a = jnp.asarray(a, dtype=jnp.int32)
    td = delta * (r - q[a])
    return q.at[a].add(td)

