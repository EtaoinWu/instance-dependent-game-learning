import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Integer, Scalar

from onlineax.common import typed


@typed
def choice(
    p: Float[Array, " n"], draws: Float[Scalar, ""]
) -> Integer[Scalar, ""]:
    """
    Fast version of np.random.choice for a single draw.

    Parameters
    ----------
    p : float[n]
        Sampling distribution over arms. NOTE: not checked for sum to 1.
    draws : float
        Uniform random draw from [0, 1).

    Returns
    -------
    integer
        Index of the chosen arm.
    """
    return jnp.searchsorted(jnp.cumsum(p), draws)


_choice_v = vmap(choice, in_axes=(1, 0), out_axes=0)


@typed
def choice_v(
    ps: Float[Array, "n t"], draws: Float[Array, " t"]
) -> Integer[Array, " t"]:
    """
    Fast version of np.random.choice for multiple draws.

    Parameters
    ----------
    ps : float[n,t]
        Sampling distribution over arms. NOTE: not checked for sum to 1.
    draws : float[t]
        Uniform random draws from [0, 1).

    Returns
    -------
    integer[t]
        Indices of the chosen arms.
    """
    return _choice_v(ps, draws)
