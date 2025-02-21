from jax import numpy as jnp
from jaxtyping import Array, Float

from onlineax.typing import typed


@typed
def proj_sim(x: Float[Array, " n"]) -> Float[Array, " n"]:
    """Project onto the unit simplex.
    y = argmin_{z in ∆^n} |z - x|

    Parameters
    ----------
    x: Float[Array, " n"]
        Input vector.

    Returns
    -------
    y: Float[Array, " n"]
        Projected vector.
    """
    d = x.shape[0]
    u = jnp.sort(x)[::-1]
    v = jnp.cumsum(u)
    j = jnp.arange(1, d + 1)
    cond = u - (v - 1) / j > 0
    ρ = d - 1 - jnp.argmax(cond[::-1])
    λ = (1 - v[ρ]) / (ρ + 1)
    return jnp.maximum(x + λ, 0)
