import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array, Float, Integer, Scalar

from onlineax.common import typed

# (action: int, loss: float, weights: float[n]) -> loss': float[n]
type Estimator = Callable[
    [Integer[Scalar, ""], Float[Scalar, ""], Float[Array, " n"]],
    Float[Array, " n"],
]


@typed
def importance_weighed_estimator(
    action: Integer[Scalar, ""],
    loss: Float[Scalar, ""],
    weights: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Importance weighted estimator.

    `loss'[i] = loss / weights[i] if i == action else 0`
    """

    return (
        jnp.zeros_like(weights).at[action].set(loss / weights[action])
    )


@typed
def variance_reduced_estimator(
    action: Integer[Scalar, ""],
    loss: Float[Scalar, ""],
    weights: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Variance reduced estimator.

    `loss'[i] = ((loss - 0.5) / weights[i] if i == action else 0) + 1/2
    """

    return (
        (jnp.ones_like(weights) * 0.5)
        .at[action]
        .add((loss - 0.5) / weights[action])
    )
