import jax
import jax.numpy as jnp
from beartype.typing import override
from jax import (
    lax,
)
from jaxtyping import Array, Bool, Float, Integer, Scalar

from onlineax.common import typed
from onlineax.typing import typed_module

from .ftrl import (
    EtaRead,
    EtaWRead,
    FTRLBase,
    NothingWrite,
    WeightedFTRLBase,
)


@typed_module
class Hedge(FTRLBase[EtaRead, NothingWrite]):
    @override
    def project(
        self, total_loss: Float[Array, " n"]
    ) -> Float[Array, " n"]:
        return jax.nn.softmax(-self.eta * total_loss)


@typed_module
class HedgeW(WeightedFTRLBase[EtaWRead, NothingWrite], Hedge):
    pass


@typed
def asymmetric_shannon_naive(
    loss: Float[Array, " n"],
    gamma: Float[Array, " n"],
    x0: Float[Scalar, ""] | None = None,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
    """
    Optimization of the asymmetric shannon entropy function.
    Computes arg min_(w in Delta^n) <w, loss> - sum gamma_i w_i (log w_i - 1).

    Parameters
    ----------
    loss : float array of shape (n,)
        Loss vector.
    gamma : float array of shape (n,)
        Weight vector.
    x0 : float scalar, optional
        Initial point for the Lagrange multipliers
    max_iter : int, optional
        Maximum number of iterations, by default 10.

    Returns
    -------
    float array of shape (n,)
        Optimal point for the optimization problem.
    float scalar
        Lagrange multiplier.
    """

    min_loss = jnp.min(loss)
    x_bound = -min_loss
    if x0 is None:
        x0 = x_bound
    x = x0

    for _ in range(max_iter):
        x = jnp.maximum(x, x_bound)
        xs = jnp.exp(-(x + loss) / gamma)
        f = xs.sum() - 1
        f_prime = jnp.sum(-xs / gamma)
        x -= f / f_prime
    return jnp.exp(-(x + loss) / gamma), x


@typed
def asymmetric_shannon_opt(
    loss: Float[Array, " n"],
    gamma: Float[Array, " n"],
    x0: Float[Scalar, ""] | None = None,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
    """
    Newton optimization of the asymmetric shannon entropy function.
    Computes arg min_(w in Delta^n) <w, loss> - sum gamma_i w_i (log w_i - 1).

    Parameters
    ----------
    loss : float array of shape (n,)
        Loss vector.
    gamma : float array of shape (n,)
        Weight vector.
    x0 : float scalar, optional
        Initial point for the Lagrange multipliers
    max_iter : int, optional
        Maximum number of iterations, by default 10.

    Returns
    -------
    float array of shape (n,)
        Optimal point for the optimization problem.
    float scalar
        Lagrange multiplier.
    """

    min_loss = jnp.min(loss)
    x_bound = -min_loss
    if x0 is None:
        x0 = x_bound

    @typed
    def while_body(
        state: tuple[
            Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
        ],
    ) -> tuple[
        Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
    ]:
        x, f, i = state
        x = jnp.maximum(x, x_bound)
        xs = jnp.exp(-(x + loss) / gamma)
        f = xs.sum() - 1
        f_prime = jnp.sum(-xs / gamma)
        return x - f / f_prime, f, i + 1

    @typed
    def while_cond(
        state: tuple[
            Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
        ],
    ) -> Bool[Scalar, ""]:
        x, f, i = state
        return (i < max_iter) & (jnp.abs(f) > 1e-8)

    x1, _, _ = lax.while_loop(
        while_cond, while_body, (x0, jnp.array(jnp.inf), jnp.array(0))
    )
    return jnp.exp(-(x1 + loss) / gamma), x1
