import functools as ft

import jax
import jax.numpy as jnp
from beartype.typing import override
from jax import (
    lax,
)
from jaxtyping import Array, Bool, Float, Integer, Scalar

from onlineax.common import typed
from onlineax.typing import typed_module

from .ftrl import NewtonFTRLBase
from .types import EtaRead, NothingWrite


@typed
def tsallis_entropy(
    p: Float[Array, " n"], alpha: Float[Scalar, ""] = jnp.array(0.5)
) -> Float[Scalar, ""]:
    """Tsallis entropy of a probability distribution.

    Parameters
    ----------
    p : float array of shape (n,)
        Probability distribution.
    alpha : float scalar, optional
        Tsallis entropy parameter, by default 1/2.

    Returns
    -------
    float scalar
        Tsallis entropy of the probability distribution.
    """
    return jnp.sum(p**alpha) / (1 - alpha)


@typed
def tsallis_opt_naive(
    loss: Float[Array, " n"],
    eta: Float[Scalar, ""],
    x0: Float[Scalar, ""] | None = None,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
    """Naive implementation of optimization of Tsallis-inf with alpha=1/2.
    Computes arg min_(w in Delta^n) <w, loss> - 4/eta sum_i sqrt(w_i).

    Parameters
    ----------
    loss : float array of shape (n,)
        Loss vector.
    eta : float scalar
        Learning rate.
    x0 : float scalar, optional
        Initial point for the Lagrange multipliers
    max_iter : int, optional
        Maximum number of iterations, by default 8.


    Returns
    -------
    float array of shape (n,)
        Optimal probability distribution w.
    float scalar
        Optimal Lagrange multiplier.
    """

    n = loss.shape[0]
    min_loss: Array = jnp.min(loss)
    if x0 is None:
        x0 = min_loss * 0.5
    x0 = lax.select(x0 < min_loss, x0, min_loss - jnp.sqrt(n))
    x = x0
    for _ in range(max_iter):
        w_sqrt = 2 / (eta * (loss - x))
        w = w_sqrt**2
        f = jnp.sum(w) - 1
        f_prime = eta * jnp.sum(w_sqrt**3)
        x = x - f / f_prime
    return (2 / (eta * (loss - x))) ** 2, x


tsallis_opt_naive_jax = jax.jit(
    tsallis_opt_naive, static_argnames=("max_iter",)
)


@ft.wraps(tsallis_opt_naive, assigned=("__doc__",))
@ft.partial(jax.jit, static_argnames=("max_iter",))
@typed
def tsallis_opt_lax(
    loss: Float[Array, " n"],
    eta: Float[Scalar, ""],
    x0: Float[Scalar, ""] | None = None,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
    n = loss.shape[0]
    min_loss: Array = jnp.min(loss)
    if x0 is None:
        x0 = min_loss * 0.5
    x0 = lax.select(x0 < min_loss, x0, min_loss - jnp.sqrt(n))
    x = x0

    @typed
    def fori_body(
        _: Integer[Scalar, ""], x: Float[Scalar, ""]
    ) -> Float[Scalar, ""]:
        w_sqrt = 2 / (eta * (loss - x))
        w = w_sqrt**2
        f = jnp.sum(w) - 1
        f_prime = eta * jnp.sum(w_sqrt**3)
        return x - f / f_prime

    x = lax.fori_loop(0, max_iter, fori_body, x)
    w = (2 / (eta * (loss - x))) ** 2
    return w, x


# @ft.partial(jax.jit, static_argnames=("max_iter",))
@typed
def tsallis_opt(
    loss: Float[Array, " n"],
    eta: Float[Scalar, ""],
    x0: Float[Scalar, ""] | None = None,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
    """Optimization of Tsallis-inf with alpha=1/2.
    Computes arg min_(w in Delta^n) <w, loss> - 4/eta sum_i sqrt(w_i).

    Parameters
    ----------
    loss : float array of shape (n,)
        Loss vector.
    eta : float scalar
        Learning rate.
    x0 : float scalar, optional
        Initial point for the Lagrange multipliers
    max_iter : int, optional
        Maximum number of iterations, by default 8.


    Returns
    -------
    float array of shape (n,)
        Optimal probability distribution w.
    float scalar
        Optimal Lagrange multiplier.
    """

    n = loss.shape[0]
    min_loss = jnp.min(loss)
    if x0 is None:
        x0 = min_loss * 0.5
    x0 = lax.select(x0 < min_loss, x0, min_loss - jnp.sqrt(n))
    x = x0

    @typed
    def while_body(
        state: tuple[
            Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
        ],
    ) -> tuple[
        Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
    ]:
        x, f, i = state
        w_sqrt = 2 / (eta * (loss - x))
        w = w_sqrt**2
        f = jnp.sum(w) - 1
        f_prime = eta * jnp.sum(w_sqrt**3)
        return x - f / f_prime, f, i + 1

    @typed
    def while_cond(
        state: tuple[
            Float[Scalar, ""], Float[Scalar, ""], Integer[Scalar, ""]
        ],
    ) -> Bool[Scalar, ""]:
        x, f, i = state
        return (i < max_iter) & (jnp.abs(f) > 1e-8)

    x, final_w, final_i = lax.while_loop(
        while_cond, while_body, (x, jnp.array(jnp.inf), jnp.array(0))
    )
    play = (2 / (eta * (loss - x))) ** 2
    return play / jnp.sum(play), x


TsallisRead = EtaRead
TsallisWrite = NothingWrite


@typed_module
class TsallisFTRL(NewtonFTRLBase[TsallisRead, TsallisWrite]):
    @override
    def newton(
        self, total_loss: Float[Array, " n"]
    ) -> tuple[Float[Array, " n"], Float[Scalar, ""]]:
        return tsallis_opt(total_loss, self.eta, self.x0)
