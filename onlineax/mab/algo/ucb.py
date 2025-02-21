import equinox as eqx
import jax
from beartype.typing import Callable, Self, override
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, Scalar

from onlineax.online.algo.types import NothingRead, NothingWrite
from onlineax.typing import typed_module

from ..adaptor import MABWrapWrite
from ..base import MABLearner

# Confidence bound function: (mean, times pulled) -> confidence radius
type ConfidenceBoundFn = Callable[
    [Float[Scalar, ""], Float[Scalar, ""]], Float[Scalar, ""]
]
type ConfidenceBoundFnArray = Callable[
    [Float[Scalar, ""], Float[Array, " n"]], Float[Array, " n"]
]


def ucb1_confidence_bound(
    alpha: Float[Scalar, ""],
) -> ConfidenceBoundFn:
    def ucb1_confidence_bound_fn(
        current_round: Float[Scalar, ""],
        times_pulled: Float[Scalar, ""],
    ) -> Float[Scalar, ""]:
        return jnp.sqrt(
            alpha * jnp.log(current_round) / (times_pulled + 1e-6)
        )

    return ucb1_confidence_bound_fn


def _vmap_confidence_bound_func(
    f: ConfidenceBoundFn,
) -> ConfidenceBoundFnArray:
    return jax.vmap(f, in_axes=(None, 0), out_axes=0)


@typed_module
class UCB1(
    eqx.Module,
    MABLearner[NothingRead, NothingWrite | MABWrapWrite[NothingWrite]],
):
    n: int = eqx.field(static=True)
    bound_func: ConfidenceBoundFn = eqx.field(static=True)
    total_loss: Float[Array, " n"]
    times_pulled: Float[Array, " n"]
    last_played: Integer[Scalar, ""]
    round_number: Integer[Scalar, ""] = eqx.field(
        default=1, converter=jnp.array
    )
    output_internal: bool = eqx.field(default=True, static=True)

    @override
    def action(
        self, r: NothingRead
    ) -> tuple[Self, Integer[Scalar, ""]]:
        ucb = _vmap_confidence_bound_func(self.bound_func)(
            self.round_number, self.times_pulled
        )
        play = jnp.argmin(
            self.total_loss / (self.times_pulled + 1e-6) - ucb
        )
        return self.__class__(
            n=self.n,
            bound_func=self.bound_func,
            total_loss=self.total_loss,
            times_pulled=self.times_pulled.at[play].add(1.0),
            last_played=play,
            round_number=self.round_number + 1,
        ), play

    @override
    def update(
        self, loss: Float[Scalar, ""]
    ) -> tuple[Self, NothingWrite | MABWrapWrite[NothingWrite]]:
        return self.__class__(
            n=self.n,
            bound_func=self.bound_func,
            total_loss=self.total_loss.at[self.last_played].add(loss),
            times_pulled=self.times_pulled,
            last_played=self.last_played,
            round_number=self.round_number,
        ), (
            NothingWrite(),
            (jnp.arange(self.n) == self.last_played).astype(float),
        ) if self.output_internal else NothingWrite()
