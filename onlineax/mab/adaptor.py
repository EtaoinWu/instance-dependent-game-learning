import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Self, TypeVar, override
from jax import (
    random as jr,
)
from jaxtyping import Array, Float, Integer, Key, Scalar

import onlineax.sampling as sampling
from onlineax.online.base import OnlineLearner
from onlineax.typing import typed_module

from .base import MABLearner
from .estimator import Estimator

MAB_R = TypeVar("MAB_R")
MAB_W = TypeVar("MAB_W")
MABWrapRead = tuple[MAB_R, Float[Scalar, ""]]
MABWrapWrite = tuple[MAB_W, Float[Array, " n"] | None]


@typed_module
class MABFromOnline[MAB_R, MAB_W](
    eqx.Module, MABLearner[MABWrapRead[MAB_R], MABWrapWrite[MAB_W]]
):
    online_learner: OnlineLearner[MAB_R, MAB_W]
    estimator: Estimator = eqx.field(static=True)
    weights: Float[Array, " n"]
    track_weights: bool = eqx.field(default=True, static=True)
    choice: Integer[Scalar, ""] = eqx.field(
        default_factory=lambda: jnp.array(0)
    )  # type: ignore

    @override
    def action(
        self,
        r: MABWrapRead,
    ) -> tuple[Self, Integer[Scalar, ""]]:
        inner_r, draw = r
        online_learner, p = self.online_learner.action(inner_r)
        choice = sampling.choice(p, draw)
        return (
            self.__class__(
                online_learner=online_learner,
                estimator=self.estimator,
                weights=p,
                track_weights=self.track_weights,
                choice=choice,
            ),
            choice,
        )

    @override
    def update(
        self,
        loss: Float[Scalar, ""],
    ) -> tuple[Self, MABWrapWrite]:
        estimated_loss = self.estimator(self.choice, loss, self.weights)
        online_learner, w = self.online_learner.update(estimated_loss)
        return (
            self.__class__(
                online_learner=online_learner,
                estimator=self.estimator,
                weights=self.weights,
                track_weights=self.track_weights,
                choice=self.choice,
            ),
            (w, self.weights if self.track_weights else None),
        )

    @staticmethod
    def wrap_read(
        r: MAB_R,
        key: Key[Scalar, ""],
        t: int,
    ) -> tuple[MAB_R, Float[Array, "..."]]:
        return r, jr.uniform(key, (t,))
