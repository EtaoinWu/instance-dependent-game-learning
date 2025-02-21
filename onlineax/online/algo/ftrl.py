import abc

import equinox as eqx
from beartype.typing import Self, cast, override
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar

from onlineax.typing import typed_module

from ..base import OnlineLearner
from .types import EtaRead, EtaWRead, NothingWrite


@typed_module
class FTRLBase[R: EtaRead, W: NothingWrite](
    eqx.Module, OnlineLearner[R, W]
):
    n: int = eqx.field(static=True)
    eta: Float[Scalar, ""]
    total_loss: Float[Array, " n"]

    @abc.abstractmethod
    def project(
        self, total_loss: Float[Array, " n"]
    ) -> Float[Array, " n"]: ...

    @override
    @override
    def action(self, r: R) -> tuple[Self, Float[Array, " n"]]:
        new_self = self.__class__(
            n=self.n, eta=r.eta, total_loss=self.total_loss
        )
        return (
            new_self,
            new_self.project(self.total_loss),
        )

    @override
    def update(self, loss: Float[Array, " n"]) -> tuple[Self, W]:
        return self.__class__(
            n=self.n,
            eta=self.eta,
            total_loss=self.total_loss + loss,
        ), cast(W, NothingWrite())


@typed_module
class WeightedFTRLBase[R: EtaWRead, W: NothingWrite](FTRLBase[R, W]):
    n: int = eqx.field(static=True)
    eta: Float[Scalar, ""]
    weight: Float[Scalar, ""]
    total_loss: Float[Array, " n"]

    @override
    def action(self, r: R) -> tuple[Self, Float[Array, " n"]]:
        new_self = self.__class__(
            n=self.n,
            eta=r.eta,
            weight=r.weight,
            total_loss=self.total_loss,
        )
        return (
            new_self,
            new_self.project(self.total_loss),
        )

    @override
    def update(self, loss: Float[Array, " n"]) -> tuple[Self, W]:
        return self.__class__(
            n=self.n,
            eta=self.eta,
            weight=self.weight,
            total_loss=self.total_loss + loss * self.weight,
        ), cast(W, NothingWrite())


@typed_module
class NewtonFTRLBase[R: EtaRead, W: NothingWrite](FTRLBase[R, W]):
    x0: Float[Scalar, ""] = eqx.field(converter=jnp.array, default=0.1)

    @abc.abstractmethod
    def newton(
        self, total_loss: Float[Array, " n"]
    ) -> tuple[Float[Array, " n"], Float[Scalar, ""]]: ...

    @override
    def project(
        self, total_loss: Float[Array, " n"]
    ) -> Float[Array, " n"]:
        return self.newton(total_loss)[0]

    @override
    def action(self, r: R) -> tuple[Self, Float[Array, " n"]]:
        new_self = self.__class__(
            n=self.n, eta=r.eta, total_loss=self.total_loss, x0=self.x0
        )
        play, x1 = new_self.newton(self.total_loss)
        return self.__class__(
            n=self.n, total_loss=self.total_loss, eta=r.eta, x0=x1
        ), play

    @override
    def update(self, loss: Float[Array, " n"]) -> tuple[Self, W]:
        return self.__class__(
            n=self.n,
            total_loss=self.total_loss + loss,
            eta=self.eta,
            x0=self.x0,
        ), cast(W, NothingWrite())


@typed_module
class WeightedNewtonFTRLBase[R: EtaWRead, W: NothingWrite](
    NewtonFTRLBase[R, W], WeightedFTRLBase[R, W]
):
    @override
    def action(self, r: R) -> tuple[Self, Float[Array, " n"]]:
        new_self = self.__class__(
            n=self.n,
            eta=r.eta,
            total_loss=self.total_loss,
            weight=r.weight,
            x0=self.x0,
        )
        play, x1 = new_self.newton(self.total_loss)
        return self.__class__(
            n=self.n,
            total_loss=self.total_loss,
            eta=r.eta,
            weight=r.weight,
            x0=x1,
        ), play

    @override
    def update(self, loss: Float[Array, " n"]) -> tuple[Self, W]:
        return self.__class__(
            n=self.n,
            total_loss=self.total_loss + self.weight * loss,
            eta=self.eta,
            weight=self.weight,
            x0=self.x0,
        ), cast(W, NothingWrite())
