import abc

from beartype import beartype
from beartype.typing import Self
from jaxtyping import Array, Float


@beartype
class OnlineLearner[R, W](abc.ABC):
    @abc.abstractmethod
    def action(self, r: R) -> tuple[Self, Float[Array, " n"]]:
        pass

    @abc.abstractmethod
    def update(self, loss: Float[Array, " n"]) -> tuple[Self, W]:
        pass
