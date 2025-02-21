import abc

from beartype import beartype
from beartype.typing import Self
from jaxtyping import Float, Integer, Scalar


@beartype
class MABLearner[R, W](abc.ABC):
    @abc.abstractmethod
    def action(self, r: R) -> tuple[Self, Integer[Scalar, ""]]:
        pass

    @abc.abstractmethod
    def update(self, loss: Float[Scalar, ""]) -> tuple[Self, W]:
        pass
