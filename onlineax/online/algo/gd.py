from beartype.typing import override
from jaxtyping import Array, Float

from onlineax.typing import typed_module

from .ftrl import FTRLBase
from .proj import proj_sim
from .types import EtaRead, NothingWrite


@typed_module
class L2FTRL(FTRLBase[EtaRead, NothingWrite]):
    @override
    def project(
        self, total_loss: Float[Array, " n"]
    ) -> Float[Array, " n"]:
        return proj_sim(-self.eta * total_loss)
