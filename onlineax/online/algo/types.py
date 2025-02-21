import equinox as eqx
from jaxtyping import Float, Key, Scalar

from onlineax.common import C
from onlineax.typing import typed_module


@typed_module
class NothingRead(eqx.Module):
    pass


@typed_module
class KeyRead(eqx.Module):
    key: Key[Scalar, ""]


@typed_module
class EtaRead(eqx.Module):
    eta: Float[Scalar, ""] = eqx.field(converter=C, default=0.1)


@typed_module
class EtaWRead(EtaRead):
    weight: Float[Scalar, ""] = eqx.field(converter=C, default=0.1)


@typed_module
class NothingWrite(eqx.Module):
    pass
