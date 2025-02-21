from typing import Annotated

import jax.numpy as jnp

from onlineax.typing import (
    jaxtype_all_methods as jaxtype_all_methods,
    typed as typed,
    typed_module as typed_module,
)

Vmapped = Annotated
C = jnp.array
