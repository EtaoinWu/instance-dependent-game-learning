import equinox as eqx
from beartype import beartype
from jaxtyping import jaxtyped

from . import _chore as _chore

typed = jaxtyped(typechecker=beartype)


def jaxtype_all_methods[T: type](module: T) -> T:
    """Decorator to apply jaxtyped to all methods of a class or an `eqx.Module`."""
    for name, method in module.__dict__.items():
        if hasattr(method, "__beartype_wrapper"):
            setattr(module, name, jaxtyped(method))
        elif type(method) is eqx._module._wrap_method:
            setattr(module, name, jaxtyped(method.method))
    return module  # type: ignore


def typed_module[T: type](module: T) -> T:
    """Decorator to apply jaxtyped and beartype to a class or an `eqx.Module`."""
    return jaxtype_all_methods(beartype(typed(module)))
