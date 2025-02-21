import jax
import jax.tree as jt

from .utils import tree_stack, tree_unstack


class MapHelper:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *xs):
        return self.fn(*xs)

    @property
    def j(self):
        """Just-in-time compilation."""
        return self.__class__(jax.jit(self.fn))

    def s(self, serial_dim: int):
        """Serial map."""
        return self.__class__(serial_map(serial_dim, self.fn))

    def v(self, vectorize_dim: int):
        """Vectorize map."""
        return self.__class__(vectorize_map(vectorize_dim, self.fn))

    def p(self, parallelism_dim: int):
        """Parallel map."""
        return self.__class__(parallel_map(parallelism_dim, self.fn))

    def i(self, iterate_dim: int):
        """Iterative map."""
        return self.__class__(iterative_map(iterate_dim, self.fn))


def map_helper(fn):
    def inner(*xs):
        leaves = jt.leaves(xs)
        assert all(x.shape[0] == 1 for x in leaves)
        result = fn(*jt.map(lambda x: x[0], xs))
        return jt.map(lambda x: x.reshape(1, *x.shape), result)

    return MapHelper(inner)


def undim(xs, d):
    if d == -1:
        return xs.reshape(xs.shape[0], 1, *xs.shape[1:])
    assert xs.shape[0] % d == 0
    return xs.reshape(d, xs.shape[0] // d, *xs.shape[1:])


def undim_tree(xs, d):
    return jt.map(lambda x: undim(x, d), xs)


def redim(xs):
    return xs.reshape(-1, *xs.shape[2:])


def redim_tree(xs):
    return jt.map(redim, xs)


def serial_map(serial_dim: int, fn):
    def inner(none: None, xs):
        return None, fn(*xs)

    def outer(*xs):
        xs = undim_tree(xs, serial_dim)
        _, result = jax.lax.scan(inner, None, xs)
        return redim_tree(result)

    return outer


def vectorize_map(vectorize_dim: int, fn):
    inner = jax.vmap(fn)

    def outer(*xs):
        xs = undim_tree(xs, vectorize_dim)
        return redim_tree(inner(*xs))

    return outer


def parallel_map(parallelism_dim: int, fn):
    inner = jax.pmap(fn)

    def outer(*xs):
        xs = undim_tree(xs, parallelism_dim)
        return redim_tree(inner(*xs))

    return outer


def iterative_map(iterative_dim: int, fn):
    def outer(*xs):
        xs = undim_tree(xs, iterative_dim)
        xsl = [fn(*x) for x in tree_unstack(xs)]
        xs = tree_stack(xsl)
        return redim_tree(xs)

    return outer
