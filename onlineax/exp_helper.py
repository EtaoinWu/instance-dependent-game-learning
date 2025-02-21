import equinox as eqx
import jax.tree as jt
import numpy as np


def choose_subset(x, d=100, dlin=None, dsqr=None, dlog=None):
    dlin = dlin or d
    dsqr = dsqr or d
    dlog = dlog or d
    x = np.asarray(x)
    n = x.shape[0]
    first_n = np.arange(min(n, d))
    lin_n = np.rint(np.linspace(0, n - 1, dlin)).astype(np.int32)
    sqr_n = np.rint(np.linspace(0, np.sqrt(n - 1), dsqr) ** 2).astype(
        np.int32
    )
    log_n = np.rint(np.logspace(0, np.log10(n), dlog) - 1).astype(
        np.int32
    )
    joined = np.concatenate([first_n, lin_n, log_n, sqr_n])
    return x[np.unique(joined)]


def expspace(n, Δmin, Δmax):
    return np.exp(np.linspace(np.log(Δmin), np.log(Δmax), n))


def filter_map(fn, xt):
    return jt.map(lambda x: fn(x) if eqx.is_inexact_array(x) else x, xt)


def filter_block(xt):
    return filter_map(lambda x: x.block_until_ready(), xt)
