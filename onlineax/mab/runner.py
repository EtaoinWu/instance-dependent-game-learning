import jax.numpy as jnp
from beartype import beartype
from jax import (
    lax,
    random as jr,
)
from jaxtyping import Array, Float, Integer, Key, Scalar

from onlineax.common import Vmapped, typed

from .base import MABLearner


@typed
def mab_bernoulli_step[R, W, ML: MABLearner](
    learner: ML,
    loss_avg: Float[Array, " n"],
    input: R,
    mab_draw: Float[Scalar, ""],
) -> tuple[ML, tuple[Integer[Scalar, ""], Float[Scalar, ""], W]]:
    """Multi-armed bandit step.

    Parameters
    ----------
    learner : ML, subclass of MABLearner[R, W]
        Learner.
    input : R
        Scanned learner state.
    mab_draw : float
        Random draw to generate the loss.

    Returns
    -------
    final_learner : ML
        Updated MAB learner.
    action : integer
        Action taken by the learner.
    loss : float
        Loss in step.
    output : W
        Learner output.
    """

    learner, action = learner.action(input)
    loss = (loss_avg[action] > mab_draw).astype(jnp.float32)
    learner, output = learner.update(loss)
    return learner, (action, loss, output)


def bernoulli_draws(key: Key[Scalar, ""], t: int) -> Float[Array, " t"]:
    return jr.uniform(key, (t,))


@typed
def mab_bernoulli_play[R, W, ML: MABLearner](
    learner: ML,
    loss_avg: Float[Array, " n"],
    input: Vmapped[R, " t"],
    mab_draws: Float[Array, " t"],
) -> tuple[
    ML,
    tuple[Integer[Array, " t"], Float[Array, " t"], Vmapped[W, " t"]],
]:
    """Multi-armed bandit play (0-1 bernoulli loss).

    Parameters
    ----------
    learner : ML, subclass of MABLearner[R, W]
        Learner.
    loss_avg : float[n]
        Average loss of each arm.
    input : R[t]
        Scanned learner state.
    mab_draws : float[t]
        Random draws to generate the losses.

    Returns
    -------
    final_learner : ML
        Final learner.
    actions : integer[t]
        Action taken by the learner at each iteration.
    losses : float[t]
        Loss in each iteration.
    outputs : W[t]
        Learner output.
    """

    @beartype
    def body(
        state: ML, consumed: tuple[Float[Scalar, ""], R]
    ) -> tuple[ML, tuple[Integer[Scalar, ""], Float[Scalar, ""], W]]:
        mab_draw, input = consumed
        state, (action, loss, output) = mab_bernoulli_step(
            state, loss_avg, input, mab_draw
        )
        return state, (action, loss, output)

    return lax.scan(body, learner, (mab_draws, input))


@typed
def mab_gaussian_step[R, W, ML: MABLearner](
    learner: ML,
    loss_avg: Float[Array, " n"],
    input: R,
    mab_noise: Float[Scalar, ""],
) -> tuple[ML, tuple[Integer[Scalar, ""], Float[Scalar, ""], W]]:
    """Multi-armed bandit step.

    Parameters
    ----------
    learner : ML, subclass of MABLearner[R, W]
        Learner.
    input : R
        Scanned learner state.
    mab_noise : float
        Random noise to generate the loss.

    Returns
    -------
    final_learner : ML
        Updated MAB learner.
    action : integer
        Action taken by the learner.
    loss : float
        Loss in step.
    output : W
        Learner output.
    """

    learner, action = learner.action(input)
    loss = loss_avg[action] + mab_noise
    learner, output = learner.update(loss)
    return learner, (action, loss, output)


@typed
def gaussian_noises(key: Key[Scalar, ""], t: int) -> Float[Array, " t"]:
    return jr.normal(key, (t,))


@typed
def mab_gaussian_play[R, W, ML: MABLearner](
    learner: ML,
    loss_avg: Float[Array, " n"],
    input: Vmapped[R, " t"],
    mab_noises: Float[Array, " t"],
) -> tuple[
    ML,
    tuple[Integer[Array, " t"], Float[Array, " t"], Vmapped[W, " t"]],
]:
    """Multi-armed bandit play (0-1 bernoulli loss).

    Parameters
    ----------
    learner : ML, subclass of MABLearner[R, W]
        Learner.
    loss_avg : float[n]
        Average loss of each arm.
    input : R[t]
        Scanned learner state.
    mab_noises : float[t]
        Random noises to generate the losses.

    Returns
    -------
    final_learner : ML
        Final learner.
    actions : integer[t]
        Action taken by the learner at each iteration.
    losses : float[t]
        Loss in each iteration.
    outputs : W[t]
        Learner output.
    """

    @beartype
    def body(
        state: ML, consumed: tuple[Float[Scalar, ""], R]
    ) -> tuple[ML, tuple[Integer[Scalar, ""], Float[Scalar, ""], W]]:
        mab_noise, input = consumed
        state, (action, loss, output) = mab_bernoulli_step(
            state, loss_avg, input, mab_noise
        )
        return state, (action, loss, output)

    return lax.scan(body, learner, (mab_noises, input))


@typed
def mab_bernoulli_matrix_game_step[
    R1,
    R2,
    W1,
    W2,
    ML1: MABLearner,
    ML2: MABLearner,
](
    learners: tuple[ML1, ML2],
    loss_matrix: Float[Array, " n m"],
    input: tuple[R1, R2],
    mab_draw: Float[Scalar, ""],
) -> tuple[
    tuple[ML1, ML2],
    tuple[
        tuple[Integer[Scalar, ""], Integer[Scalar, ""]],
        Float[Scalar, ""],
        tuple[W1, W2],
    ],
]:
    """Multi-armed bandit matrix game step.

    Parameters
    ----------
    learner : (ML1, ML2), subclasses of MABLearner[R1, W1], and \
                MABLearner[R2, W2] respectively
        Learners.
    loss_matrix : float[n, m]
        Loss matrix.
    learner_read : (R1, R2)
        Scanned learner inputs.
    mab_draw : float
        Random draw to generate the loss.

    Returns
    -------
    final_learner : (ML1, ML2)
        Updated learners.
    action : (integer, integer)
        Actions taken by the learners.
    loss : float
        Loss in step.
    output : (W1, W2)
        Learner outputs.
    """

    learner1, learner2 = learners
    input1, input2 = input
    learner1, action1 = learner1.action(input1)
    learner2, action2 = learner2.action(input2)
    expected_loss = loss_matrix[action1, action2]
    loss = (expected_loss > mab_draw).astype(jnp.float32)
    learner1, output1 = learner1.update(loss)
    learner2, output2 = learner2.update(1 - loss)
    return (learner1, learner2), (
        (action1, action2),
        loss,
        (output1, output2),
    )


@typed
def mab_bernoulli_matrix_game_play[
    R1,
    R2,
    W1,
    W2,
    ML1: MABLearner,
    ML2: MABLearner,
](
    learners: tuple[ML1, ML2],
    loss_matrix: Float[Array, " n m"],
    inputs: tuple[Vmapped[R1, " t"], Vmapped[R2, " t"]],
    mab_draws: Float[Array, " t"],
) -> tuple[
    tuple[ML1, ML2],
    tuple[
        tuple[Integer[Array, " t"], Integer[Array, " t"]],
        Float[Array, " t"],
        tuple[Vmapped[W1, " t"], Vmapped[W2, " t"]],
    ],
]:
    """Multi-armed bandit matrix game play (0-1 bernoulli loss).

    Parameters
    ----------
    learner : (ML1, ML2), subclasses of MABLearner[R1, W1], and \
                MABLearner[R2, W2] respectively
        Learners.
    loss_matrix : float[n, m]
        Loss matrix.
    learner_reads : (R1[t], R2[t])
        Scanned learner inputs.
    mab_draws : float[t]
        Random draws to generate the losses.

    Returns
    -------
    final_learner : (ML1, ML2)
        Final learners.
    actions : (integer[t], integer[t])
        Actions taken by the learners at each iteration.
    losses : float[t]
        Realized loss in each iteration.
    outputs : (W1[t], W2[t])
        Learner outputs.
    """

    @beartype
    def body(
        learners: tuple[ML1, ML2],
        consumed: tuple[Float[Scalar, ""], tuple[R1, R2]],
    ) -> tuple[
        tuple[ML1, ML2],
        tuple[
            tuple[Integer[Scalar, ""], Integer[Scalar, ""]],
            Float[Scalar, ""],
            tuple[W1, W2],
        ],
    ]:
        mab_draw, input = consumed
        (
            learners,
            (actions, loss, outputs),
        ) = mab_bernoulli_matrix_game_step(
            learners, loss_matrix, input, mab_draw
        )
        return learners, (actions, loss, outputs)

    return lax.scan(body, learners, (mab_draws, inputs))
