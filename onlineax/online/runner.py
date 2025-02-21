import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Scalar

from onlineax.common import Vmapped, typed

from .base import OnlineLearner


@typed
def online_learn_step[R, W, OL: OnlineLearner](
    learner: OL,
    gradient: Float[Array, " n"],
    input: R,
) -> tuple[OL, tuple[Float[Array, " n"], Float[Scalar, ""], W]]:
    """Online learning step.

    Parameters
    ----------
    learner : OL, subclass of OnlineLearner[R, W]
        Learner.
    gradient : float[n]
        Gradient of the loss.
    input : R
        Scanned learner input.

    Returns
    -------
    final_learner : OL
        Updated learner state.
    actions : float[n]
        Action taken by the learner.
    loss : float
        Loss in step.
    output : W
        Learner output.
    """
    learner, action = learner.action(input)
    loss = jnp.dot(action, gradient)
    learner, output = learner.update(gradient)
    return learner, (action, loss, output)


@typed
def online_learn_play[R, W, OL: OnlineLearner](
    learner: OL,
    gradients: Float[Array, "t n"],
    inputs: Vmapped[R, " t"],
) -> tuple[
    OL, tuple[Float[Array, "t n"], Float[Array, " t"], Vmapped[W, " t"]]
]:
    """Online learning simulation.

    Parameters
    ----------
    learner : OL, subclass of OnlineLearner[R, W]
        Learner.
    gradients : float[t, n]
        Loss vector at each iteration.
    inputs : R[t]
        Learner input at each iteration.

    Returns
    -------
    final_learner : OL, subclass of OnlineLearner[R, W]
        Final learner state.
    actions : float[t, n]
        Action taken by the learner at each iteration.
    losses : float[t]
        Realized loss at each iteration.
    outputs : W[t]
        Learner output at each iteration.
    """

    @beartype
    def body(
        state: OL,
        consumed: tuple[Float[Array, " n"], Vmapped[R, " t"]],
    ) -> tuple[
        OL,
        tuple[Float[Array, " n"], Float[Scalar, ""], Vmapped[W, " t"]],
    ]:
        gradient, learner_read = consumed
        state, (action, loss, output) = online_learn_step(
            state, gradient, learner_read
        )
        return state, (action, loss, output)

    return lax.scan(body, learner, (gradients, inputs))


@typed
def online_matrix_game_step[
    R1,
    R2,
    W1,
    W2,
    OL1: OnlineLearner,
    OL2: OnlineLearner,
](
    learner: tuple[OL1, OL2],
    game: Float[Array, "n m"],
    input: tuple[R1, R2],
) -> tuple[
    tuple[OL1, OL2],
    tuple[
        tuple[Float[Array, " n"], Float[Array, " m"]],
        Float[Scalar, ""],
        tuple[W1, W2],
    ],
]:
    """Online learning in matrix game step.
    It is assumed that the 1st player is the minimizer.

    Parameters
    ----------
    learner : (OL1, OL2), where OL1 is a subclass of OnlineLearner[R1, W1] \
                and OL2 is a subclass of OnlineLearner[R2, W2]
        Learner for each player.
    game : float[n, m]
        Game matrix.
    input : (R1, R2)
        Scanned learner input for each player at each iteration.

    Returns
    -------
    final_learner : (OL1, OL2)
        Updated learner state for each player.
    action : (float[n], float[m])
        Actions taken by each player.
    loss : float
        Value in step.
    output : (W1, W2)
        Learner output for each player.
    """
    learner1, learner2 = learner
    learner1, action1 = learner1.action(input[0])
    learner2, action2 = learner2.action(input[1])
    loss1 = game @ action2
    loss2 = 1 - game.T @ action1
    learner1, output1 = learner1.update(loss1)
    learner2, output2 = learner2.update(loss2)
    value = jnp.dot(action1, loss1)
    return (
        (learner1, learner2),
        ((action1, action2), value, (output1, output2)),
    )


@typed
def online_matrix_game_play[
    R1,
    R2,
    W1,
    W2,
    OL1: OnlineLearner,
    OL2: OnlineLearner,
](
    learner: tuple[OL1, OL2],
    game: Float[Array, "n m"],
    inputs: tuple[Vmapped[R1, " t"], Vmapped[R2, " t"]],
    length: int | None = None,
) -> tuple[
    tuple[OL1, OL2],
    tuple[
        tuple[Float[Array, "t n"], Float[Array, "t m"]],
        Float[Array, " t"],
        tuple[Vmapped[W1, " t"], Vmapped[W2, " t"]],
    ],
]:
    """Online learning in matrix game simulation.
    It is assumed that the 1st player is the minimizer.

    Parameters
    ----------
    learner : (OL1, OL2), where OL1 is a subclass of OnlineLearner[R1, W1] \
                and OL2 is a subclass of OnlineLearner[R2, W2]
        Learner for each player.
    game : float[n, m]
        Game matrix.
    inputs : (R1[t], R2[t])
        Scanned learner input for each player.

    Returns
    -------
    learner : (OL1, OL2)
        Updated learner state for each player.
    actions : (float[t, n], float[t, m])
        Actions taken by each player at each iteration.
    losses : float[t]
        Value at each iteration.
    outputs : (W1[t], W2[t])
        Learner output for each player at each iteration.
    """

    @beartype
    def body(
        state: tuple[OL1, OL2],
        consumed: tuple[Vmapped[R1, " t"], Vmapped[R2, " t"]],
    ) -> tuple[
        tuple[OL1, OL2],
        tuple[
            tuple[Float[Array, " n"], Float[Array, " m"]],
            Float[Scalar, ""],
            tuple[W1, W2],
        ],
    ]:
        (input1, input2) = consumed
        state, (actions, value, outputs) = online_matrix_game_step(
            state, game, (input1, input2)
        )
        return state, (actions, value, outputs)

    return lax.scan(body, learner, inputs, length=length)
