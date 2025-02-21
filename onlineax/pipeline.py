import abc
import functools as ft

import equinox as eqx
import jax
from beartype.typing import Any, Callable, Literal, Self, cast, override
from jax import (
    numpy as jnp,
    random as jr,
    tree as jt,
)
from jaxtyping import Array, Float, Integer, Key, Scalar

from onlineax.common import Vmapped, typed
from onlineax.mab.algo.ucb import (
    UCB1,
    ConfidenceBoundFn,
    ucb1_confidence_bound,
)
from onlineax.typing import typed_module
from onlineax.utils import duality_gap, treemap_concat

from .mab.adaptor import MABFromOnline
from .mab.base import MABLearner
from .mab.estimator import (
    Estimator,
    importance_weighed_estimator,
    variance_reduced_estimator,
)
from .mab.runner import (
    bernoulli_draws,
    mab_bernoulli_matrix_game_play,
)
from .online.algo.ftrl import FTRLBase
from .online.algo.types import (
    EtaRead,
    NothingRead,
    NothingWrite,
)
from .online.base import OnlineLearner
from .online.runner import online_matrix_game_play

type ConvertToLearningRate = (
    int
    | float
    | Float[Scalar, ""]
    | Float[Array, " t"]
    | Callable[[Integer[Scalar, ""]], Float[Scalar, ""]]
)


@typed_module
class LearningRateBase(eqx.Module):
    @abc.abstractmethod
    def __call__(self, t: int) -> Float[Array, " t"]:
        pass


@typed_module
class ConstantLearningRate(LearningRateBase):
    value: Float[Scalar, ""] = eqx.field(converter=jnp.array)

    @override
    def __call__(self, t: int) -> Float[Array, " t"]:
        return jnp.repeat(self.value, t)


@typed_module
class CustomArrayLearningRate(LearningRateBase):
    value: Float[Array, " t"]

    @override
    def __call__(self, t: int) -> Float[Array, " t"]:
        assert self.value.shape == (t,), (
            f"Invalid shape {self.value.shape} for learning rate."
        )
        return self.value


@typed_module
class CustomFunctionLearningRate(LearningRateBase):
    value: Callable[[Integer[Scalar, ""]], Float[Scalar, ""]] = (
        eqx.field(static=True)
    )

    @override
    def __call__(self, t: int) -> Float[Array, " t"]:
        return jax.vmap(self.value)(jnp.arange(1, t + 1))


@typed
def _convert_learning_rate(
    lr: ConvertToLearningRate,
) -> LearningRateBase:
    if isinstance(lr, Array):
        if lr.ndim != 1:
            raise ValueError("Invalid shape for learning rate.")
        return CustomArrayLearningRate(value=lr)
    elif callable(lr):
        return CustomFunctionLearningRate(value=lr)
    elif isinstance(lr, int | float):
        return ConstantLearningRate(value=lr)


@typed_module
class OnlineLearnerConfigBase[R, W](eqx.Module):
    @abc.abstractmethod
    def _init_learner(
        self, n: int, t: int
    ) -> tuple[OnlineLearner[R, W], Vmapped[Any, " t"]]:
        pass


@typed_module
class FTRLConfig[T_FTRL: FTRLBase, R: EtaRead, W: NothingWrite](
    OnlineLearnerConfigBase[R, W]
):
    learner: type[T_FTRL] = eqx.field(static=True)
    eta: LearningRateBase = eqx.field(converter=_convert_learning_rate)
    read: type[R] = eqx.field(default=EtaRead, static=True)

    @override
    def _init_learner(
        self, n: int, t: int
    ) -> tuple[T_FTRL, Vmapped[Any, " t"]]:
        eta = self.eta(t)
        read = jax.vmap(self.read)(eta)
        return self.learner(
            n=n, eta=eta[0], total_loss=jnp.zeros(n)
        ), read


OnlineLearnerConfig = OnlineLearnerConfigBase


type OnlineGameLearnerConfig = (
    OnlineLearnerConfig
    | tuple[OnlineLearnerConfig, OnlineLearnerConfig]
)


@typed
def canonicalize_online_game_learner_config(
    config: OnlineGameLearnerConfig,
) -> tuple[OnlineLearnerConfig, OnlineLearnerConfig]:
    if isinstance(config, OnlineLearnerConfig):
        return (config, config)
    else:
        return config


@typed_module
class MABLearnerConfigBase[R, W](eqx.Module):
    @abc.abstractmethod
    def _init_learner(
        self, n: int, t: int
    ) -> tuple[MABLearner[R, W], Vmapped[Any, " t"]]:
        pass


@typed_module
class MABAdaptorConfig[MAB_R, MAB_W](
    MABLearnerConfigBase[MAB_R, MAB_W]
):
    inner: OnlineLearnerConfig
    estimator: (
        Literal["importance_weighted", "variance_reduced"] | Estimator
    ) = eqx.field(static=True)
    key: Key[Scalar, ""] | None = eqx.field(default=None)
    track_weights: bool = eqx.field(default=True, static=True)

    @property
    def _estimator(self) -> Estimator:
        if self.estimator == "importance_weighted":
            return importance_weighed_estimator
        elif self.estimator == "variance_reduced":
            return variance_reduced_estimator
        else:
            return self.estimator

    @override
    def _init_learner(
        self, n: int, t: int
    ) -> tuple[MABLearner[MAB_R, MAB_W], Vmapped[Any, " t"]]:
        if self.key is None:
            raise ValueError("MABAdaptorConfig: key is not provided.")
        inner_learner, inner_input = self.inner._init_learner(n, t)
        return (
            cast(
                MABLearner[MAB_R, MAB_W],
                MABFromOnline(
                    online_learner=inner_learner,
                    estimator=self._estimator,
                    weights=jnp.zeros(n),
                    track_weights=self.track_weights,
                ),
            ),
            MABFromOnline.wrap_read(inner_input, self.key, t),
        )

    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        if self.key is not None or key is None:
            return self
        return self.__class__(self.inner, self.estimator, key)


@typed_module
class MABUCBConfig(MABLearnerConfigBase):
    alpha_or_confidence_bound: Float[Scalar, ""] | ConfidenceBoundFn

    @override
    def _init_learner(
        self, n: int, t: int
    ) -> tuple[UCB1, Vmapped[Any, " t"]]:
        if callable(self.alpha_or_confidence_bound):
            bound_func = self.alpha_or_confidence_bound
        else:
            bound_func = ucb1_confidence_bound(
                self.alpha_or_confidence_bound
            )
        return UCB1(
            n=n,
            bound_func=bound_func,
            total_loss=jnp.zeros(n),
            times_pulled=jnp.zeros(n),
            last_played=jnp.array(0),
        ), NothingRead()

    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        return self


MABLearnerConfig = MABAdaptorConfig | MABUCBConfig
type MABCompatibleLearnerConfig = OnlineLearnerConfig | MABLearnerConfig


@typed
def canonicalize_mab_learner_config(
    config: MABCompatibleLearnerConfig,
    key: Key[Scalar, ""] | None = None,
) -> MABLearnerConfig:
    if isinstance(config, OnlineLearnerConfig):
        config = MABAdaptorConfig(
            inner=config, estimator="importance_weighted", key=key
        )
    return config._feed_key(key)


MABGameLearnerConfig = (
    MABCompatibleLearnerConfig
    | tuple[MABCompatibleLearnerConfig, MABCompatibleLearnerConfig]
)


@typed
def canonicalize_mab_game_learner_config(
    config: MABGameLearnerConfig,
    key: Key[Scalar, ""] | None = None,
) -> tuple[MABLearnerConfig, MABLearnerConfig]:
    if not isinstance(config, tuple):
        config = (config, config)
    keys = jr.split(key) if key is not None else (None, None)
    config = (
        canonicalize_mab_learner_config(config[0], keys[0]),
        canonicalize_mab_learner_config(config[1], keys[1]),
    )
    return config


@typed_module
class GameSetupBase(eqx.Module):
    @abc.abstractmethod
    def _game(self) -> Float[Array, "n m"]:
        pass

    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        return self


@typed_module
class FixedGameSetup(GameSetupBase):
    game: Float[Array, "n m"]

    @override
    def _game(self) -> Float[Array, "n m"]:
        return self.game


@typed_module
class RandomGameSetup(GameSetupBase):
    shape: tuple[int, int]
    distribution: Literal["uniform"] = eqx.field(static=True)
    key: Key[Scalar, ""] | None = eqx.field(default=None)

    @override
    def _game(self) -> Float[Array, "n m"]:
        if self.key is None:
            raise ValueError("RandomGameSetup: key is not provided.")
        return jr.uniform(self.key, self.shape)

    @override
    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        if self.key is not None or key is None:
            return self
        return self.__class__(self.shape, self.distribution, key)


type GameSetup = FixedGameSetup | RandomGameSetup


@typed_module
class MABGameSetupBase(eqx.Module):
    @abc.abstractmethod
    def _game(self) -> Float[Array, "n m"]:
        pass

    @abc.abstractmethod
    def _draws(self, t: int) -> Vmapped[Any, " t"]:
        pass

    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        return self


@typed_module
class MABBernoulliGameSetup(MABGameSetupBase):
    online_setup: GameSetup
    draw_key: Key[Scalar, ""] | None = eqx.field(default=None)

    @override
    def _game(self) -> Float[Array, "n m"]:
        return self.online_setup._game()

    @override
    def _draws(self, t: int) -> Vmapped[Any, " t"]:
        if self.draw_key is None:
            raise ValueError(
                "MABBernoulliGameSetup: draw_key is not provided."
            )

        return bernoulli_draws(self.draw_key, t)

    @override
    def _feed_key(self, key: Key[Scalar, ""] | None) -> Self:
        if key is None:
            return self
        inner_key, self_key = jr.split(key)
        return self.__class__(
            self.online_setup._feed_key(inner_key),
            self.draw_key if self.draw_key is not None else self_key,
        )


type MABGameSetup = MABBernoulliGameSetup | GameSetup


def canonicalize_game_setup(
    setup: MABGameSetup,
) -> MABBernoulliGameSetup:
    if isinstance(setup, MABBernoulliGameSetup):
        return setup
    return MABBernoulliGameSetup(online_setup=setup)


@typed_module
class PipelineOutputShared(eqx.Module):
    learner: eqx.AbstractVar[tuple[eqx.Module, eqx.Module]]
    actions: eqx.AbstractVar[tuple[eqx.Module, eqx.Module]]
    game: Float[Array, "n m"]
    max_iter: int = eqx.field(static=True)

    losses: Float[Array, " t"]
    outputs: tuple[Vmapped[Any, " t"], Vmapped[Any, " t"]]

    @property
    @abc.abstractmethod
    def actions_vec(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        pass

    @property
    def avg_actions(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        return jt.map(
            lambda x: jnp.cumsum(x, axis=0)
            / jnp.arange(1, self.max_iter + 1)[:, None],
            self.actions_vec,
        )

    @ft.cached_property
    def iter_duality_gaps(self) -> Float[Array, " t"]:
        return jax.vmap(duality_gap, in_axes=(None, 0, 0))(
            self.game, self.actions_vec[0], self.actions_vec[1]
        )

    @ft.cached_property
    def avg_duality_gaps(self) -> Float[Array, " t"]:
        return jax.vmap(duality_gap, in_axes=(None, 0, 0))(
            self.game, self.avg_actions[0], self.avg_actions[1]
        )


@typed_module
class OnlinePipelineOutput(PipelineOutputShared):
    learner: tuple[OnlineLearner, OnlineLearner]
    actions: tuple[Float[Array, "t n"], Float[Array, "t m"]]

    @property
    @override
    def actions_vec(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        return self.actions


@typed_module
class MABPipelineOutput(PipelineOutputShared):
    learner: tuple[MABLearner, MABLearner]
    actions: tuple[Integer[Array, " t"], Integer[Array, " t"]]
    game_setup: MABGameSetupBase

    def get_inner_actions(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        output_0, output_1 = self.outputs
        return (
            output_0[1],
            output_1[1],
        )

    @property
    @override
    def actions_vec(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        return (
            jax.nn.one_hot(self.actions[0], self.game.shape[0]),
            jax.nn.one_hot(self.actions[1], self.game.shape[1]),
        )

    def to_inner_output(self) -> "MABInnerOutput":
        return MABInnerOutput(
            max_iter=self.max_iter,
            game=self.game,
            learner=self.learner,
            actions=self.actions,
            losses=self.losses,
            outputs=self.outputs,
            inner_actions=self.get_inner_actions(),
            game_setup=self.game_setup,
        )


@typed_module
class MABInnerOutput(MABPipelineOutput):
    inner_actions: tuple[Float[Array, "t n"], Float[Array, "t m"]]

    @property
    @override
    def actions_vec(
        self,
    ) -> tuple[Float[Array, "t n"], Float[Array, "t m"]]:
        return self.inner_actions


@typed
def online_game_pipeline(
    game_setup: GameSetup,
    learner_config: OnlineGameLearnerConfig,
    max_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> OnlinePipelineOutput:
    (game_key,) = jr.split(key, 1) if key is not None else (None,)
    game = game_setup._feed_key(key)._game()
    n, m = game.shape

    learner_config = canonicalize_online_game_learner_config(
        learner_config
    )

    learner_and_inputs = (
        learner_config[0]._init_learner(n, max_iter),
        learner_config[1]._init_learner(m, max_iter),
    )
    learner = (learner_and_inputs[0][0], learner_and_inputs[1][0])
    inputs = (learner_and_inputs[0][1], learner_and_inputs[1][1])

    # assert is_bearable(learner, tuple[OnlineLearner, OnlineLearner])
    # assert is_bearable(inputs, tuple[Vmapped[Any, " t"], Vmapped[Any, " t"]])
    learner, (actions, losses, outputs) = online_matrix_game_play(
        learner, game, inputs, length=max_iter
    )

    return OnlinePipelineOutput(
        max_iter=max_iter,
        game=game,
        learner=learner,
        actions=actions,
        losses=losses,
        outputs=outputs,
    )


@typed
def online_game_pipeline_continue(
    last_result: OnlinePipelineOutput,
    learner_config: OnlineGameLearnerConfig,
    max_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> OnlinePipelineOutput:
    game = last_result.game
    n, m = game.shape
    learner_config = canonicalize_online_game_learner_config(
        learner_config
    )
    inputs = (
        learner_config[0]._init_learner(n, max_iter)[1],
        learner_config[1]._init_learner(m, max_iter)[1][1],
    )
    learner = last_result.learner

    # assert is_bearable(learner, tuple[OnlineLearner, OnlineLearner])
    # assert is_bearable(inputs, tuple[Vmapped[Any, " t"], Vmapped[Any, " t"]])
    learner, (actions, losses, outputs) = online_matrix_game_play(
        learner, game, inputs, length=max_iter
    )

    return OnlinePipelineOutput(
        max_iter=last_result.max_iter + max_iter,
        game=game,
        learner=learner,
        actions=treemap_concat(last_result.actions, actions),
        losses=jnp.concatenate([last_result.losses, losses]),
        outputs=treemap_concat(last_result.outputs, outputs),
    )


@typed
def mab_game_pipeline(
    game_setup: MABGameSetup,
    learner_config: MABCompatibleLearnerConfig,
    max_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> MABPipelineOutput:
    (game_key, learner_key) = (
        jr.split(key, 2) if key is not None else (None, None)
    )
    game_setup = canonicalize_game_setup(game_setup)
    game_setup = game_setup._feed_key(game_key)
    game = game_setup._game()
    n, m = game.shape
    learner_config_ = canonicalize_mab_game_learner_config(
        learner_config, learner_key
    )
    learner_and_inputs = (
        learner_config_[0]._init_learner(n, max_iter),
        learner_config_[1]._init_learner(m, max_iter),
    )
    learner = (learner_and_inputs[0][0], learner_and_inputs[1][0])
    inputs = (learner_and_inputs[0][1], learner_and_inputs[1][1])

    # assert is_bearable(learner, tuple[MABLearner, MABLearner])
    # assert is_bearable(inputs, tuple[Vmapped[Any, " t"], Vmapped[Any, " t"]])
    (
        learner,
        (actions, losses, outputs),
    ) = mab_bernoulli_matrix_game_play(
        learner, game, inputs, game_setup._draws(max_iter)
    )

    return MABPipelineOutput(
        max_iter=max_iter,
        game=game,
        learner=learner,
        actions=actions,
        losses=losses,
        outputs=outputs,
        game_setup=game_setup,
    )


@typed
def mab_game_pipeline_continue(
    last_result: MABPipelineOutput,
    learner_config: MABCompatibleLearnerConfig,
    max_iter: int,
    key: Key[Scalar, ""] | None = None,
) -> MABPipelineOutput:
    (game_key, learner_key) = (
        jr.split(key, 2) if key is not None else (None, None)
    )
    game_setup = last_result.game_setup._feed_key(game_key)
    game = last_result.game
    n, m = game.shape
    learner_config_ = canonicalize_mab_game_learner_config(
        learner_config, learner_key
    )
    inputs = (
        learner_config_[0]._init_learner(n, max_iter)[1],
        learner_config_[1]._init_learner(m, max_iter)[1],
    )
    learner = last_result.learner

    # assert is_bearable(learner, tuple[MABLearner, MABLearner])
    # assert is_bearable(inputs, tuple[Vmapped[Any, " t"], Vmapped[Any, " t"]])
    learner, (actions, losses, outputs) = (
        mab_bernoulli_matrix_game_play(
            learner, game, inputs, game_setup._draws(max_iter)
        )
    )

    return MABPipelineOutput(
        max_iter=last_result.max_iter + max_iter,
        game=game,
        learner=learner,
        actions=treemap_concat(last_result.actions, actions),
        losses=jnp.concatenate([last_result.losses, losses]),
        outputs=treemap_concat(last_result.outputs, outputs),
        game_setup=game_setup,
    )
