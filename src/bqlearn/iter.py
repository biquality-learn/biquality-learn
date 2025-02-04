import warnings
from functools import singledispatch

from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor


@singledispatch
def iterative(estimator, steps: int = 1, reset: bool = True):
    raise NotImplementedError(
        f"{estimator.__class__.__name__} doesn't support iterative"
        " learning. Register the estimator class to iterative."
    )


# @iterative.register(Pipeline)
# def _iterative_pipeline(estimator, steps=1, reset=True):
#     it = iterative(estimator[-1], steps=steps, reset=reset)
#     if reset:
#         yield Pipeline(
#             estimator[:-1].steps + [(estimator.steps[-1][0], next(it))],
#             memory=estimator.memory,
#             verbose=estimator.verbose,
#         )
#     while True:
#         FrozenEstimator(estimator[:-1].steps)


@iterative.register(GradientBoostingClassifier)
@iterative.register(GradientBoostingRegressor)
@iterative.register(RandomForestClassifier)
@iterative.register(RandomForestRegressor)
@iterative.register(ExtraTreesClassifier)
@iterative.register(ExtraTreesRegressor)
@iterative.register(BaggingClassifier)
@iterative.register(BaggingRegressor)
def _iterative_ensemble(estimator, steps=1, reset=True):
    if reset:
        estimator.n_estimators = steps
        estimator.warm_start = False
        yield estimator
    estimator.warm_start = True
    while True:
        estimator.n_estimators += steps
        yield estimator


@iterative.register(HistGradientBoostingClassifier)
@iterative.register(HistGradientBoostingRegressor)
def _iterative_hgb(estimator, steps=1, reset=True):
    if reset:
        estimator.max_iter = steps
        estimator.warm_start = False
        yield estimator
    estimator.warm_start = True
    while True:
        estimator.max_iter += steps
        yield estimator


@iterative.register(LogisticRegression)
@iterative.register(SGDClassifier)
@iterative.register(SGDRegressor)
@iterative.register(MLPClassifier)
@iterative.register(MLPRegressor)
def _iterative_gd(estimator, steps=1, reset=True):
    if hasattr(estimator, "learning_rate") and estimator.learning_rate != "constant":
        warnings.warn(
            f"{estimator.__class__.__name__} has a non constant learning rate \
                and is not supported with iterative learning."
        )
    if reset:
        estimator.max_iter = steps
        estimator.warm_start = False
        yield estimator
    estimator.warm_start = True
    while True:
        estimator.max_iter = steps
        yield estimator
