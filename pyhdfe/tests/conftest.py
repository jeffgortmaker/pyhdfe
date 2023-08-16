"""Fixtures used by tests."""

import itertools
from typing import Any, Iterator, Sequence, Tuple, Union

import numpy as np
import pytest
import scipy.linalg

from .. import create
from ..utilities import Array


# define the type of fixed effect problems
Problem = Tuple[int, int, Array, Array, Array, Array, Array, Array]


@pytest.fixture(scope='session', autouse=True)
def configure() -> Iterator[None]:
    """Configure NumPy so that it raises all warnings as exceptions."""
    with np.errstate(all='raise'):  # type: ignore
        yield


def get_parameters() -> Iterator[Any]:
    """Generate parameters for different data configurations."""
    for covariates, singletons, dimensions in itertools.product([1, 13], [0.1, 0.0], [1, 2, 3, 4]):
        repeated_product = lambda x: itertools.product(x, repeat=dimensions)
        for scales, levels in itertools.product(repeated_product([1, 2]), repeated_product([1, 4])):
            if np.less(scales, 2).all():
                continue
            if np.less(levels, 2).all():
                continue
            if np.less(levels[1:], 2).any():
                continue
            if np.prod(scales) * np.prod(levels) < covariates + np.sum(levels):
                continue
            for weighted in [False, True]:
                yield pytest.param([covariates, scales, levels, singletons, weighted], id="-".join([
                    f"covariates: {covariates}",
                    f"scales: {scales}",
                    f"levels: {levels}",
                    f"singletons: {int(100 * singletons)}%",
                    f"weighted: {weighted}",
                ]))


def simulate_data(
        state: np.random.RandomState, covariates: int, scales: Sequence[int], levels: Sequence[int], singletons: float,
        weighted: bool) -> Tuple[Array, Array, Array, Union[Array, None]]:
    """Simulate IDs and data matrices."""

    # simulate fixed effects
    ids = np.array(list(itertools.product(*(np.repeat(np.arange(l), s) for s, l in zip(scales, levels)))))
    fe = np.array(list(itertools.product(*(np.repeat(state.normal(size=l), s) for s, l in zip(scales, levels)))))

    # count dimensions
    N, M = ids.shape

    # shuffle the IDs
    for index in range(M):
        indices = np.arange(N)
        state.shuffle(indices)
        ids[indices, index] = ids.copy()[:, index]

    # shuffle and replace shares of the data with singletons
    indices = np.arange(N)
    for index in range(M):
        state.shuffle(indices)
        singleton_indices = indices[:int(singletons * N / M)]
        ids[indices, index] = ids.copy()[:, index]
        ids[singleton_indices, index] = -np.arange(singleton_indices.size)

    # simulate remaining data
    error = state.normal(size=(N, 1))
    X = state.normal(size=(N, covariates))
    y = X.sum(axis=1, keepdims=True) + fe.sum(axis=1, keepdims=True) + error
    weights = state.uniform(size=(N, 1)) if weighted else None

    return ids, X, y, weights


@pytest.fixture(scope='session', params=get_parameters())
def problem(request: Any) -> Problem:
    """Simulate a fixed effect problem against which algorithms can be compared."""

    # simulate the data
    state = np.random.RandomState(hash(tuple(request.param)) % 2**32)
    ids, X, y, weights = simulate_data(state, *request.param)

    # count degrees of freedom
    algorithm = create(ids, degrees_method='exact')
    assert algorithm.degrees is not None

    # drop any weights associated with singletons
    dropped_weights = weights
    if weights is not None and algorithm.singleton_indices is not None:
        dropped_weights = weights[~algorithm.singleton_indices]

    # residualize the matrices
    y1, X1 = np.split(algorithm.residualize(np.c_[y, X], weights), [1], axis=1)

    # optionally weight them
    if dropped_weights is not None:
        X1 *= np.sqrt(dropped_weights)
        y1 *= np.sqrt(dropped_weights)

    # run a regression
    beta = scipy.linalg.inv(X1.T @ X1) @ X1.T @ y1

    return algorithm.observations, algorithm.degrees, y, X, ids, beta[:X.shape[1]], weights, dropped_weights
