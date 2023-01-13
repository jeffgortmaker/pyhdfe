"""Fixtures used by tests."""

import itertools
from typing import Any, Iterator, Sequence, Tuple

import numpy as np
import pytest
import scipy.linalg

from .. import create
from ..utilities import Array


# define the type of fixed effect problems
Problem = Tuple[int, int, Array, Array, Array, Array]


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
            yield pytest.param([covariates, scales, levels, singletons], id="-".join([
                f"covariates: {covariates}",
                f"scales: {scales}",
                f"levels: {levels}",
                f"singletons: {int(100 * singletons)}%"
            ]))


def simulate_data(
        covariates: int, scales: Sequence[int], levels: Sequence[int], singletons: float,
        state: np.random.RandomState) -> Tuple[Array, Array, Array]:
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
    return ids, X, y


@pytest.fixture(scope='session', params=get_parameters())
def problem(request: Any) -> Problem:
    """Simulate a fixed effect problem against which algorithms can be compared."""
    covariates, scales, levels, singletons = request.param

    # simulate the data
    state = np.random.RandomState(hash(tuple(request.param)) % 2**32)
    ids, X, y = simulate_data(covariates, scales, levels, singletons, state)

    # count degrees of freedom
    algorithm = create(ids, degrees_method='exact')
    assert algorithm.degrees is not None

    # run a regression
    y1, X1 = np.split(algorithm.residualize(np.c_[y, X]), [1], axis=1)
    beta = scipy.linalg.inv(X1.T @ X1) @ X1.T @ y1
    return algorithm.observations, algorithm.degrees, y, X, ids, beta[:X.shape[1]]
