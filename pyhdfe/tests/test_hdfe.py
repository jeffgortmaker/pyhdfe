"""Primary tests."""

import operator

import pytest
import numpy as np
import scipy.linalg

from .. import create
from .conftest import Problem


@pytest.mark.usefixtures('problem')
@pytest.mark.parametrize('drop_singletons', [
    pytest.param(False, id="singletons retained"),
    pytest.param(True, id="singletons dropped")
])
@pytest.mark.parametrize(['residualize_method', 'options'], [
    pytest.param('within', {}, id="WT"),
    pytest.param('sw', {}, id="SW"),
    pytest.param('lsmr', {}, id="LSMR"),
    pytest.param('lsmr', {'converged': np.allclose, 'residual_tol': 0}, id="LSMR-custom"),
    pytest.param('lsmr', {'tol': 0}, id="LSMR-residual"),
    pytest.param('map', {'transform': 'kaczmarz', 'acceleration': 'none'}, id="MAP-K"),
    pytest.param('map', {'transform': 'kaczmarz', 'acceleration': 'gk'}, id="MAP-K-GK"),
    pytest.param('map', {'transform': 'symmetric', 'acceleration': 'none'}, id="MAP-SK"),
    pytest.param('map', {'transform': 'symmetric', 'acceleration': 'gk'}, id="MAP-SK-GK"),
    pytest.param('map', {'transform': 'symmetric', 'acceleration': 'cg'}, id="MAP-SK-CG"),
    pytest.param('map', {'transform': 'cimmino', 'acceleration': 'none'}, id="MAP-C"),
    pytest.param('map', {'transform': 'cimmino', 'acceleration': 'gk'}, id="MAP-C-GK"),
    pytest.param('map', {'transform': 'cimmino', 'acceleration': 'cg'}, id="MAP-C-CG"),
    pytest.param('map', {'converged': np.allclose}, id="MAP-custom"),
    pytest.param('map', {'acceleration': 'gk', 'acceleration_tol': np.inf}, id="MAP-GK-simple"),
    pytest.param('map', {'transform': 'cimmino', 'acceleration': 'cg', 'acceleration_tol': np.inf}, id="MAP-CG-simple"),
])
@pytest.mark.parametrize('standardize_weights', [
    pytest.param(True, id="standardize weights"),
    pytest.param(False, id="raw weights")
])
def test_algorithms(problem: Problem, drop_singletons: bool, residualize_method: str, options: dict,
                    standardize_weights: bool) -> None:
    """Test that algorithms give correct estimates."""
    _, _, y, X, ids, beta, weights = problem
    try:
        algorithm = create(ids, drop_singletons=drop_singletons, residualize_method=residualize_method,
                           options=options, standardize_weights=standardize_weights)
    except ValueError as exception:
        if "fixed effects supported" in str(exception):
            return pytest.skip(f"This algorithm does not support {ids.shape[1]}-dimensional fixed effects.")
        raise
    try:
        y1, X1 = np.split(algorithm.residualize(np.c_[y, X], weights), [1], axis=1)
    except NotImplementedError as exception:
        if "weights are not supported" in str(exception):
            if residualize_method == "map":
                return pytest.skip(f"""Weights are not supported for algorithms {residualize_method}
                                   and acceleration {options['acceleration']}.""")
            return pytest.skip(f"Weights are not supported for algorithms {residualize_method}.")
        raise
    if weights is not None:
        if drop_singletons:
            if algorithm._singleton_indices is not None:
                weights = weights[~algorithm._singleton_indices]
        X1 = np.sqrt(weights) * X1
        y1 = np.sqrt(weights) * y1
    beta1 = scipy.linalg.inv(X1.T @ X1) @ X1.T @ y1
    np.testing.assert_allclose(beta, beta1, atol=1e-12, rtol=1e-12, verbose=True)


@pytest.mark.usefixtures('problem')
@pytest.mark.parametrize(['residualize_method', 'options', 'error_text'], [
    pytest.param('lsmr', {'condition_limit': 0}, "condition", id="LSMR-condition"),
    pytest.param('lsmr', {'iteration_limit': 1}, "iteration", id="LSMR-iteration"),
    pytest.param('map', {'iteration_limit': 1}, "iteration", id="MAP-iteration"),
])
def test_limits(problem: Problem, residualize_method: str, options: dict, error_text: str) -> None:
    """Test that iteration and condition number limits can be reached."""
    _, _, y, X, ids, _, _ = problem
    algorithm = create(ids, residualize_method=residualize_method, options=options)
    try:
        algorithm.residualize(np.c_[y, X])
    except Exception as exception:
        assert error_text in str(exception)
    else:
        assert False


@pytest.mark.usefixtures('problem')
@pytest.mark.parametrize('drop_singletons', [
    pytest.param(False, id="singletons retained"),
    pytest.param(True, id="singletons dropped"),
])
@pytest.mark.parametrize(['degrees_method', 'equality_bound'], [
    pytest.param('simple', 1, id="simple"),
    pytest.param('pairwise', 2, id="pairwise"),
])
def test_degrees(problem: Problem, drop_singletons: bool, degrees_method: str, equality_bound: int) -> None:
    """Test that degrees of freedom are well approximated."""
    observations, degrees, _, _, ids, _, _ = problem
    algorithm = create(ids, drop_singletons=drop_singletons, degrees_method=degrees_method)
    np.testing.assert_array_equal(observations, algorithm.observations, verbose=True)
    if ids.shape[1] <= equality_bound:
        np.testing.assert_array_equal(degrees, algorithm.degrees, verbose=True)
    else:
        np.testing.assert_array_compare(operator.le, degrees, algorithm.degrees, verbose=True)


@pytest.mark.usefixtures('problem')
@pytest.mark.parametrize('degrees_method', [
    pytest.param('simple', id="simple"),
    pytest.param('pairwise', id="pairwise"),
    pytest.param('exact', id="exact"),
])
def test_clusters(problem: Problem, degrees_method: str) -> None:
    """Test that fixed effects nested within clusters do not contribute to degrees of freedom."""
    _, _, _, _, ids, _, _ = problem
    algorithm = create(ids, cluster_ids=ids, degrees_method=degrees_method)
    np.testing.assert_array_equal(algorithm.degrees, 0, verbose=True)
