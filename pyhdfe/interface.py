"""Public-facing interface."""

from typing import Dict, Optional

from .algorithms import Algorithm, Dummy, Within, SW, MAP, LSMR
from .utilities import Array


def create(
        ids: Array, weights: Optional[Array] = None, cluster_ids: Optional[Array] = None, drop_singletons: bool = True, compute_degrees: bool = True,
        degrees_method: Optional[str] = None, residualize_method: Optional[str] = None,
        options: Optional[dict] = None) -> Algorithm:
    r"""Initialize an algorithm for absorbing fixed effects.

    By default, simple de-meaning is used for a single fixed effect, and non-accelerated de-meaning is used for more
    than one dimension. This is the most conservative and simplest algorithm for fixed effect absorption. If it is
    taking a long time, consider switching to a faster ``residualize_method`` and using different ``options``.

    When an algorithm is initialized, by default, singletons are dropped and degrees of freedom are computed. If either
    behavior isn't needed, or if degrees of freedom computation is taking a long time, consider using a more
    conservative ``degrees_method`` or disabling these behaviors with ``drop_singletons`` and ``compute_degrees``.

    .. warning::

       This function assumes that all of your data have already been cleaned. For example, it will not drop observations
       with null values.

    Parameters
    ----------
    ids : `array-like`
        Two-dimensional array of fixed effect identifiers. Columns are fixed effect dimensions and rows are
        observations. Identifiers can be integers, strings, or other hashable data types. Columns after the first should
        have more than one unique value.
    weights: `array-like, optional`
        Two-dimensional array of weights. Weights should be non-negative, have the number of rows as ``ids``, and one column.
        By default, all observations are weighted equally.
    cluster_ids : `array-like, optional`
        Two-dimensional array of cluster group identifiers, which if specified will be used when computing degrees of
        freedom. If a fixed effect (i.e., a column in ``ids``) is nested within a cluster (i.e., a column of this
        matrix), it will not contribute towards degrees of freedom used by the fixed effects. For more information, see
        :ref:`references:Correia (2015)`.
    drop_singletons : `bool, optional`
        Whether to drop singleton groups or observations in ``ids`` when initializing the algorithm. Singletons groups
        are fixed effect groups with only one observation. By default, singletons are dropped. When dropped, the number
        of singleton groups is equal to the number of rows in ``ids`` minus :attr:`Algorithm.observations`. For more
        information about singletons and why they are typically dropped, see :ref:`references:Correia (2015)`.
    compute_degrees : `bool, optional`
        Whether to compute the number of degrees of freedom used by the fixed effects. By default, degrees of freedom
        are computed.
    degrees_method : `str, optional`
        How to compute or approximate the number of degrees of freedom used by the fixed effects that aren't nested
        within any ``cluster_ids``. The following methods are supported:

            - ``'none'`` (default for one dimension) - Assume there are no redundant fixed effects. This method is exact
              for one dimension (i.e., for one column in ``ids``). It provides the most conservative upper bound for
              multiple dimensions but requires no additional computation.

              For one dimension this method simply counts the number of fixed effect levels (i.e., the number of
              distinct values in ``ids``). Each dimension after the first contributes its number of levels minus one.

            - ``'pairwise'`` (default for multiple dimensions) - Apply the algorithm of
              :ref:`references:Abowd, Creecy, and Kramarz (2002)` to each pair of fixed effect dimensions. This method
              is exact for two dimensions. It provides a smaller upper bound for more than two dimensions but can be
              computationally expensive.

              For one dimension this method is the same as ``'none'``. However, the second dimension contributes its
              number of levels minus the number of connected components in the bipartite graph formed by the two
              dimensions. Each dimension after the second contributes its number of levels minus the maximum number of
              connected components in the bipartite graphs that it forms with prior dimensions. This is the method used
              by :ref:`references:reghdfe`.

            - ``'exact'`` - Apply :func:`numpy.linalg.matrix_rank` to dummy variables constructed from ``ids``. This
              method is exact for any number of dimensions but is typically computationally infeasible. It is meant to
              be a benchmark.

    residualize_method : `str, optional`
        Type of algorithm to initialize. The following methods are supported:

            - ``'within'`` (default for one dimension) - Within transform. Matrix columns are de-meaned within each
              fixed effect group (i.e., each unique value in ``ids``). This algorithm only works for a single fixed
              effect dimension (i.e., one column in ``ids``).

            - ``'map'`` (default for multiple dimensions) - Method of alternating projections applied to fixed effect
              absorption by :ref:`references:Guimarães and Portugal (2010)`, :ref:`references:Gaure (2013a)`,
              :ref:`references:Gaure (2013b)`, and :ref:`references:Correia (2017)`, among others. Matrix columns are
              iteratively de-meaned until convergence. This method works for any number of fixed effect dimensions but
              will be slower than ``'within'`` for one dimension. Variations on this method are used by
              :ref:`references:lfe` and :ref:`references:reghdfe`.

            - ``'lsmr'`` - LSMR method of :ref:`references:Fong and Saunders (2011)`. This implementation is taken from
              :func:`scipy.sparse.linalg.lsmr` and modified for simultaneous iteration over multiple matrix columns and
              custom convergence criteria. Matrix columns are iterated on until convergence. This method works for any
              number of fixed effect dimensions but will be slower than ``'within'`` for one dimension. This is the
              method used by :ref:`references:FixedEffectModels.jl`.

            - ``'sw'`` - Method of :ref:`references:Somaini and Wolak (2016)`. This non-iterative method only works for
              two dimensions (i.e., two columns in ``ids``). To minimize memory usage, the first dimension of fixed
              effects should have fewer levels than the second dimension (i.e., the first column in ``ids`` should have
              fewer unique values than the second column). This is the method used by :ref:`references:res2fe`.

            - ``'dummy'`` - Matrix columns are replaced by residuals from regressions on dummy variables constructed
              from ``ids``. This method works for any number of dimensions but is typically computationally infeasible.
              It is meant to be a benchmark.

    options : `dict, optional`
        Configuration options for the chosen ``method``. The ``'within'``, ``'sw'``, and ``'dummy'`` methods do not
        support any configuration options. The following options are supported by both ``'map'`` and ``'lsmr'``:

            - **iteration_limit** : (`int, optional`) - Maximum number of iterations, after which an exception will be
              raised if the algorithm has not converged. By default, the maximum number of iterations is ``1000000``.

            - **tol** : (`float, optional`) - Common convergence criteria based on the differences between two
              iterations' residualized matrices. By default, algorithms will converge when the maximum absolute value
              of these differences is less than ``1e-8``. Convergence based on this criteria can be disabled by setting
              this value to ``0``.

            - **converged** : (`callable or None, optional`) - Custom convergence criteria, which should be a function
              of the form ``converged(last_matrix, matrix) -> bool`` that accepts the current iteration's residualized
              ``matrix`` and the last iteration's residualized ``last_matrix``. It should return a boolean indicating
              whether the routine has converged. When a custom convergence criteria is used, ``tol`` is ignored.

        The following options are supported only by ``'map'``:

            - **transform** : (`str, optional`) - Transform operator :math:`T` that determines the order of projections
              :math:`P_1, P_2, \dots, P_n` for each of the :math:`n` columns of fixed effects in ``ids``. The following
              transforms are supported:

                  - ``'kaczmarz'`` (default) - Kaczmarz or von Neumann-Halpering operator :math:`T = P_n \cdots P_1`,
                    which is asymmetric and hence does not support ``'cg'`` acceleration.

                  - ``'symmetric'`` - Symmetric Kaczmarz operator :math:`T = P_n \cdots P_1 \cdots P_n`.

                  - ``'cimmino'`` - Symmetric Cimmino operator :math:`T = (P_1 + \cdots + P_n) / n`.

            - **acceleration** : (`str, optional`) - Method used to accelerate fixed point iteration. The following
              methods are supported:

                  - ``'none'`` (default) - Simple non-accelerated fixed point iteration.

                  - ``'gk'`` - Line search method of :ref:`references:Gearhart and Koshy (1989)` applied to fixed effect
                    absorption by :ref:`references:Gaure (2013a)`.

                  - ``'cg'`` - Conjugate gradient method described by
                    :ref:`references:Hernández-Ramos, Escalante, and Raydan (2011)`. This method is not supported by
                    the asymmetric ``'kaczmarz'`` transform.

            - **acceleration_tol** : (`float, optional`) - Acceleration method-specific tolerance for when to stop
              accelerating the convergence of a vector and switch to simple iteration.

              For ``'gk'``, each vector's convergence is accelerated only when the sum of squared residuals relative to
              the sum of squared vector values is greater than this value, which is by default ``1e-16``.

              For ``'cg'``, each vector's convergence is accelerated up until the first time that its sum of squared
              residuals is greater than this value.

        The following options are supported only by ``'lsmr'``:

            - **residual_tol** : (`float, optional`) - Convergence criteria S2 from
              :ref:`references:Fong and Saunders (2011)` based on Stewart's backwards error estimate. This is by default
              ``1e-8``. Convergence based on this criteria can be disabled by setting this value to ``0``.

            - **condition_limit** : (`float, optional`) - Maximum estimated condition number of the matrix of fixed
              effects. For higher estimated condition numbers, an exception will be raised. By default, the maximum
              estimated condition number is ``100000000``.

    Returns
    -------
    `Algorithm`
        Initialized :class:`Algorithm` for absorbing fixed effects. Class attributes contain information about the
        number of observations, the number of fixed effect dimensions, and if computed, the number of singletons and
        degrees of freedom used by the fixed effects.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    # validate the method
    methods = {
        'within': Within,
        'map': MAP,
        'lsmr': LSMR,
        'sw': SW,
        'dummy': Dummy,
    }
    if residualize_method is None:
        residualize_method = 'map' if ids.shape[1] > 1 else 'within'
    if residualize_method not in methods:
        raise ValueError(f"residualize_method must be None or one of {sorted(methods)}.")

    if residualize_method not in ['map', 'within'] and weights is not None:
        raise NotImplementedError(f"residualize_method '{residualize_method}' does not support weights.")

    # validate options
    default_fixed_point_options = {
        'iteration_limit': 1000000,
        'tol': 1e-8,
        'converged': None,
    }
    default_options: Dict[str, dict] = {
        'map': {
            'transform': 'kaczmarz',
            'acceleration': 'none',
            'acceleration_tol': 1e-16,
            **default_fixed_point_options
        },
        'lsmr': {
            'residual_tol': 1e-8,
            'condition_limit': 100000000,
            **default_fixed_point_options
        },
    }
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("options must be None or a dict.")
    updated_options = default_options.get(residualize_method, {}).copy()
    extra_options = set(options) - set(updated_options)
    if extra_options:
        raise ValueError(f"The following options are not supported by '{residualize_method}': {sorted(extra_options)}.")

    # set defaults and initialize the algorithm
    updated_options.update(options)
    algorithm = methods[residualize_method]
    return algorithm(ids, weights, cluster_ids, drop_singletons, compute_degrees, degrees_method, **updated_options)
