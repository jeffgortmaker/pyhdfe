"""Algorithms for fixed effect absorption."""

import abc
import itertools
import functools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from .utilities import Array, Groups, identify_singletons, max_norm_convergence


class Algorithm(abc.ABC):
    """Algorithm for absorbing fixed effects. Class attributes contain counts of observations and fixed effect
    dimensions, and if computed, singletons and degrees of freedom used by the fixed effects.

    An algorithm is initialized by :func:`create` with one or more dimensions of fixed effects specified by ``ids``.
    Once initialized, :meth:`Algorithm.residualize` absorbs the fixed effects into a matrix and returns the residuals
    from a regression of each matrix column on the fixed effects.

    Attributes
    ----------
    observations : `int`
        Number of observations in the data (i.e., the number of rows in ``ids``).
    dimensions : `int`
        Number of fixed effect dimensions (i.e., the number of columns in ``ids``).
    singletons : `int or None`
        Number of singleton groups or observations. This will be ``None`` if there was no need to identify singletons
        (i.e., if ``drop_singletons`` and ``compute_degrees`` were both ``False`` in :func:`create`).
    degrees : `int or None`
        Exact or approximate number of degrees of freedom used by the fixed effects computed according to
        ``degrees_method`` in :func:`create`. This will be ``None`` if ``compute_degrees`` was ``False`` in
        :func:`create`.

    Examples
    --------
        - :doc:`Tutorial </tutorial>`

    """

    observations: int
    dimensions: int
    singletons: Optional[int]
    degrees: Optional[int]
    _lb: int = 1
    _ub: Optional[int] = None
    _groups_list: List[Groups]
    _singleton_indices: Optional[Array]

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str]) -> None:
        """Validate IDs, optionally drop singletons, initialize group information, and compute counts."""

        # validate fixed effect IDs
        ids = np.atleast_2d(ids)
        if len(ids.shape) != 2:
            raise ValueError("Fixed effect IDs should be a two-dimensional array.")

        # validate dimensions
        self.observations, self.dimensions = ids.shape
        if self.dimensions > self.observations:
            raise ValueError("Fixed effect IDs should not have more columns than rows.")
        if self.dimensions < self._lb:
            raise ValueError(f"The minimum number of fixed effects supported by this algorithm is {self._lb}.")
        if self._ub is not None and self.dimensions > self._ub:
            raise ValueError(f"The maximum number of fixed effects supported by this algorithm is {self._ub}.")

        # validate groups
        self._groups_list = [Groups(i) for i in ids.T]
        if any(g.group_count < 2 for g in self._groups_list[1:]):
            raise ValueError("All fixed effects after the first one should have more than one level.")

        # count and drop singletons
        self._singleton_indices = self.singletons = None
        if drop_singletons:
            self._singleton_indices = identify_singletons(self._groups_list)
            self.singletons = int(self._singleton_indices.sum())
            self._groups_list = [Groups(g.codes[~self._singleton_indices]) for g in self._groups_list]

        # count degrees of freedom and singletons as a by-product
        self.degrees = None
        if compute_degrees:
            self.degrees, self.singletons = self._compute_degrees(cluster_ids, degrees_method)

    def _compute_degrees(self, cluster_ids: Optional[Array], degrees_method: Optional[str]) -> Tuple[int, int]:
        """Exactly compute or approximate the degrees of freedom used by the fixed effects. As a by-product, count the
        number of singletons.
        """

        # validate the method
        methods = {'simple', 'pairwise', 'exact'}
        if degrees_method is None:
            degrees_method = 'pairwise' if self.dimensions > 1 else 'simple'
        if degrees_method not in methods:
            ValueError(f"degrees_method should be None or one of {sorted(methods)}.")

        # drop singletons
        if self._singleton_indices is not None:
            singleton_indices = self._singleton_indices
            groups_list = self._groups_list
        else:
            singleton_indices = identify_singletons(self._groups_list)
            groups_list = [Groups(g.codes[~singleton_indices]) for g in self._groups_list]

        # validate cluster IDs and drop groups that are nested within cluster groups
        if cluster_ids is not None:
            cluster_ids = np.atleast_2d(cluster_ids)
            if len(cluster_ids.shape) != 2:
                raise ValueError("Cluster IDs should be a two-dimensional array.")
            if cluster_ids.shape[0] != self.observations:
                raise ValueError("Cluster IDs should have the same number of observations as fixed effect IDs.")
            cluster_groups_list = [Groups(i) for i in cluster_ids[~singleton_indices].T]
            groups_list = [g for g in groups_list if not any(g.within(c) for c in cluster_groups_list)]

        # count degrees of freedom
        if not groups_list:
            degrees = 0
        elif degrees_method == 'exact':
            D = np.hstack([g.dense_dummies(drop_last=i > 0) for i, g in enumerate(groups_list)])
            degrees = int(np.linalg.matrix_rank(D))
        else:
            degrees = sum(g.group_count for g in groups_list)
            if degrees_method == 'simple':
                degrees -= len(groups_list) - 1
            else:
                assert degrees_method == 'pairwise'
                degrees -= sum(max(map(g.components, groups_list[:i])) for i, g in enumerate(groups_list) if i > 0)

        # count singletons
        singletons = int(singleton_indices.sum())
        return degrees, singletons

    def residualize(self, matrix: Array, weights: Optional[Array] = None) -> Array:
        """Absorb the fixed effects into a matrix and return the residuals from a regression of each column of the
        matrix on the fixed effects.

        .. warning::

           This function assumes that all of your data have already been cleaned. For example, it will not drop
           observations with null values.

        Parameters
        ----------
        matrix : `array-like`
            The two-dimensional array to residualize, which should have a number of rows equal to
            :attr:`Algorithm.observations` (i.e., the number of rows in the ``ids`` passed to :func:`create`).
        weights: `array-like`.
            Two-dimensional array with weights, which should have number of rows equal to
            :attr:`Algorithm.observations` (i.e., the number of rows in the ``ids`` passed to :func:`create`)
            and one column. Optional argument, `None` by default. If provided, `matrix` is residualized by
            substracting weighted means.

        Returns
        -------
        `ndarray`
            Residuals from a regression of each column of ``matrix`` on the fixed effects. This matrix has the same
            number of columns as ``matrix``. If any singleton observations were dropped when initializing the
            :class:`Algorithm` (this is the default behavior of :func:`create`), the residualized matrix will have
            correspondingly fewer rows.

        Examples
        --------
            - :doc:`Tutorial </tutorial>`

        """
        matrix = np.atleast_2d(matrix)
        if len(matrix.shape) != 2:
            raise ValueError("matrix should be a two-dimensional array.")
        if matrix.shape[0] != self.observations:
            raise ValueError("matrix should have the same number of rows as fixed effect IDs.")
        if self._singleton_indices is not None:
            matrix = matrix[~self._singleton_indices]

        if weights is not None:
            #print(type(self))
            #if type(self) not in (Within, MAP):
            #    raise ValueError("weights are only supported for Dummy, Within and MAP algorithms.")
            weights = np.atleast_2d(weights)
            if len(weights.shape) != 2:
                raise ValueError("weights should be a two-dimensional array.")
            if weights.shape[0] != self.observations:
                raise ValueError("weights should have the same number of rows as fixed effect IDs.")
            if self._singleton_indices is not None:
                weights = weights[~self._singleton_indices]

        return self._residualize_matrix(matrix, weights)

    @abc.abstractmethod
    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Residualize a matrix. If weights are provided, residualize by the *weighted* mean."""


class Dummy(Algorithm):
    """Fixed effect absorption with dummy variables."""

    _D: Array

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str]) -> None:
        """Create dummy variables."""
        super().__init__(ids, cluster_ids, drop_singletons, compute_degrees, degrees_method)
        self._D = np.hstack([g.dense_dummies(drop_last=i > 0) for i, g in enumerate(self._groups_list)])

    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Compute residuals from regressions of each matrix column on the dummy variables.
           Weights are currently not supported.
        """
        if weights is not None:
            raise ValueError("weights are not supported for the Dummy algorithm.")
        return matrix - self._D @ scipy.linalg.inv(self._D.T @ self._D) @ self._D.T @ matrix

class Within(Algorithm):
    """One-dimensional fixed effect absorption with the within transformation."""

    _ub = 1

    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """De-mean a matrix within groups. If weights are provided, substract a weighted mean
           (instead of an unweighted mean).
        """
        assert len(self._groups_list) == 1
        groups = self._groups_list[0]
        return matrix - groups.expand(groups.mean(matrix, weights))


class SW(Algorithm):
    """Two-dimensional fixed effect absorption with the algorithm of Somaini and Wolak (2016)."""

    _lb = _ub = 2
    _D: scipy.sparse.csr_matrix
    _H: scipy.sparse.csc_matrix
    _DH: scipy.sparse.csr_matrix
    _DD_inv: scipy.sparse.csr_matrix
    _C: scipy.sparse.csr_matrix
    _B: scipy.sparse.csr_matrix

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str]) -> None:
        """Construct algorithm components."""
        super().__init__(ids, cluster_ids, drop_singletons, compute_degrees, degrees_method)

        # construct sparse matrices
        assert len(self._groups_list) == 2
        self._D = self._groups_list[0].sparse_dummies().tocsr()
        self._H = self._groups_list[1].sparse_dummies(drop_last=True).tocsr()

        # compute the straightforward components of the annihilator matrix
        self._DH = self._D.T @ self._H
        self._DD_inv = scipy.sparse.diags(1 / (self._D.T @ self._D).diagonal())

        # attempt to compute the only non-diagonal inverse
        C_inv = self._H.T @ self._H - self._DH.T @ self._DD_inv @ self._DH
        try:
            self._C = scipy.sparse.csr_matrix(scipy.linalg.inv(C_inv.toarray()))
        except scipy.linalg.LinAlgError:
            raise RuntimeError("Failed to invert the C matrix in the Somaini-Wolak algorithm.")

        # compute the remaining component
        self._B = -self._DD_inv @ self._DH @ self._C

    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Complete the algorithm. Weights are currently not supported."""

        if weights is not None:
            raise ValueError("weights are not supported for the SW algorithm.")

        matrix = scipy.sparse.csr_matrix(matrix)
        Dx = self._D.T @ matrix
        Hx = self._H.T @ matrix
        ADx = self._DD_inv @ Dx + self._DD_inv @ (self._DH @ (self._C @ (self._DH.T @ (self._DD_inv @ Dx))))
        delta_hat = ADx + self._B @ Hx
        tau_hat = self._B.T @ Dx + self._C @ Hx
        return (matrix - self._D @ delta_hat - self._H @ tau_hat).toarray()


class FixedPoint(Algorithm, abc.ABC):
    """Abstract fixed point iteration algorithm."""

    _iteration_limit: int
    _converged: Optional[functools.partial]

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str], iteration_limit: int, tol: float,
            converged: Optional[Callable[[Array, Array], bool]]) -> None:
        """Validate fixed point options."""
        super().__init__(ids, cluster_ids, drop_singletons, compute_degrees, degrees_method)
        if not isinstance(iteration_limit, int) or iteration_limit <= 0:
            raise ValueError("iteration_limit should be a positive integer.")
        if not isinstance(tol, (int, float)) or tol < 0:
            raise ValueError("tol should be a nonnegative float.")
        if converged is not None and not callable(converged):
            raise TypeError("converged should be None or a callable function.")
        self._iteration_limit = iteration_limit
        if converged is not None:
            self._converged = functools.partial(converged)
        elif tol > 0:
            self._converged = functools.partial(max_norm_convergence, tol=tol)
        else:
            self._converged = None

    def _terminate(self, last_matrix: Array, matrix: Array, iterations: int) -> bool:
        """Check for convergence for whether the iteration limit has been exceeded."""
        converged = False
        if self._converged is not None:
            converged = self._converged(last_matrix, matrix)
        if not converged and iterations >= self._iteration_limit:
            raise RuntimeError(f"Failed to converge after {iterations} iterations.")
        return converged


class MAP(FixedPoint):
    """Fixed effect absorption with the method of alternating projections."""

    _transform: str
    _acceleration: str
    _acceleration_tol: float

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str], iteration_limit: int, tol: float,
            converged: Optional[Callable[[Array, Array], bool]], transform: str, acceleration: str,
            acceleration_tol: float) -> None:
        """Validate transform and acceleration options."""
        super().__init__(
            ids, cluster_ids, drop_singletons, compute_degrees, degrees_method, iteration_limit, tol, converged
        )
        transforms = {'kaczmarz', 'symmetric', 'cimmino'}
        accelerations = {'none', 'gk', 'cg'}
        if transform not in transforms:
            raise ValueError(f"transform must be one of {sorted(transforms)}.")
        if acceleration not in accelerations:
            raise ValueError(f"acceleration must be one of {sorted(accelerations)}.")
        if transform == 'kaczmarz' and acceleration == 'cg':
            raise ValueError("The asymmetric 'kaczmarz' transform does not support 'cg' acceleration.")
        if not isinstance(acceleration_tol, (int, float)) or acceleration_tol < 0:
            raise ValueError("acceleration_tol should be a nonnegative float.")
        if self._converged is None:
            raise ValueError("There should be a convergence criteria.")
        self._transform = transform
        self._acceleration = acceleration
        self._acceleration_tol = acceleration_tol

    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Residualize a matrix with fixed point iteration. If weights are provided,
           residualize by a *weighted* mean.
        """
        accelerations = {
            'none': self._iterate,
            'gk': self._apply_gk,
            'cg': self._apply_cg,
        }
        return accelerations[self._acceleration](matrix, weights)

    def _iterate(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Iteratively transform a matrix without acceleration."""
        iterations = 0
        while True:
            last_matrix = matrix
            matrix = self._transform_matrix(matrix, weights)

            # check for termination
            iterations += 1
            if self._terminate(last_matrix, matrix, iterations):
                break

        return matrix

    def _apply_gk(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Accelerate iteration with the Gearhart-Koshy method. For each vector, acceleration is only used when the sum
        of squared residuals relative to the sum of squared vector values is greater than the acceleration tolerance and
        when the t value is greater than its expected upper bound of 0.5.
        """

        iterations = 0
        while True:
            last_matrix = matrix
            matrix = self._transform_matrix(matrix, weights)

            # accelerated step
            for vector, last_vector in zip(matrix.T, last_matrix.T):
                residual = vector - last_vector
                ssr = residual @ residual
                if ssr > self._acceleration_tol * (last_vector @ last_vector):
                    t = -(last_vector @ residual) / ssr
                    if t > 0.5:
                        vector[:] = t * vector + (1 - t) * last_vector

            # check for termination
            iterations += 1
            if self._terminate(last_matrix, matrix, iterations):
                break

        return matrix

    def _apply_cg(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Accelerate iteration with the conjugate gradient method. For each vector, acceleration is used until the
        first time that the sum of squared residuals is less than the acceleration tolerance.
        """

        # initialize algorithm components
        matrix = matrix.copy()
        residual = self._transform_matrix(matrix, weights) - matrix
        ssr = np.sum(residual**2, axis=0, keepdims=True)
        u = residual.copy()

        # identify vectors that can be accelerated
        last_apply = np.ones(ssr.size, bool)
        apply = ssr.flatten() >= self._acceleration_tol

        # iterate until termination
        iterations = 0
        while True:
            last_matrix = matrix.copy()
            if not apply.all():
                transform = ~apply
                matrix[:, transform] = self._transform_matrix(matrix[:, transform], weights)

            # accelerated step
            if apply.any():
                if (apply != last_apply).any():
                    new_apply = apply[last_apply]
                    residual = residual[:, new_apply]
                    ssr = ssr[:, new_apply]
                    u = u[:, new_apply]

                # apply the step to the accelerated vectors
                if self._transform == 'cimmino':
                    v = self._transform_matrix(u, weights, cimmino_difference=True)
                else:
                    v = u - self._transform_matrix(u, weights)
                alpha = ssr / np.sum(u * v, axis=0, keepdims=True)
                matrix[:, apply] += alpha * u
                residual -= alpha * v
                last_ssr = ssr
                ssr = np.sum(residual**2, axis=0, keepdims=True)
                beta = ssr / last_ssr
                u = residual + beta * u

                # identify vectors that can be accelerated
                last_apply = apply.copy()
                apply[apply] = ssr.flatten() >= self._acceleration_tol

            # check for termination
            iterations += 1
            if self._terminate(last_matrix, matrix, iterations):
                break

        return matrix

    def _transform_matrix(self, matrix: Array, weights: Union[Array, None], cimmino_difference: bool = False) -> Array:
        """Transform a matrix according to using the specified method. Optionally compute the difference compared to the
        original matrix for the Cimmino transform (this isn't possible for the others).
        """
        if self._transform == 'kaczmarz':
            for groups in self._groups_list:
                matrix = matrix - groups.expand(groups.mean(matrix, weights))
        elif self._transform == 'symmetric':
            for groups in itertools.chain(self._groups_list, reversed(self._groups_list)):
                matrix = matrix - groups.expand(groups.mean(matrix, weights))
        else:
            assert self._transform == 'cimmino'
            difference = sum(g.expand(g.mean(matrix, weights)) for g in self._groups_list) / self.dimensions
            matrix = difference if cimmino_difference else matrix - difference
        return matrix


class LSMR(FixedPoint):
    """Fixed effect absorption with the LSMR solver of Fong and Sanders (2011). This is a trimmed down version of
    scipy.sparse.linalg.lsmr that has been modified for simultaneous iteration over multiple matrix columns, custom
    convergence criteria, and optional termination conditions.
    """

    _residual_tol: float
    _condition_limit: float
    _A: scipy.sparse.linalg.LinearOperator

    def __init__(
            self, ids: Array, cluster_ids: Optional[Array], drop_singletons: bool, compute_degrees: bool,
            degrees_method: Optional[str], iteration_limit: int, tol: float,
            converged: Optional[Callable[[Array, Array], bool]], residual_tol: float, condition_limit: float) -> None:
        """Validate tolerances and create a sparse matrix of dummy variables."""
        super().__init__(
            ids, cluster_ids, drop_singletons, compute_degrees, degrees_method, iteration_limit, tol, converged
        )
        if not isinstance(residual_tol, (int, float)) or residual_tol < 0:
            raise ValueError("residual_tol should be a nonnegative float.")
        if not isinstance(condition_limit, (int, float)) or condition_limit < 0:
            raise ValueError("condition_limit should be a nonnegative float.")
        if self._converged is None and residual_tol == 0:
            raise ValueError("There should be at least one convergence criteria.")
        self._residual_tol = residual_tol
        self._condition_limit = condition_limit
        self._A = scipy.sparse.linalg.aslinearoperator(
            scipy.sparse.hstack([g.sparse_dummies(drop_last=i > 0).tocsc() for i, g in enumerate(self._groups_list)])
        )

    @staticmethod
    def _orthogonal_transformation(a: Array, b: Array) -> Tuple[Array, Array, Array]:
        """Construct Given's plane rotation."""
        c, s = scipy.linalg.blas.drotg(a, b)
        r = b / s if abs(b) > abs(a) else a / c
        return c, s, r

    def _residualize_matrix(self, matrix: Array, weights: Union[Array, None]) -> Array:
        """Compute fitted values for each column with LSMR and form residuals.
           Weights are currently not supported.
        """

        if weights is not None:
            raise ValueError("weights are not supported for the LSMR algorithm.")

        # collect dimensions
        matrix_transpose = matrix.T
        j, k = matrix_transpose.shape
        m, n = self._A.shape

        # initialize primary variables
        u = matrix_transpose.copy()
        v, h, h_bar, x = np.zeros((4, j, n))
        rho, rho_bar, c_bar = np.ones((3, j))
        alpha, alpha_bar, zeta, zeta_bar, s_bar = np.zeros((5, j))

        # initialize variables for condition number estimation
        max_rho_bar = min_rho_bar = np.zeros(0)
        if np.isfinite(self._condition_limit):
            max_rho_bar = np.zeros(j)
            min_rho_bar = np.full(j, np.inf)

        # initialize variables for fixed effect residual computation
        rho_dot = beta_dot_dot = beta_dot = last_tau_tilde = theta_tilde = d = norm_B = np.zeros(0)
        if self._residual_tol > 0:
            rho_dot = np.ones(j)
            beta_dot_dot, beta_dot, last_tau_tilde, theta_tilde, d, norm_B = np.zeros((6, j))

        # compute non-constant initial variables
        for i, vector in enumerate(matrix_transpose):
            # initialize the bidiagonalization
            beta = np.linalg.norm(u[i])
            if beta > 0:
                u[i] /= beta
                v[i] = self._A.rmatvec(u[i])
                alpha[i] = np.linalg.norm(v[i])
                if alpha[i] > 0:
                    v[i] /= alpha[i]
            else:
                v[i] = np.zeros(n)
                alpha[i] = 0

            # fill primary variables
            zeta_bar[i] = alpha[i] * beta
            alpha_bar[i] = alpha[i]
            h[i] = v[i].copy()

            # fill variables for fixed effect residual computation
            if self._residual_tol > 0:
                beta_dot_dot[i] = beta
                norm_B[i] = alpha[i]**2

        # iterate until each vector converges
        iterations = 0
        matrix = matrix.copy()
        converged = np.zeros(j, bool)
        while True:
            last_matrix = None if self._converged is None else matrix.copy()
            for i, vector in enumerate(matrix_transpose):
                if self._converged is None and converged[i]:
                    continue

                # continue the bidiagonalization
                u[i] = self._A.matvec(v[i]) - alpha[i] * u[i]
                beta = np.linalg.norm(u[i])
                if beta > 0:
                    u[i] /= beta
                    v[i] = self._A.rmatvec(u[i]) - beta * v[i]
                    alpha[i] = np.linalg.norm(v[i])
                    if alpha[i] > 0:
                        v[i] /= alpha[i]

                # construct rotation P hat
                c_hat, s_hat, alpha_hat = self._orthogonal_transformation(alpha_bar[i], b=0.0)

                # construct and apply rotation P
                last_rho = rho[i]
                c, s, rho[i] = self._orthogonal_transformation(alpha_hat, beta)
                theta = s * alpha[i]
                alpha_bar[i] = c * alpha[i]

                # construct and apply rotation P bar
                last_rho_bar = rho_bar[i]
                last_zeta = zeta[i]
                theta_bar = s_bar[i] * rho[i]
                last_c_bar_rho = c_bar[i] * rho[i]
                c_bar[i], s_bar[i], rho_bar[i] = self._orthogonal_transformation(last_c_bar_rho, theta)
                zeta[i] = c_bar[i] * zeta_bar[i]
                zeta_bar[i] *= -s_bar[i]

                # check whether the condition number limit has been exceeded
                if np.isfinite(self._condition_limit):
                    max_rho_bar[i] = max(max_rho_bar[i], last_rho_bar)
                    if iterations > 0:
                        min_rho_bar[i] = min(min_rho_bar[i], last_rho_bar)
                    cond_B = max(max_rho_bar[i], last_c_bar_rho) / min(min_rho_bar[i], last_c_bar_rho)
                    if cond_B > self._condition_limit:
                        raise RuntimeError(f"Failed to converge with an estimated condition number of {cond_B}.")

                # update h bar, x, and h
                h_bar[i] = h[i] - (theta_bar * rho[i] / (last_rho * last_rho_bar)) * h_bar[i]
                x[i] += (zeta[i] / (rho[i] * rho_bar[i])) * h_bar[i]
                h[i] = v[i] - (theta / rho[i]) * h[i]

                # compute the residualized vector and move on if fixed effect residual termination is disabled
                matrix[:, i] = vector - self._A.matvec(x[i])
                if self._residual_tol == 0:
                    continue

                # apply rotation P hat
                beta_acute = c_hat * beta_dot_dot[i]
                beta_check = -s_hat * beta_dot_dot[i]

                # apply rotation P
                beta_hat = c * beta_acute
                beta_dot_dot[i] = -s * beta_acute

                # construct and apply rotation P tilde
                last_theta_tilde = theta_tilde[i]
                last_c_tilde, last_s_tilde, last_rho_tilde = self._orthogonal_transformation(rho_dot[i], theta_bar)
                theta_tilde[i] = last_s_tilde * rho_bar[i]
                rho_dot[i] = last_c_tilde * rho_bar[i]
                beta_dot[i] = -last_s_tilde * beta_dot[i] + last_c_tilde * beta_hat

                # update t tilde by forward substitution
                last_tau_tilde[i] = (last_zeta - last_theta_tilde * last_tau_tilde[i]) / last_rho_tilde
                tau_dot = (zeta[i] - theta_tilde[i] * last_tau_tilde[i]) / rho_dot[i]

                # check for fixed effect residual convergence
                d[i] += beta_check**2
                norm_r = d[i] + (beta_dot[i] - tau_dot)**2 + beta_dot_dot[i]**2
                norm_A_r = abs(zeta_bar[i])
                norm_B[i] += beta**2
                converged[i] = norm_A_r <= self._residual_tol * np.sqrt(norm_B[i]) * np.sqrt(norm_r)
                norm_B[i] += alpha[i]**2

            # check for termination
            iterations += 1
            if self._terminate(last_matrix, matrix, iterations) or converged.all():
                break

        return matrix
