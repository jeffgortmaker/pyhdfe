"""General functionality."""

from typing import Any, Sequence

import numpy as np
import scipy.sparse


# define common types
Array = Any


class Groups(object):
    """Computation of grouped statistics."""

    sort_indices: Array
    reduce_indices: Array
    codes: Array
    counts: Array
    group_count: int
    total_count: int

    def __init__(self, ids_column: Array) -> None:
        """Sort and index IDs that define groups."""

        # sort the IDs
        flat = ids_column.flatten()
        self.sort_indices = flat.argsort()
        sorted_ids = flat[self.sort_indices]

        # identify groups
        changes = np.ones(sorted_ids.shape, bool)
        changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
        self.reduce_indices = np.nonzero(changes)[0]

        # encode the groups
        sorted_codes = np.cumsum(changes) - 1
        self.codes = sorted_codes[self.sort_indices.argsort()]

        # compute counts
        self.total_count = self.codes.size
        self.group_count = self.reduce_indices.size
        self.counts = np.diff(np.append(self.reduce_indices, self.total_count))

    def within(self, other: 'Groups') -> bool:
        """Check if these groups are nested within another set of groups."""
        assert self.total_count == other.total_count
        return (self.min(other.codes) == self.max(other.codes)).all()

    def components(self, other: 'Groups') -> int:
        """Compute the number of connected components in the bipartite graph between these and another set of groups."""
        assert self.total_count == other.total_count
        combined_group_count = self.group_count + other.group_count
        graph = scipy.sparse.coo_matrix(
            (np.ones(self.total_count), (self.codes, self.group_count + other.codes)),
            (combined_group_count, combined_group_count)
        )
        return scipy.sparse.csgraph.connected_components(graph, directed=False)[0]

    def dense_dummies(self, drop_last: bool = False) -> Array:
        """Create a dense matrix of dummy variables, optionally without the first column."""
        columns = self.group_count
        if drop_last:
            columns -= 1
        return np.eye(self.group_count, columns).take(self.codes, axis=0)

    def sparse_dummies(self, drop_last: bool = False) -> scipy.sparse.coo_matrix:
        """Create a sparse matrix of dummy variables, optionally without the first column."""
        rows = self.total_count
        columns = self.group_count
        row_indices = np.arange(self.total_count)
        column_indices = self.codes
        if drop_last:
            columns -= 1
            keep = column_indices < columns
            row_indices = row_indices[keep]
            column_indices = column_indices[keep]
        return scipy.sparse.coo_matrix((np.ones_like(column_indices), (row_indices, column_indices)), (rows, columns))

    def min(self, matrix: Array) -> Array:
        """Compute the minimum of each group."""
        return np.minimum.reduceat(matrix[self.sort_indices], self.reduce_indices)

    def max(self, matrix: Array) -> Array:
        """Compute the maximum of each group."""
        return np.maximum.reduceat(matrix[self.sort_indices], self.reduce_indices)

    def sum(self, matrix: Array) -> Array:
        """Compute the sum of each group."""
        return np.add.reduceat(matrix[self.sort_indices], self.reduce_indices)

    def mean(self, matrix: Array) -> Array:
        """Compute the mean of each group."""
        return self.sum(matrix) / self.counts[:, None]

    def expand(self, statistics: Array) -> Array:
        """Expand statistics for each group to the size of the original matrix."""
        return statistics[self.codes]


def identify_singletons(groups_list: Sequence[Groups]) -> Array:
    """Identify indices of singleton groups."""
    indices = np.logical_or.reduce([g.expand(g.counts < 2) for g in groups_list])
    if len(groups_list) > 1 and indices.any():
        while True:
            last_indices = indices.copy()
            indices = np.logical_or.reduce([g.expand(g.counts - g.sum(indices) < 2) for g in groups_list])
            if (last_indices == indices).all():
                break
    return indices


def max_norm_convergence(last_matrix: Array, matrix: Array, tol: float) -> bool:
    """Check for max norm convergence."""
    return np.abs(last_matrix - matrix).max() < tol
