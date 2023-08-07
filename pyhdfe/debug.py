import pyhdfe
import numpy as np

N = 1000
X = np.random.normal(0, 1, 2*N).reshape((N, 2))
g = np.random.choice(["a", "b", "c", "d", "e"], N).reshape((N, 1))
weights = np.random.uniform(0, 1, N).reshape((N, 1))

within = pyhdfe.create(g, residualize_method="dummy").residualize(X, weights = weights)
