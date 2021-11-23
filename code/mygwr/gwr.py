import numpy as np
import scipy.sparse as sp
from numba import njit

"""
Most of this code is based on pysal/mgwr. This implementation is only for
testing different optimizers on GWR, not for production-grade reuse!
"""

@njit
def local_cdist(coords_i, coords):
    """
    Compute Euclidean distance for a local kernel.
    """
    return np.sqrt(np.sum((coords_i - coords)**2, axis=1))

class Kernel:
    """
    GWR kernel function specifications. Written by Taylor Oshan.
    """

    def __init__(self, i, data, bw=None, fixed=True, function='bisquare', eps=1.0000001):

        self.dvec = local_cdist(data[i], data).reshape(-1)
        self.function = function.lower()

        if fixed:
            self.bandwidth = float(bw)
        else:
            self.bandwidth = np.partition(self.dvec, int(bw) - 1)[int(bw) - 1] * eps  #partial sort in O(n) Time

        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        if self.function == "bisquare":  #Truncate for bisquare
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs**2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs**2)**2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':
            return (1 - (zs)**2)**2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)

class GWR:
    """
    Barebones GWR implementation with swap-out optimizer.
    Only supports minimal functionality (e.g., no intercept support here).
    """

    def __init__(self, coords, y, X, bw, kernel='bisquare', fixed=False):
        self.coords = coords
        self.y = y
        self.X = X
        self.n, self.d = X.shape
        self.bw = bw
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.fitted = False

    def _build_wi(self, i, bw):
        if bw == np.inf:
            return np.ones((self.n))
        return Kernel(i, self.coords, bw, fixed=self.fixed, function=self.kernel).kernel

    def _local_fit(self, i):
        wi = sp.diags(self._build_wi(i, self.bw).reshape(-1))
        xtw = self.X.T @ wi
        betas = np.linalg.solve(xtw @ self.X, xtw @ self.y)
        predy = self.X[i, :] @ betas
        resid = self.y[i] - predy
        hat_matrix_i = (self.X[i, :] @ np.linalg.pinv(xtw @ self.X)).reshape(-1)
        tr_hat_matrix_i = np.sum(hat_matrix_i * hat_matrix_i)
        return betas.reshape(-1), predy, resid, hat_matrix_i, tr_hat_matrix_i

    def loss(self):
        if not self.fitted:
            self.fit()
        k = self.trS
        sigma2 = np.linalg.norm(self.resid, 2)**2/(self.n-k-1)
        return self.n*np.log(sigma2) + self.n*np.log(2*np.pi) + self.n*(self.n + k)/(self.n - k - 2)

    def fit(self, pool=None):
        if pool:
            rslt = pool.map(self._local_fit, range(self.n))
        else:
            rslt = map(self._local_fit, range(self.n))
        
        results = list(zip(*rslt))
        self.params = np.array(results[0])
        self.predy = np.array(results[1])
        self.resid = np.array(results[2]).reshape(-1, 1)
        self.trS = np.array(results[-1]).sum()

        return self
        
        

