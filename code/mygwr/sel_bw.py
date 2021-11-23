import numpy as np
from .gwr import *
from timeit import default_timer as timer

# Optimizer imports
from scipy.optimize import minimize_scalar
from pyswarms.single.global_best import GlobalBestPSO

class Sel_BW:
    """
    Barebones bandwidth selector using inputted optimizer.
    Supports the following optimizers:
        - golden: default golden section search
        - grid: brute force grid search
        - nelder-mead: nelder-mead simplex method
        - pso: particle swarm optimization
        - more??? genetic algorithm, simulated annealing planned
    """
    def __init__(self, coords, y, X, kernel='bisquare', optimizer='golden'):
        self.coords = coords
        self.y = y
        self.X = X
        self.kernel = kernel.lower()
        self.optimizer = optimizer.lower()
        self.objective = lambda bw : GWR(self.coords, self.y, self.X, bw, kernel=self.kernel).loss()

    def select(self, bw_min=None, bw_max=None, pool=None, tol=1e-5,
            n_particles=10, max_iter=1000, full=False, **kwargs):
        """ Dispatcher for the different optimization routines
        n_particles: number of particles for PSO
        bw_min: min bandwidth to check
        bw_max: max bandwidth to check
        pool: pool for parallel processing (or number of processes for PSO)
        tol: convergence tolerance (where applicable)
        max_iter: max number of iterations
        full: report full output (AICc on each iteration)
        """

        if self.optimizer in ['golden', 'grid', 'brent', 'pso']:
            if bw_min is None or bw_max is None:
                raise ValueError('bw_min and bw_max must be defined for this search method')
            if bw_min >= bw_max:
                raise ValueError('bw_min and bw_max are improperly ordered')
            
        if self.optimizer == 'golden':
            return self._golden(bw_min, bw_max, max_iter=max_iter, full=full, tol=tol)
        elif self.optimizer == 'grid':
            return self._grid(bw_min, bw_max, pool=pool, full=full)
        elif self.optimizer == 'brent':
            bounds = (bw_min, bw_max)
            return self._brent(bounds, full=full, tol=tol)
        elif self.optimizer == 'pso':
            bounds = (bw_min, bw_max)
            inertia_wgt = cog_coef = soc_coef = 1
            if 'inertia_wgt' in kwargs:
                inertia_wgt = kwargs['inertia_wgt']
            if 'cog_coef' in kwargs:
                cog_coef = kwargs['cog_coef']
            if 'soc_coef' in kwargs:
                soc_coef = kwargs['soc_coef']

            return self._pso(n_particles, bounds, max_iter=max_iter, tol=tol,
                inertia_wgt=inertia_wgt, cog_coef=cog_coef, soc_coef=soc_coef, full=full)
        elif self.optimizer == 'ga':
            return self._ga()
        elif self.optimizer == 'sa':
            return self._sa()
        else:
            raise RuntimeError('optimizer requested is not implemented')

    def _grid(self, bw_min, bw_max, pool=None, full=False):
        # Perform grid search
        search_range = range(bw_min, bw_max + 1)
        
        if pool:
            t0 = timer()
            rslt = pool.map(self.objective, search_range)
            pool.close()
            pool.join()
            t1 = timer()
        else:
            t0 = timer()
            rslt = map(self.objective, search_range)
            t1 = timer()
        
        results = list(rslt)
        optval = search_range[np.argmin(results)]
        if full:
            return {'min' : optval, 'candidates' : list(search_range), 'losses' : results, 'wall' : t1 - t0}
        else:
            return optval

    def _golden(self, bw_min, bw_max, tol=1e-5, max_iter=1000, full=False):
        # Adapts the implementation from PySAL for custom output here
        delta = 0.38197 # (sqrt(5) - 1)/2, function of phi
        a = bw_min; c = bw_max
        b = bw_min + delta * np.abs(bw_max - bw_min)
        d = bw_max - delta * np.abs(bw_max - bw_min)
        
        opt_score = np.inf
        diff = 1.0e9
        iters = 0
        output = []
        tracker = {}
        t0 = timer()
        while np.abs(diff) > tol and iters < max_iter and a != np.inf:
            iters += 1

            if b in tracker:
                loss_b = tracker[b]
            else:
                loss_b = self.objective(b)
                tracker[b] = loss_b

            if d in tracker:
                loss_d = tracker[d]
            else:
                loss_d = self.objective(d)
                tracker[d] = loss_d

            if loss_b <= loss_d:
                optval = b
                opt_score = loss_b
                c = d
                d = b
                b = a + delta * np.abs(c - a)

            else:
                optval = d
                opt_score = loss_d
                a = b
                b = d
                d = c - delta * np.abs(c - a)

            output.append((optval, opt_score))
            
            optval = np.round(optval, 2)
            if (optval, opt_score) not in output:
                output.append((optval, opt_score))
            
            diff = loss_b - loss_d
            score = opt_score
            
        
        if a == np.inf or bw_max == np.inf:
            score_ols = self.objective(np.inf)
            output.append((np.inf, score_ols))
                
            if score_ols <= opt_score:
                opt_score = score_ols
                optval = np.inf
            
        t1 = timer()

        candidates = [x[0] for x in output]
        losses = [x[1] for x in output]

        if full:
            return {'min' : optval, 'candidates' : candidates, 'losses' : losses, 'iters' : iters, 'wall' : t1 - t0}
        else:
            return optval

    def _brent(self, bounds, tol=1e-5, full=False):       
        t0 = timer() 
        rslts = minimize_scalar(self.objective, method='bounded', bounds=bounds, options={'xatol' : tol})
        t1 = timer()

        if full:
            return {'min' : rslts.x, 'losses' : rslts.fun, 'wall' : t1 - t0}
        else:
            return rslts.x
        
    def _pso(self, n_particles, bounds, inertia_wgt=1, cog_coef=1, soc_coef=1, 
        tol=1e-5, max_iter=100, full=False):
        # Vectorized implementation of particle swarm optimization

        # initialization
        positions = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_particles,))
        losses = list(map(self.objective, positions))
        best_known_pos = positions
        best_known_los = losses
        best_overall = np.argmin(best_known_los)  # index of best particle
        prev_best_los = best_known_los[best_overall]
        velocities = np.random.uniform(low=-abs(bounds[1] - bounds[0]), high=abs(bounds[1] - bounds[0]))
        
        itrctr = 0
        diff = np.inf  # solution improvement
        t0 = timer()
        while diff > tol and itrctr < max_iter: 
            rp = np.random.uniform(size=(n_particles,))
            rg = np.random.uniform(size=(n_particles,))

            # compute velocities and move
            velocities = inertia_wgt*velocities + cog_coef*rp*(best_known_pos - positions) + \
                soc_coef*rg*(np.repeat(positions[best_overall], n_particles) - positions)
            positions += velocities

            # impose boundary constraints (sinks) -- TODO reevaluate this decision
            positions = np.where(positions < bounds[0], bounds[0]*np.ones((n_particles,)), positions)
            positions = np.where(positions > bounds[1], bounds[1]*np.ones((n_particles,)), positions)

            # compute losses and best known losses and positions for particles
            losses = list(map(self.objective, positions))
            best_known_pos = np.where(losses < best_known_los, positions, best_known_pos)
            best_known_los = np.minimum(losses, best_known_los)
            best_overall = np.argmin(best_known_los)

            # update termination thresholds
            diff = np.abs(prev_best_los - best_known_los[best_overall])
            prev_best_los = best_known_los[best_overall]
            itrctr += 1
        t1 = timer()
        
        if full:
            return {'min' : best_known_pos[best_overall], 'loss' : best_known_los[best_overall], 
                    'wall' : t1 - t0, 'iters' : itrctr}
        else:
            return best_known_pos[best_overall]

    def _par_pso(self, n_particles, bounds, inertia_wgt=1, cog_coef=1, soc_coef=1, tol=1e-5, max_iter=100):
        pass

    def _box_pso(self, n_particles, bounds, n_process=None, max_iter=1000):
        # Out-of-the-box PSO -- doesn't quite do the trick bc objective isn't vectorized
        # Set up hyperparameters (to be tuned)
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # TODO vectorize loss function!!
        optimizer = GlobalBestPSO(options=options, n_particles=n_particles, dimensions=1, bounds=bounds)
        rslts = optimizer.optimize(self.objective, iters=max_iter, n_processes=n_process)
        print(rslts)

    def _ga(self):
        raise RuntimeError('unimplemented')

    def _sa(self):
        raise RuntimeError('unimplemented')