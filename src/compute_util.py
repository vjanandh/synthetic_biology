import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
import copy

from onestrainMOO import ONE_STRAIN_UTILS

class Compute():
    CellY0 = None
    params = None
    pool = None
    seed = 42

    def __init__(self, **kwargs):
        self.mparams = None
        self.model = None
        self.bestX = None

        for key, val in kwargs.items():
            self.__dict__[key] = val
        
        if any(var is None for var in (self.model, self.params)):
            print("Compute class not properly initialised", flush=True)
            return None

        if self.CellY0 is not None:
            Compute.CellY0 = self.CellY0
        
        if Compute.CellY0 is None:
            print("Simulating one strain steady state")
            sol = self.solve_steady_state()
            if sol is None:
                print("Simulating one strain steady state failed", flush=True)
                return None

        self.Y0, self.hPR, self.cPR = self.model["init_arrays"](self.params, self.mparams, self.CellY0)

    def solve_steady_state(self):
        """
        Solve the steady state conditions for one strain model
        """
        ode_term = ONE_STRAIN_UTILS["ode_scipy"]
        Y0, hPR, cPR = ONE_STRAIN_UTILS["init_arrays"](self.params, self.mparams, None)
   
        # Solve the ODE system
        try:
            tspan = (0, self.params.runintmax)
            sol_SS = solve_ivp(ode_term, tspan, Y0, args=(hPR, cPR), method="Radau")
            Y10_SS = np.array(sol_SS.y.T[-1])
            Y10_SS = np.maximum(Y10_SS, 0)
            Compute.CellY0 = Y10_SS.flatten()[4:20]
        except:
            sol_SS = None
        return sol_SS
    
    def solve_odes(self, ode_term=None, tspan=(0, 1e6), Y0=None, args=None, method="Radau"):
        """
        Solves the ODE system using scipy.integrate.solve_ivp. Takes in the ODE function, a tuple specifying the time span,
        initial condition Y0, a tuple of additional arguments to the ODE function, solver method to use.
        
        Returns: a solve_ivp solution object containing the solutions at several time points across the time span among other fields.
        """
        if any (arg is None for arg in [ode_term, Y0, args]):
            print("Insufficient arguments to solve the ODE system", flush=True)
            return None
        
        # Solve the ODE system
        try:
            sol = solve_ivp(ode_term, tspan, Y0, args=args, method=method)
        except:
            sol = None
        return sol
    
    def update_arrays(self, sol):
        """
        Updates the initial condition Y0 using the solution object obtained from solving the ODE system.
        """
        Y_ON = np.array(sol.y.T[-1])
        self.Y0[:] = Y_ON.flatten()
        self.Y0, self.hPR, self.cPR = self.model["update_arrays"](self.params, self.Y0, self.hPR, self.cPR, self.bestX)

    def run_model(self):
        """
        Simulates a specific model for a set time span.
        
        Returns: a solve_ivp solution object containing the solutions at several time points across the time span among other fields.
        """
        print(f"Simulate {self.model["name"]} for {self.params.tmax} minutes")

        try:
            self.Y0, self.hPR, self.cPR = self.model["update_arrays"](self.params, self.Y0, self.hPR, self.cPR, self.bestX)
            sol = solve_ivp(self.model["ode_scipy"], (0, self.params.tmax), self.Y0, args=(self.hPR, self.cPR), method='BDF')
            self.update_arrays(sol)
            sol = solve_ivp(self.model["ode_scipy"], (0, self.params.tmax), self.Y0, args=(self.hPR, self.cPR), method='BDF')
            self.update_arrays(sol)
        except:
            sol = None
        return sol

    def run_MOO(self, algorithms, terminations):
        """
        Runs multi-objective optimisation
        
        Returns: a pymoo solution object containing objective values and design parameters values among other fields.
        """
        lb, ub = self.model["var_bounds"]
        nvars = len(lb)
        runner =  StarmapParallelization(self.pool.starmap)

        print(f"Multi-objective optimisation - {self.model["name"]} model", flush=True)
        # Single objective optimisation
        SF = np.array([1., 0.])
        args = (self.hPR, self.cPR, self.Y0, self.params.N0, self.params.tmax, SF)

        problem = self.model["sop"](nvars=nvars, nobjs=1, lb=lb, ub=ub, args=args,
                                  elementwise_runner=runner)

        ## Productivity optimisation
        res = minimize(problem, algorithms["sop"], terminations["sop"], seed=self.seed, verbose=False)
        max_yield = abs(res.F)
        
        ## Yield optimisation
        SF[:] = 0., 1.
        problem = self.model["sop"](nvars=nvars, nobjs=1, lb=lb, ub=ub, args=args,
                                  elementwise_runner=runner)
        res = minimize(problem, algorithms["sop"], terminations["sop"], seed=self.seed, verbose=False)
        max_prod = abs(res.F)

        # Multi-objective optimisation
        SF[:] = max_yield[0], max_prod[0]
        print(f"Max yield and productivity: {SF}", flush=True)
        problem = self.model["mop"](nvars=nvars, nobjs=2, lb=lb, ub=ub, args=args,
                                  elementwise_runner=runner)
        res = minimize(problem, algorithms["mop"], terminations["mop"], seed=self.seed, verbose=False)

        return res, SF
    
    def run_dynamics(self, sol):
        """
        Takes in a solve_ivp solution object and computes the population levels, growth rates and levels of 
        extracellular substances.
        
        Returns: Arrays containing population, growth rate and quantity of extracellular substances.
        """
        T = sol.t.T
        Y = sol.y.T

        pop = []
        growth = []
        em = []

        for i in range(len(T)):
            dyn_vals = self.model["ode"](T[i], Y[i], (self.hPR, self.cPR))
            pop.append(dyn_vals[1])
            growth.append(dyn_vals[2])
            em.append(dyn_vals[3])

        return pop, growth, em
    
    def run_rnd(self, bestX, args, rnd_params, rnd_range=0.5):
        """
        Runs the robustness test. Inputs are the x values, arguments to the objevtive functions, 
        design parameters to be perturbed, level of uncertainty to add to each parameter.
        
        Returns: A list of objective values and the corresponding perturbed design parameter values
        """
        rand = np.random.default_rng(seed=self.seed)
        bestX_rnd = copy.deepcopy(bestX)
        bestX_rnd = np.array(bestX_rnd)
        hPR, cPR, Y0 = args
        objvals = []

        rnds = rand.uniform(-rnd_range, rnd_range, bestX_rnd.shape)
        bestX_rnd[:, rnd_params] *= (1 + rnds[:, rnd_params])
        tsols = bestX_rnd.shape[0]
        size = int(tsols*0.2) if tsols > 40 else tsols
        x_chosen = bestX_rnd[np.random.choice(tsols, size=size, replace=False)]
        for x in x_chosen:
            _, prod_and_yld = self.model["calc_objs"](x, args=(hPR, cPR, Y0, self.params.N0, self.params.tmax, 
                                                            np.array([1, 1])))
            objvals.append(prod_and_yld.tolist())
        vals = [objs + xvals for objs, xvals in zip(objvals, x_chosen.tolist())]
        return vals