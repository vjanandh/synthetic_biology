import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import time
import json
import copy

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination, SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter

from system_parameters import SystemParameters
from onestrainMOO import ONE_STRAIN_UTILS
from twostrainsMOO import TWO_STRAINS_UTILS
from twostrainsXfeedMOO import TWO_STRAINS_XFEED_UTILS
from twostrainsQSBooMOO import TWO_STRAINS_QS_BOO_UTILS
from twostrainsQSDinhMOO import TWO_STRAINS_QS_DINH_UTILS
from compute_util import Compute

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from functools import partial

jax.config.update("jax_enable_x64", True)
KEY = 10
random.seed(KEY)
np.random.seed(KEY)

file_id = 1 # For numbering files
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to store the output files
res_dir = os.path.join(script_dir, "out_runs")
# Path containing the data file for robustness test
data_dir = os.path.join(script_dir, "out_runs_data")

# Models to run
models = {"m1": ONE_STRAIN_UTILS,
          "m2": TWO_STRAINS_UTILS,
          "m3": TWO_STRAINS_XFEED_UTILS,
          "m4": TWO_STRAINS_QS_BOO_UTILS,}
#          "m5": TWO_STRAINS_QS_DINH_UTILS,
#         }

# Colours for each model used in plotting graphs
colours = ['blue', 'red', 'green', 'orange', 'purple']

# MOO termination criterion
termination_gen = get_termination("n_gen", 2)
terminations = {"sop": DefaultSingleObjectiveTermination(n_max_gen=200, period=30),
                "mop": DefaultMultiObjectiveTermination(n_max_gen=300, period=50)
            }

# MOO algorithms
algorithms = {"sop": GA(pop_size=200, n_offsprings=100, sampling=LHS(), seed=KEY),
              "mop": AGEMOEA2(pop_size=200, n_offsprings=300, sampling=LHS(), seed=KEY,
                        crossover=SBX(eta=15),
                        mutation=PolynomialMutation(eta=5))
            }

## Values of kBe, vBe and nB values to try
vBe_arr = np.array([58])  # [58, 580, 5800]
kBe_arr = np.array([1e3]) # [1e3, 1e6, 1e9]
nB_arr = np.array([900])  # [300, 900, 1500]
KB, NB, VB = np.meshgrid(kBe_arr, nB_arr, vBe_arr)

## Best design parameters obtained from MOO
bestX_vals_1S = {
    "lyld" : np.array([5.86426453, 43.70048822, 1.0002147]),
    #"myld" : np.array([5.02087033, 1.90293695, 1.00149636]),
    "hyld" : np.array([12.42483521, 43.33756613, 1.00000038]),
}

bestX_vals_2S = {
    "lyld" : np.array([9.05223555, 321.35108325, 0.54280569,   1.00017842, 1.72184112]),
    #"myld" : np.array([5.09166366e+00, 9.99957846e+02, 5.84391358e-01, 1.00000586e+00, 4.92414204e+01]),
    "hyld" : np.array([1.69506952e+01, 9.99999103e+02, 5.41047505e-01, 1.00002070e+00, 4.99532340e+01]),
}

bestX_vals = bestX_vals_1S

def run_moo(params, pool, models2run):
    # Setup plots for pareto-fronts
    plt.figure()
    
    outputs_dat = []
    
    for i, model in enumerate(models2run):
        # Output data format
        output = {"name": model["name"],
                  "Y0": [],
                  "hPR": [],
                  "cPR": [],
                  "objVals": [],
                  "bestX": [],
                  "maxObjVals": []
                 }
        
        # Create a compute object that holds the model and its associated functions and parameters
        compute_obj = Compute(model=model, params=params, pool=pool, mparams=model["mparams"])
    
        if compute_obj is None:
            return False
        
        # Simulate the model
        sol = compute_obj.run_model()
    
        if sol is not None:
            # Do multi-objective optimisation
            res, SF = compute_obj.run_MOO(algorithms, terminations)
            print(f"Optimisation time for {model["name"]} model: {res.exec_time}", flush=True)
            print(f"Number of solutions = {res.X.shape}", flush=True)
            objVals = (- res.F * SF)
    
            output["Y0"].append(compute_obj.Y0.tolist())
            output["hPR"].append(compute_obj.hPR.tolist())
            output["cPR"].append(compute_obj.cPR.tolist())
            output["objVals"].append(objVals.tolist())
            output["bestX"].append(res.X.tolist())
            output["maxObjVals"].append(SF.tolist())
            outputs_dat.append(output)
            
            # Plot the Pareto fronts
            vals_sorted = objVals[objVals[:,1].argsort()[::-1]]
            plt.plot(np.abs(vals_sorted[:,0]), np.abs(vals_sorted[:,1]), 
                     marker='o', color=colours[i], label=model["name"])
        else:
            return False
    
    img_file = os.path.join(res_dir, f"fig_{file_id:03}_r{KEY}_{params.nB}_{params.vBe}_{int(params.kBe)}.png")
    dat_file = os.path.join(res_dir, f"dat_{file_id:03}_r{KEY}_{params.nB}_{params.vBe}_{int(params.kBe)}.json")
    print(f"Img file: {img_file}")
    print(f"Dat file: {dat_file}", flush=True)

    plt.title(f"vBe={params.vBe}, nB={params.nB}, kBe={params.kBe}, tmax={params.tmax}")
    plt.legend()
    
    # Save the graph and data
    plt.savefig(img_file)
    plt.close()

    with open(dat_file, "a") as file:
        json.dump(outputs_dat, file, indent=4)
        
    return True

def run_dyn(params, pool, models2run, bestX=None):
    # Setup plots for dynamics
    num_models = len(models2run)

    for yloc, xvals in bestX.items():
        fig, axs = plt.subplots(num_models, 3, figsize=(15, num_models*5 - 0.5))
        axs = axs.flatten()

        for i, model in enumerate(models2run):
            # Create a compute object that holds the model and its associated functions and parameters
            compute_obj = Compute(model=model, params=params, pool=pool, mparams=model["mparams"], bestX=xvals)
            if compute_obj is None:
                return False

            sol = compute_obj.run_model()
            
            if sol is not None:
                T = sol.t.T
                pop, growth, em = compute_obj.run_dynamics(sol)
                norm_em = np.array(em)/np.max(em, axis=0)
                # Plot
                axs[i*3].plot(T, pop, label='N' if model["short_name"] == "1S" else ['N1', 'N2'])
                axs[i*3].legend()
                axs[i*3 + 1].plot(T, growth, label='lambda')
                axs[i*3 + 1].legend()
                axs[i*3 + 2].plot(T, norm_em[:, :3], label=['xS', 'xI', 'xP'])
                axs[i*3 + 2].legend()
            else:
                return False
          
        img_file = os.path.join(res_dir, f"dyn_{file_id:03}_{model["short_name"]}_{yloc}_{params.nB}_{params.vBe}_{int(params.kBe)}.png")
        fig.suptitle(f"model={model["name"]}, vBe={params.vBe}, kBe={params.kBe}, nB={params.nB}")
        plt.tight_layout()
        # Save the plot
        plt.savefig(img_file)
        plt.close()
        print(img_file, flush=True)
    
    return True

def compute_objectives(_, rnd_func, bestX, args, rnd_range, rnd_idx, params, x_bounds):
    rand = np.random.default_rng(seed=KEY)
    bestX_rnd = copy.deepcopy(bestX)
    bestX_rnd = np.array(bestX_rnd)
    hPR, cPR, Y0 = args
    objvals = []

    # If there are more than 40 solutions on the Pareto-front, randomly choose 40 solutions
    # for faster computations and to reduce clutter
    tsols = bestX_rnd.shape[0]
    size = int(tsols*0.2) if tsols > 40 else tsols
    x_chosen = bestX_rnd[np.random.choice(tsols, size=size, replace=False)]
    rnds = rand.uniform(-rnd_range, rnd_range, x_chosen.shape)
    x_chosen[:, rnd_idx] *= (1 + rnds[:, rnd_idx])
    
    # Clip the values to ensure they are within the design bounds
    x_chosen = np.clip(x_chosen, x_bounds[0], x_bounds[1])

    for x in x_chosen:
        # Compute the objectives
        _, prod_and_yld = rnd_func(x, args=(hPR, cPR, Y0, params.N0, params.tmax, 
                                                        np.array([1, 1])))
        objvals.append(prod_and_yld.tolist())
    vals = [objs + xvals for objs, xvals in zip(objvals, x_chosen.tolist())]
    return vals

def run_rnd(params, pool, models2run, fname):
    params.tmax *= 6
    data_file = os.path.join(data_dir, fname)
    n_samples = 1000
    rnd_range = 0.1
    plt.figure()
    rnd_file = os.path.join(res_dir, f"rnd_{file_id:03}_r{KEY}_rnd{int(rnd_range*100)}_t{params.tmax}_{params.nB}_{params.vBe}_{int(params.kBe)}.json")
    img_path = os.path.join(res_dir, f"rnd_{file_id:03}_r{KEY}_rnd{int(rnd_range*100)}_{params.nB}_{params.vBe}_{int(params.kBe)}.png")
    
    outputs_dat = []

    with open(data_file, 'r') as file:
        data = json.load(file)
    
    for i, model in enumerate(models2run):
        # Output data format
        output = {"name": model["name"],
                  "objVals": [],
                  "xvals": [],
                 }
        print(f"Running robustness test for model {model["name"]}", flush=True)
        mdata = next((d for d in data if d["name"] == model["name"]), None)
        bestX = np.array(mdata["bestX"])
        bestX = bestX.reshape((bestX.shape[-2], bestX.shape[-1]))
        hPR = np.array(mdata["hPR"]).flatten()
        cPR = np.array(mdata["cPR"]).flatten()
        Y0 = np.array(mdata["Y0"]).flatten()
        args=(hPR, cPR, Y0)
        CellY0 = Y0.flatten()[4:20]
        
        compute_obj = Compute(model=model, params=params, mparams=model["mparams"], CellY0=CellY0)
        print("Compute object created", flush=True)
        
        if compute_obj is None:
            print(f"Could not create compute object for model {model["short_name"]}", flush=True)
            return False
        
        if model["short_name"] == "2QSB":
            rnd_idx = [True, True, True, True, True, True, True, True, True, True, True]
        elif model["short_name"] == "2S" or model["short_name"] == "2XF":
            rnd_idx = [True, True, True, True, True]
        else:
            rnd_idx = [True, True, True]
            
        x_bounds = model["var_bounds"]

        # Parallelisation
        partial_compute = partial(compute_objectives, rnd_func=model["calc_objs"], bestX=bestX, args=args, 
                                  rnd_range=rnd_range, rnd_idx=rnd_idx, params=params, x_bounds=x_bounds)
        obj_and_xvals = pool.map(partial_compute, range(n_samples))
        obj_and_xvals = np.array(obj_and_xvals)
        objvals = obj_and_xvals[:, :, :2]
        xvals = obj_and_xvals[:, :, 2:]

        output["objVals"].append(objvals.tolist())
        output["xvals"].append(xvals.tolist())
        outputs_dat.append(output)

        # Plot the best objective values
        fvals = np.array(mdata["objVals"])
        fvals = fvals.reshape((fvals.shape[-2], fvals.shape[-1]))
        fvals_sorted = fvals[fvals[:,1].argsort()[::-1]]
        plt.plot(np.abs(fvals_sorted[:,0]), np.abs(fvals_sorted[:,1]), color="black", marker='o', markerfacecolor='none', alpha=0.2)
        
        # Plot the samples
        for j, vals in enumerate(objvals):
            vals = np.array(vals)
            vals_sorted = vals[vals[:, 1].argsort()[::-1]]
            plt.scatter(np.abs(vals_sorted[:,0]), np.abs(vals_sorted[:,1]), c=colours[i], 
                     label=model["name"] if j==0 else "", alpha=0.2)

    plt.legend()
    plt.title(f"Uncertainty={rnd_range}, samples={n_samples}\nvBe={params.vBe}, nB={params.nB}, kBe={params.kBe}, tmax={params.tmax}")
    
    # Save the plot and data
    plt.savefig(img_path)
    plt.close()

    with open(rnd_file, "a") as file:
        json.dump(outputs_dat, file, indent=4)
    
    return True

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ncpus', type=int, default=1, help="Number of cpus")
    parser.add_argument('--model', type=str, default="all", help="Run model")
    parser.add_argument('--params', type=int, default=1, help="Parameter set")
    parser.add_argument('--cpu', type=str, default='cpu', help="cpu or hpc")
    parser.add_argument('--fname', type=str, default="", help="Data file name")

    opt_group = parser.add_mutually_exclusive_group(required=True)
    opt_group.add_argument('--moo', action='store_true', help="Optimise")
    opt_group.add_argument('--dyn', action='store_true', help="Get dynamics")
    opt_group.add_argument('--rnd', action='store_true', help="Robustness test")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    run_options = {"moo": args.moo,
                   "dyn": args.dyn,
                   "rnd": args.rnd}
    run_selected = next((key for key, val in run_options.items() if val), None)

    if os.path.exists(res_dir):
        if run_selected == "moo":
            ids = [int(item.split('_')[1])
                for item in os.listdir(res_dir) if item.startswith("dat_")]
            file_id = (max(ids) + 1) if ids else 1
        elif run_selected == "dyn":
            ids = [int(item.split('_')[1])
                for item in os.listdir(res_dir) if item.startswith("dyn_")]
            file_id = (max(ids) + 1) if ids else 1
        else:
            ids = [int(item.split('_')[1])
                for item in os.listdir(res_dir) if item.startswith("rnd_")]
            file_id = (max(ids) + 1) if ids else 1
    else:
        try:
            os.mkdir(res_dir)
            file_id = 1
        except:
            print("Unable to create a directory to store results", flush=True)

    if args.params == 1:
        params = SystemParameters()
    else:
        params = SystemParameters(phie0=100, vIxS=72, wA=5, kappaG=3e8, tmax=24*60)

    # Setup parallelisation
    ncpus = args.ncpus
    if args.cpu == "cpu":
        pool = ThreadPool(ncpus)
    else:
        pool = Pool(ncpus)
    
    models2run = []

    if args.model == "all":
        for vals in models.values():
            models2run.append(vals)
    elif args.model in models.keys():
        models2run.append(models[args.model])
    else:
        print(f"Error: Model name should be either all or one of these {models.keys()}", 
              flush=True)
        pool.close()
        sys.exit()

    print(f"Program called with args: model={args.model}, num_cpus={args.ncpus}, params={args.params}, run={run_selected}",
          flush=True)
    
    start_time = time.time()
    # For each combination of kBe, nB, vBE values, run the simulations
    for kBe, nB, vBe in zip(KB.ravel(), NB.ravel(), VB.ravel()):
        params.kBe = kBe
        params.nB = nB
        params.vBe = vBe
        
        if args.moo:
            ret = run_moo(params, pool, models2run)
            if not ret:
                print(f"Multi-objective optimisation failed for case {nB}_{int(kBe)}_{vBe}", flush=True)
        elif args.dyn:
            ret = run_dyn(params, pool, models2run, bestX_vals)
            if not ret:
                print(f"Getting dynamics failed for case {nB}_{int(kBe)}_{vBe}", flush=True)
        elif args.rnd:
            if args.fname == "":
                print(f"Data file required for running robustness test. Use option fname", flush=True)
            else:
                ret = run_rnd(params, pool, models2run, args.fname)
                if not ret:
                    print(f"Robustness test failed for case {nB}_{int(kBe)}_{vBe}", flush=True)

    print("--- Loop time: %s seconds ---" % (time.time() - start_time), flush=True)
    pool.close()
