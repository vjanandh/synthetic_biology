# Synthetic biology
This repository contains mathematical models implemented in Python that compare control strategies to maximise biomanufacturing performance from microbial division of labour consortia.

# File organisation
## Main
- run_models.py - script from which the simulations are run. Takes in arguments from the users, calls the utility functions to execute the algorithms, plots and stores the results.

## Utility
- compute_util.py - contains a python class with methods from which the actual ODE solvers and multi-objective optimisation algorithms are executed. This file is written in a model agnostic way so as to maximise code reuse.

## Models
The following files contain the ODE function, multi-objective optimisation problem classes and a few other helper functions for the monoculture and coculture models - one file for each model. 
- onestrainMoo.py - monoculture
- twostrainsMoo.py - coculture without any control strategy
- twostrainsXfeed.py - coculture with cross-feeding control mechanism
- twostrainsQSBoo.py - coculture with quorum sensing mediated burden tuning

# Usage
The main script handles three different processes - multi-objective optimisation, system dynamics and robustness test. The script can be invoked with an option to run one of these processes.

## Command line options
The following are the command line options that can be passed to run_models.py:

| Cmd options | Description                                                                                               | Default | Accepted values             |
| ----------- | --------------------------------------------------------------------------------------------------------- | ------- | --------------------------- |
| --cpu       | where the program is being run, CPU or HPC (cluster)                                                      | cpu     | \[cpu, hpc\]                |
| --fname     | name of the file containing results from multi-objective optimisation. Required for robustness test.      | -       | Filename                    |
| --model     | name of the model to run. Either one or all the models can be run.                                        | all     | \[all, m1, m2, m3, m4,...\] |
| --ncpus     | number of CPUs to use. When cpu is specified for --cpu option, this denotes the number of threads to run. | 1       | An integer                  |
| --params    | parameter set to use for simulations. Currently, there are 2 different system parameter sets.             | 1       | \[1, 2\]                    |
| --moo       | Mutually exclusive options to run one of multi-objective optimisation, system dynamics, robustness test.  | -       | -                           |
| --dyn       | ^                                                                                                         | ^       | ^                           |
| --rnd       | ^                                                                                                         | ^       | ^                           |

# Packages
In addition to standard python packages, the following packages need to be installed:
>	numpy, matplotlib, scipy, jax, equinox, diffrax, optimistix, pymoo

# Examples
1. Running multi-objective optimisation for monocultures on hpc
```
python run_models.py --moo --hpc --npus=24 --model=m1
```
2. Running system dynamics for all models on cpu with parameter set 2
```
python run_models.py --dyn --cpu --model=all --params=2
```
3. Running robustness test for all models
```
python run_models.py --rnd --hpc --ncpus=24 --fname="obj_vals.json"
```
Note: The results of multi-objective optimisation and robustness tests are stored in JSON format. The file to pass to the robustness test must be present in the directory specified by *data_dir* variable in run_models.py.

# Extension
- New models can be added by creating new files containing the ODE function, optimisation functions and classes and other methods which can be referenced from the current files.
- Currently, there is no option to specify to change the reaction kinetics (for example, we change the kinetics of heterologous protein production to simulate different initial conditions). This has to done by editing run_models.py. But new command line options can be added if required.
