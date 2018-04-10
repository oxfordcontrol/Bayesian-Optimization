# Bayesian-Optimization
This is the implementation of a new acquisition function for **Batch Bayesian Optimization**, named **Optimistic Expected Improvement (OEI)**. For details, results and theoretical analysis, refer to the paper titled *Distributionally Ambiguous Optimization Techniques in Batch Bayesian Optimization* by Nikitas Rontsis, Michael A.  Osborne, Paul J. Goulart.

This is a **cleaned** and **updated** version of the [ICML](https://github.com/oxfordcontrol/Bayesian-Optimization/tree/ICML) branch which includes code for testing against the following batch acquisition functions:
* Multipoint expected improvement (`qei.py`)
* Multipoint expected improvement, Constant Liar Heuristic (`qei_cl.py`)
* Batch Lower Confidence Bound (`blcb.py`)
* Local Penalization of Expected Improvement (`lp_ei.py`)

Refer to the above mentioned paper for references and more detailed descriptions.

***Please use this branch for `OEI` as the one in the [ICML](https://github.com/oxfordcontrol/Bayesian-Optimization/tree/ICML) branch is outdated.***

## Installation
This package was written in `Python 3.6`. It is recommended to use the `Anaconda >=5.0.1` distribution, on a empty environment. The package is built on top of `gpflow 0.5` which has to be installed from [source]( https://github.com/GPflow/GPflow/releases/tag/0.5.0).
Then, proceed on installing the following
```
conda install numpy scipy matplotlib pyyaml
```
You should also install [`SCS`](https://github.com/cvxgrp/scs), a Convex Solver. Do compile the package (`--no-binary :all:` flag below), as this can bring a significant speedup.
```
pip install 'scs>=2.0.2' --no-binary :all:
```
Moreover, you should install a Python interface to the `Intel MKL Pardiso` Linear Solver (freely distributed by the `Anaconda` distribution):
```
conda install -c haasad pypardiso
```
Finally, you have to download the second order non-linear solver [`KNITRO`](https://www.artelys.com/en/optimization-tools/knitro), install it and copy its contents in a folder named `knitro` inside this package. `KNITRO` is proprietary, but time-limited academic and commercial licenses are provided for free by `Artelys`.

## Testing
Unit tests are included in the `tests` folder. You can invoke them using `pytest`. To do so, the following additional packages are required:
```
conda install pytest
pip install 'cvxpy<1' numdifftools
conda install -c mosek mosek
```

## Usage example
Running Batch Bayesian Optimization (BO) for the Hartmann-6d function with the `OEI` acquisition function can be invoked as following:
```
python run.py --seed=123 --algorithm=OEI --function=hart6 --batch_size=20 --initial_size=10 --iterations=15 --noise=1e-6
```
The above will create an output saved in the folder named `out`. Detailed logging is saved in the `log` folder, the verbosity of which can be controller by the `logging.yaml` configuration file.

To compare against random search and plot the results run:
```
python run.py --seed=123 --algorithm=Random --function=hart6 --batch_size=20 --initial_size=10 --iterations=15 --noise=1e-6 --nl_solver=knitro --hessian=1

python plot.py hart6 out/hart6_OEI out/hart6_Random 
```
This will save a `pdf` plot in the `results` folder.

The user can define a different function for optimization by modifying `benchmark_function.py`, or run **any of the ones presented in the paper**, the definitions of which can be found [here](TODO:add_link).

## Basic files

|  File            | Description                                                      |
|----------------|-----------------------------------------------------------------|
| `run.py`         | Script that runs BO on test functions defined in `benchmark_functions.py`.               |
| `plot.py`        | Plots results of `run.py`.                                       |
| `methods/oei.py` | Definition of `OEI`, its gradient and hessian.                   |
| `methods/bo.py`  | A parent class that implements a simple parametrizable BO setup. |

## Timing results
The main operations for the calculating value of `OEI`, its gradient and hessian are listed below:
* Value and gradient
    * Solving the Semidefinite Optimization Problem [TODO:Link code]().
* Hessian: 
    * Calculating the derivatives of the M [TODO:Link code]().
    * Chain rules of tensorflow [TODO:Link code]().

In the current implementation a considerable amount of time is spent in Python to create, reshape and move matrices, all of which could have been avoided in a more efficient version.