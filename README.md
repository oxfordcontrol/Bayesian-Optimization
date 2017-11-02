# Bayesian-Optimization
A comparison framework for different batch acquisition functions. This is the code used to evaluate the performance of the new batch acquisition function, Optimistic Expected Improvement, against the state of the art in the [paper](https://arxiv.org/abs/1707.04191) *Distributionally Ambiguous Optimization Techniques in Batch Bayesian Optimization* by Nikitas Rontsis, Michael A.  Osborne, Paul J. Goulart.

## Installation
This package was written in `Python 3.6` and uses the packages listed in `installation.txt`.

## Folders organization
The `methods` folder contains most of the source code. It implements the following acquisition functions (see the [paper](https://arxiv.org/abs/1707.04191) for references and more detailed descriptions)
* Optimistic expected improvement (`oei.py`) (**our novel algorithm**)
* Multipoint expected improvement (`qei.py`)
* Multipoint expected improvement, Constant Liar Heuristic (`qei_cl.py`)
* Batch Lower Confidence Bound (`blcb.py`)
* Local Penalization of Expected Improvement (`lp_ei.py`)

Each of these is based on the parent class `BO` (`bo.py`) that implements a simple parametrizable Bayesian Optimization setup.

The `out` folder is where the output of each run is saved, while the `results` folder is where the final figures are produced. The `test_functions` folder defines synthetic functions that are used as benchmarks for the algorithms.

## Reproduce the results of the papers
The results of the [paper](https://arxiv.org/abs/1707.04191) can be reproduced by running the script `run_me.sh`, which saves a number of figures in the `results` folder. This script calls internally `gp_posteriors.py`, `plot_ei_vs_batch.py`, `run.py` and `plot_experiments.py`.

## Timing results
To compare the timing of `OEI` as compared to `QEI` refer to the [GPy implementation](https://github.com/oxfordcontrol/Bayesian-Optimization/tree/GPy-based), as this avoids the overhead associated with `TensorFlow`.