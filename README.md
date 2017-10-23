# Bayesian-Optimization
A comparison framework for different acquisition functions. This is the code used to evaluate the performance of the new acquisition function, Optimistic Expected Improvement, against the state of the art in the [paper](https://arxiv.org/abs/1707.04191) Distributionally Robust Optimization Techniques in Batch Bayesian Optimization by Nikitas Rontsis, Michael A.  Osborne, Paul J. Goulart.

## Installation
This package was written in python 3.6 and uses the packages listed in `installation.txt`.

## Folders organization
The `methods` folder contains most of the source code. It implements the following acquisition functions (see the [paper](https://arxiv.org/abs/1707.04191) for references and more detailed descriptions)
* Optimistic expected improvement (`oei.py`) (**our novel algorithm**)
* Multipoint expected improvement (`qei.py`)
* Multipoint expected improvement, Constant Liar Heuristic (`qei_cl.py`)
* Batch Lower Confidence Bound (`blcb.py`)
* Local Penalization of Expected Improvement (`lp_ei.py`)
* Random Expected Improvement (`random_ei.py`)

Each of these is based on the parent class `BO` (`bo.py`) that implements a simple parametrizable Bayesian Optimization setup. Finally, the class `Caller` allows to perform multiple Bayesian Optimization experiments and store/plot the results.

The `results` folder is where `Caller` saves the results of the experiments, while the `test_functions` folder defines synthetic functions that are used as benchmarks for the algorithms.

The functionality of `gp_comparisons.py`, `synthetic_experiments.py` and `plot_experiments.py` is described below.

## Reproduce the results of the papers
The results of the [paper](https://arxiv.org/abs/1707.04191) can be reproduced by running the script `run_me.sh`. This script calls internally `gp_comparisons.py` `synthetic_experiments.py` and `plot_experiments.py`.

In particular, calling `gp_comparisons.py` produces Figure 1 and prints the results of Table 2. `synthetic_experiments.py` runs the experiments of Figure 2. Calling `plot_experiments.py` creates then the 4 subplots of Figure 2  (saved in `results/pdfs`)
```
python plot_experiments.py results/branin_oEI*.dat results/branin_qEI*.dat results/branin_BLCB*.dat
python plot_experiments.py results/cosines_oEI*.dat results/cosines_qEI*.dat results/cosines_BLCB*.dat
python plot_experiments.py results/sixhumpcamel_oEI*.dat results/sixhumpcamel_qEI*.dat results/sixhumpcamel_BLCB*.dat
python plot_experiments.py results/alpine1_oEI*.dat results/alpine1_qEI*.dat results/alpine1_BLCB*.dat
```

## Timing results
To compare the timing of OEI as compared to QEI run 
```
python timings.py
```
This compares the average time computing time for OEI and QEI (and their gradients) when performing Bayesian Optimization on a standard 5d optimization function (alpine1).

## Multipoint Expected Improvement Accuracy
The `R` script `methods/qEI_problem.R` demonstrates cases where the accuracy of `qEI` is arbitrarily wrong (remark of last paragraph of section 3 of the [paper](https://arxiv.org/abs/1707.04191)).
