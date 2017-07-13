# Bayesian-Optimization
A comparison framework for different acquisition functions. This is the code used to evaluate the performance of the new acqusition function, Optimistic Expected Improvement, against the state of the art in the paper
Distributionally Robust Optimization Techniques in Batch Bayesian Optimization by Nikitas Rontsis, Michael A.  Osborne, Paul J. Goulart.

## Installation
This package was written in python 3.7 and uses the packages listed in `installation.txt`.

## Folders organization
The `methods` folder contains most of the source code that implements the following acquisition functions (see the paper for references and more detailed descriptions)
* Multipoint expected improvement (`qei.py`)
* Optimistic expected improvement (`oei.py`)
* Multipoint expected improvement, Constant Liar Heuristic (`qei_cl.py`)
* Batch Lower Confidence Bound (`blcb.py`)
* Local Penalization of Expected Improvement (`lp_ei.py`)
* Random Expected Improvement (`random_ei.py`)

Each of these is based on the parent class `BO` (`bo.py`) that implements a simple parametrizable Bayesian Optimization setup. Finally, the class `Caller` allows to perform multiple Bayesian Optimization experiments and store/plot the results.

The `results` folder is where the `Caller` saves the results of the experiments, while the `test_functions` folder defines synthetic functions that are used as benchmarks for the algorithms.

The functionality of `gp_comparisons.py`, `synthetic_experiments.py` and `plot_experiments.py` is described below.

## Reproduce the results of the papers
Calling `gp_comparisons.py` produces Figure 1 and prints the results of Table 2. `synthetic_experiments.py` runs the experiments of Figure 2. Calling `plot_experiments.py` creates then the 4 subplots of Figure 2  (saved in `results/pdfs`)
```
python plot_experiments.py results/branin_oEI*.dat results/branin_qEI*.dat results/branin_BLCB*.dat
python plot_experiments.py results/cosines_oEI*.dat results/cosines_qEI*.dat results/cosines_BLCB*.dat
python plot_experiments.py results/sixhumpcamel_oEI*.dat results/sixhumpcamel_qEI*.dat results/sixhumpcamel_BLCB*.dat
python plot_experiments.py results/alpine1_oEI*.dat results/alpine1_qEI*.dat results/alpine1_BLCB*.dat
```

## Multipoint Expected Improvement Accuracy
The `R` script `methods/qEI_problem.R` demonstrates cases where the accuracy of `qEI` is arbitrarily wrong (remark of last paragraph of section 3 of the paper)
