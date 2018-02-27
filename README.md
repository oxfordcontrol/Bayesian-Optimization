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

The `out` folder is where the output of each run is saved, while the `results` folder is where the final figures are produced. The `test_functions` folder defines synthetic & real world functions that are used as benchmarks for the algorithms.

## Running BO on test functions
Invoke as the following example:
```
python run.py --algorithm=OEI --function=hart6 --batch_size=10 --initial_size=10 --iterations=20 --noise=1e-6
```

## Timing results
To compare the timing of `OEI` as compared to `QEI` refer to the [GPy implementation](https://github.com/oxfordcontrol/Bayesian-Optimization/tree/GPy-based), as this avoids the overhead associated with `TensorFlow`.
