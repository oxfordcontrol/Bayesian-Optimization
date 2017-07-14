#!/bin/sh
python synthetic_experiments.py
python plot_experiments.py results/branin_oEI*.dat results/branin_qEI*.dat results/branin_BLCB*.dat
python plot_experiments.py results/cosines_oEI*.dat results/cosines_qEI*.dat results/cosines_BLCB*.dat
python plot_experiments.py results/sixhumpcamel_oEI*.dat results/sixhumpcamel_qEI*.dat results/sixhumpcamel_BLCB*.dat
python plot_experiments.py results/alpine1_oEI*.dat results/alpine1_qEI*.dat results/alpine1_BLCB*.dat

python gp_comparisons.py
