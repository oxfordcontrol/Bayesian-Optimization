import GPy
import numpy as np
import os
from methods import QEI, OEI, BLCB, Caller
import GPyOpt.objective_examples


def define_experiments():
    experiments = []

    options = {'batch_size': 5,
               'iterations': 10,
               'gp_opt_restarts': 20,
               'acq_opt_restarts': 20,
               'initial_size': 10,
               'normalize_Y': True,
               'noiseless': True,
               'beta_multiplier': .1,   # for BLCB
               'delta': .1              # for BLCB
               }

    seeds = np.arange(123, 123 + 40)

    '''
    2d functions
    '''
    input_dim = 2
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    options['kernel'] = kernel

    # Branin
    objective = GPyOpt.objective_examples.experiments2d.branin()
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective

    experiments.append(Caller('branin_oEI', OEI(options)))
    experiments.append(Caller('branin_qEI', QEI(options)))
    experiments.append(Caller('branin_BLCB', BLCB(options)))

    # Cosines
    objective = GPyOpt.objective_examples.experiments2d.cosines()
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective

    experiments.append(Caller('cosines_oEI', OEI(options)))
    experiments.append(Caller('cosines_qEI', QEI(options)))
    experiments.append(Caller('cosines_BLCB', BLCB(options)))

    # Six Hump Camel
    objective = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective

    experiments.append(Caller('sixhumpcamel_oEI', OEI(options)))
    experiments.append(Caller('sixhumpcamel_qEI', QEI(options)))
    experiments.append(Caller('sixhumpcamel_BLCB', BLCB(options)))
    '''
    5d functions
    '''
    input_dim = 5
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    options['kernel'] = kernel

    # Alpine 1 function
    objective = GPyOpt.objective_examples.experimentsNd.alpine1(
        input_dim=input_dim)
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective

    options['job_name'] = 'alpine1_oEI'
    experiments.append(Caller('alpine1_oEI', OEI(options)))
    experiments.append(Caller('alpine1_qEI', QEI(options)))
    experiments.append(Caller('alpine1_BLCB', BLCB(options)))

    return experiments, seeds


def run_experiments():
    # This prevents libraries from using all available threads
    os.environ["OMP_NUM_THREADS"] = "1"

    num_threads = 50
    experiments, seeds = define_experiments()

    for i in range(0, len(experiments)):
        print('-------------')
        print(experiments[i].job_name)
        print('-------------')
        experiments[i].run_multiple(seeds=seeds, num_threads=num_threads)
        experiments[i].save_data()


if __name__ == "__main__":
    run_experiments()
