from __future__ import print_function
from .bo import BO
import GPyOpt
import numpy as np
import tensorflow as tf
import random
import copy
import logging
import os
import yaml


class LP_EI(BO):
    def __init__(self, options):
        '''
        This class is a wrapper for the EI-based Local Penalization acquisition
        function which is implemented in the GPyOpt package.

        Minor adaptations to GPyOpt were necessary to
        ensure that GPyOpt uses the exact same choices as the rest of
        algorithms in this code (e.g. kernel, number of optimization restarts
        in the model and the acquisition function etc.)

        For this reason the following modified version should be installed:
        ------------------------------------------------
        git clone https://github.com/nrontsis/GPyOpt.git
        cd GPyOpt
        python setup.py develop
        ------------------------------------------------

        See:
        Batch Bayesian Optimization via Local Penalization
        https://arxiv.org/abs/1505.08052
        '''
        super(LP_EI, self).__init__(options)

    def bayesian_optimization(self):
        objective = copy.copy(self.options['objective'])
        X0 = self.random_sample(self.bounds, self.initial_size)
        X0 = np.concatenate((X0, X0[0:self.options['init_replicates']]))
        ret = objective.f(X0)
        if isinstance(ret, tuple):
            y0, X0 = ret
        else:
            y0 = ret

        domain = list()
        for i in range(self.bounds.shape[0]):
            domain.append({'name': 'var_' + str(i + 1), 'type': 'continuous',
                          'domain': self.bounds[i]})

        if self.options['noise'] is None:
            exact_feval = False
        else:
            exact_feval = True
        
        model = GPyOpt.models.\
            gpmodel.GPModel(kernel=self.options['kernel'].copy(),
                            noise_var=self.options['noise'],
                            exact_feval=exact_feval,
                            optimize_restarts=self.options['model_restarts'],
                            normalize_Y=self.options['normalize_Y'],
                            verbose=False,
                            mean_function=self.options['mean_function'])

        space = GPyOpt.Design_space(domain, None)
        acquisition = GPyOpt.optimization.\
            ContAcqOptimizer(space=space, optimizer='lbfgs',
                             n_samples=self.options['opt_restarts'],
                             fast=False, random=True,
                             search=True)

        bo = GPyOpt.methods.BayesianOptimization(
            f=objective.f,
            normalize_Y=self.options['normalize_Y'],  # in ContAcqOptimizer
            noise_var=self.options['noise'],
            exact_feval=exact_feval,
            domain=domain,
            X=X0,
            Y=y0,
            initial_design_numdata=self.initial_size,
            initial_design_type='random',
            acquisition_type='EI',
            evaluator_type='local_penalization',
            batch_size=self.batch_size,
            acquisition_jitter=0,
            model=model,
            acquisition_optimizer=acquisition
            )

        # Run the optimization
        bo.run_optimization(max_iter=self.iterations)

        # Logging
        if 'seed' in self.options:
            self.log_folder = 'log/' + self.options['job_name'] + '/' + \
                str(self.options['seed']) + '/'
        else:
            self.log_folder = 'log/' + self.options['job_name'] + '/'
        try:
            os.makedirs(self.log_folder)
        except OSError:
            pass
        # Load config file
        with open('logging.yaml', 'r') as f:
            config = f.read()
        # Prepend logging folder
        config = config.replace('PATH/', self.log_folder)
        # Setup logging
        logging.config.dictConfig(yaml.load(config))
        logger = logging.getLogger('evals')
        logger.info('----------------------------')
        logger.info('Bounds:\n' + str(self.bounds))
        if hasattr(objective, 'fmin'):
            logger.info('Minimum value:' + str(objective.fmin))
        logger.info('----------------------------')
        X_ = np.asarray(bo.X).copy()
        Y_ = np.asarray(bo.Y).copy()

        for i in range(self.initial_size):
            logging.getLogger('evals').info(
                'X:' + str(X_[i, :]) + ' y: ' + str(Y_[i, :])
            )
        for i in range(self.iterations):
            logging.getLogger('').info(
                '#Iteration:' + str(i + 1)
            )
            for j in range(self.batch_size):
                index = self.initial_size + i*self.batch_size + j
                logging.getLogger('evals').info(
                    'X:' + str(X_[index, :]) + ' y: ' + str(Y_[index, :])
                )

        return np.asarray(bo.X), np.asarray(bo.Y)

    def acq_fun_optimizer(self, m):
        '''
        This function is not used, as the optimization is done inside GPyOpt.
        It is implemented in order for the class not to be abstract.
        '''
        pass
