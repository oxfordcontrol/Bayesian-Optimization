from .bo import BO
import GPyOpt
import numpy as np
import copy


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

    def bayesian_optimization(self, seed):
        # Set seed
        np.random.seed(seed)

        objective = copy.copy(self.options['objective'])
        X0 = self.random_sample(self.bounds, self.initial_size)
        y0 = objective.f(X0)

        domain = list()
        for i in range(self.bounds.shape[0]):
            domain.append({'name': 'var_' + str(i + 1), 'type': 'continuous',
                          'domain': self.bounds[i]})

        model = GPyOpt.models.\
            gpmodel.GPModel(kernel=self.options['kernel'].copy(),
                            noise_var=self.options['noise'],
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

        return np.asarray(bo.X), np.asarray(bo.Y)

    def acq_fun_optimizer(self, m):
        '''
        This function is not used, as the optimization is done inside GPyOpt.
        It is implemented in order for the class not to be abstract.
        '''
        pass
