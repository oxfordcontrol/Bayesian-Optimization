from __future__ import print_function
import numpy as np
from . import BO, QEI_CL


class Random_EI(QEI_CL):
    '''
    Naive batch strategy where the first point is the minimizer of the
    one-point expected improvement and the rest are chosen uniformly at random
    '''
    def __init__(self, options):
        super(Random_EI, self).__init__(options)

    def get_suggestion(self, batch_size):
        # First point in the batch is taken by maximing one-point EI
        # The acquisition function inherited by QEI_CL is the one-point EI
        # and we use the standard BO's get_suggestion for only one point
        X0 = BO.get_suggestion(self, batch_size=1)
        # The rest points are chosen at random
        X_random = self.random_sample(self.bounds, batch_size-1)

        return np.concatenate((X0, X_random))
