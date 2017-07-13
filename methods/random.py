from .bo import BO


class Random(BO):
    '''
    Random strategy
    '''
    def __init__(self, options):
        super(Random, self).__init__(options)

    def acq_fun_optimizer(self, m):
        return self.random_sample(self.bounds, self.batch_size)
