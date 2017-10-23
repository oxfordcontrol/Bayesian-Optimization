from .bo import BO


class Random(BO):
    '''
    Random strategy
    '''
    def __init__(self, options):
        super(Random, self).__init__(options)

    def get_suggestion(self, batch_size):
        return self.random_sample(self.bounds, batch_size)
