import numpy as np
from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import time


class svm_mnist:
    def __init__(self):
        self.bounds = np.array(
            [[-1, 2],   # log(C)
             [-3, 1],   # log(gamma)
             ]
        )
        mnist = fetch_mldata('MNIST original', data_home='./')

        x_data = mnist.data/255.0
        y_data = mnist.target

        # Split data to train and test and validation set
        x_tmp, x_test, y_tmp, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.1,
                                                        random_state=22)
        x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp,
                                                          test_size=0.1,
                                                          random_state=33)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def f(self, X):
        Y = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            C = np.power(10, X[i, 0])
            gamma = np.power(10, X[i, 1])

            start = time.time()
            print('Training with C:', C, ' Gamma:', gamma)
            classifier = svm.SVC(C=C, gamma=gamma)
            classifier.fit(self.x_train, self.y_train)

            Y[i, 0] = 1 - metrics.accuracy_score(
                self.y_val, classifier.predict(self.x_val)
            )
            Y[i, 1] = 1 - metrics.accuracy_score(
                self.y_test, classifier.predict(self.x_test)
            )
            print('Done. Time:', time.time() - start, 'Y:', Y[i, :])
