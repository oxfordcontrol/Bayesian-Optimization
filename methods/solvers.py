import numpy as np
OUTPUT_TYPE = 0

'''
L-BFGS-B wrapper
'''
import scipy as sp

def scipy_solve(x_init, bounds, hessian, bo):
    res = sp.optimize.minimize(fun=bo.acquisition,
                               x0=x_init,
                               method='L-BFGS-B',
                               jac=True,
                               bounds=bounds,
                               options={'disp': OUTPUT_TYPE}
                               )
    x = res.x
    y = res.fun[0] 

    return x, y, res


def solve(X_init, bounds, hessian, bo, solver):
    x_init = X_init.flatten()

    if solver == 'scipy':
        x, y, status = scipy_solve(x_init=x_init, bounds=bounds,
                                   hessian=hessian, bo=bo)
    else:
        assert False, 'Invalid nonlinear solver choice!'

    k = x.size // bo.dim
    X = x.reshape(k, bo.dim)
    return X, y, status
