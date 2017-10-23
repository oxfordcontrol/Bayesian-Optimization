import numpy as np
import pdb
OUTPUT_TYPE = 0

'''
Ipopt wrapper
'''
import ipopt


class Ipopt_grad():
    def __init__(self, bo):
        self.bo = bo
        self.dim = bo.dim

    def objective(self, x):
        return self.bo.acquisition(x)[0]

    def gradient(self, x):
        return self.bo.acquisition(x)[1]


class Ipopt_hess(Ipopt_grad):
    def __init__(self, bo):
        super(Ipopt_hess, self).__init__(bo=bo)

    def hessian(self, x, lagrange, obj_factor):
        hessian = self.bo.acquisition(x)[2]
        return obj_factor * hessian[self.hessianstructure()]

    def hessianstructure(self):
        '''
        The structure of the Hessian
        Important Note:
        The default hessian structure is of a lower triangular matrix, but I think there is a bug with the default setting.
        '''
        return np.tril_indices(self.dim)

def ipopt_solve(x_init, bounds, hessian, bo):
    if hessian:
        interface = Ipopt_hess
    else:
        interface = Ipopt_grad

    problem_obj = interface(bo=bo)

    nlp = ipopt.problem(
        n=bounds.shape[0],
        m=0,
        problem_obj=problem_obj,
        lb=bounds[:, 0],
        ub=bounds[:, 1])

    x, info = nlp.solve(x_init)
    obj = info['obj_val']
    return x, obj, info

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

'''
Knitro wrapper

import sys
sys.path.insert(0, '../../knitro/examples/Python')
from knitro import *
from knitroNumPy import *
from collections import namedtuple

See Knitro's callback library reference
https://www.artelys.com/tools/knitro_doc/3_referenceManual/callableLibrary/API.html
The code here is based on the example file
examples/Python/exampleHS15NumPy.py


def callbackEvalFC(bo, evalRequestCode, n, m, nnzJ, nnzH, x,
                   lambda_, obj, c, objGrad, jac, hessian, hessVector,
                   userParams):
    if evalRequestCode == KTR_RC_EVALFC:
        np.copyto(obj, bo.acquisition(x)[0])
        return 0
    else:
        return KTR_RC_CALLBACK_ERR


def callbackEvalGA(bo, evalRequestCode, n, m, nnzJ, nnzH, x,
                   lambda_, obj, c, objGrad, jac, hessian, hessVector,
                   userParams):
    if evalRequestCode == KTR_RC_EVALGA:
        np.copyto(objGrad, bo.acquisition(x)[1])
        # No jacobian matrix
        return 0
    else:
        return KTR_RC_CALLBACK_ERR


def callbackEvalH(bo, evalRequestCode, n, m, nnzJ, nnzH, x,
                  lambda_, obj, c, objGrad, jac, hessian, hessVector,
                  userParams):
    indices = np.triu_indices(n)
    np.copyto(hessian, bo.acquisition(x, output_type=2)[indices])

    if evalRequestCode == KTR_RC_EVALH:
        # In this case we want the full hessian
        return 0
    elif evalRequestCode == KTR_RC_EVALH_NO_F:
        # In this case we only want the constraint part of the lagrangian
        hessian = 0 * hessian
        return 0
    else:
        return KTR_RC_CALLBACK_ERR


def knitro_solve(x_init, bounds, hessian, bo):
    # We have to implement the case of non hessian
    n = bounds.shape[0]
    objGoal = KTR_OBJGOAL_MINIMIZE
    objType = KTR_OBJTYPE_GENERAL
    bndsLo = bounds[:, 0].copy()
    bndsUp = bounds[:, 1].copy()
    # Hessian is dense
    hessRow, hessCol = np.triu_indices(n)
    # No constraints
    m = 0
    cType = np.array([])
    cBndsLo = np.array([])
    cBndsUp = np.array([])
    jacIxConstr = np.array([])
    jacIxVar = np.array([])

    kc = KTR_new()
    assert kc is not None, "Failed to initialize knitro. Check license."

    # Set knitro parameters

    # Derivative Checker
    # assert not KTR_set_int_param_by_name(kc, "derivcheck", 2)

    # Verbosity
    if OUTPUT_TYPE == 0:
        assert not KTR_set_int_param_by_name(kc, "outlev", 0)

    if hessian:
        # Exact hessian
        assert not KTR_set_int_param_by_name(kc, "hessopt", 1)
        assert not KTR_set_int_param_by_name(kc, "hessian_no_f", 1)
    else:
        # 2: BFGS
        assert not KTR_set_int_param_by_name(kc, "hessopt", 2)

    # Compare performance of all the algorithms
    # assert not KTR_set_int_param_by_name(kc, "algorithm", 5)

    # assert not KTR_set_char_param_by_name(kc, "outlev", "all")

    # set callbacks
    assert not KTR_set_func_callback(
        kc, lambda *args: callbackEvalFC(bo, *args))
    assert not KTR_set_grad_callback(
        kc, lambda *args: callbackEvalGA(bo, *args))
    if hessian:
        assert not KTR_set_hess_callback(
            kc, lambda *args: callbackEvalH(bo, *args))

    ret = KTR_init_problem(kc, n, objGoal, objType, bndsLo, bndsUp, cType,
                           cBndsLo, cBndsUp, jacIxVar, jacIxConstr, hessRow,
                           hessCol, x_init, None)
    if ret:
        raise RuntimeError(
            "Error initializing the problem, Knitro status = %d" % ret)

    # These will hold the solutions
    x = np.zeros(n)
    lambda_ = np.zeros(m + n)
    obj = np.array([0])
    nStatus = KTR_solve(kc, x, lambda_, 0, obj, None, None, None, None, None,
                        None)
    nIter = KTR_get_number_iters(kc)
    KTR_free(kc)
    if nStatus != 0:
        print("Knitro failed to solve the problem, final status =", nStatus)

    Result = namedtuple('Result', 'status nit')

    return x, obj, Result(status=nStatus, nit=nIter)
'''

def solve(X_init, bounds, hessian, bo, solver):
    x_init = X_init.flatten()

    if solver == 'ipopt':
        x, y, status = ipopt_solve(x_init=x_init, bounds=bounds,
                                   hessian=hessian, bo=bo)
    elif solver == 'knitro':
        x, y, status = knitro_solve(x_init=x_init, bounds=bounds,
                                    hessian=hessian, bo=bo)
    elif solver == 'scipy':
        x, y, status = scipy_solve(x_init=x_init, bounds=bounds,
                                   hessian=hessian, bo=bo)
    else:
        assert False, 'Invalid nonlinear solver choice!'

    k = x.size // bo.dim
    X = x.reshape(k, bo.dim)
    return X, y, status
