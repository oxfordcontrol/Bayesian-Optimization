import numpy as np
import logging
OUTPUT_LEVEL = 0

'''
L-BFGS-B wrapper
'''
import scipy as sp

def bfgs_solve(x_init, bounds, hessian, bo):
    res = sp.optimize.minimize(fun=bo.acquisition,
                               x0=x_init,
                               method='L-BFGS-B',
                               jac=True,
                               bounds=bounds,
                               options={'disp': OUTPUT_LEVEL}
                               )
    x = res.x
    y = res.fun if isinstance(res.fun, float) else res.fun[0]

    return x, y, res

try:
    '''
    Knitro wrapper
    '''
    import sys
    sys.path.insert(0, '../../Libraries/knitro/examples/Python')
    from knitro import *
    from knitroNumPy import *
    from collections import namedtuple
    '''
    See Knitro's callback library reference
    https://www.artelys.com/tools/knitro_doc/3_referenceManual/callableLibrary/API.html
    The code here is based on the example file
    examples/Python/exampleHS15NumPy.py
    '''
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
            return 0
        else:
            return KTR_RC_CALLBACK_ERR

    def callbackEvalH(bo, evalRequestCode, n, m, nnzJ, nnzH, x,
                    lambda_, obj, c, objGrad, jac, hessian, hessVector,
                    userParams):
        indices = np.triu_indices(n)
        np.copyto(hessian, bo.acquisition_hessian(x)[indices])
        if evalRequestCode == KTR_RC_EVALH:
            # In this case we want the full hessian
            return 0
        elif evalRequestCode == KTR_RC_EVALH_NO_F:
            # In this case we only want the constraint part of the lagrangian
            np.copyto(hessian, np.zeros(hessian.shape))
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
        if OUTPUT_LEVEL == 0:
            assert not KTR_set_int_param_by_name(kc, "outlev", 0)
            # verbose:
            # assert not KTR_set_int_param_by_name(kc, "outlev", 2)
            # Super verbose: prints iterates information
            # assert not KTR_set_int_param_by_name(kc, "outlev", 4)
        if hessian:
            # Exact hessian
            assert not KTR_set_int_param_by_name(kc, "hessopt", 1)
            assert not KTR_set_int_param_by_name(kc, "hessian_no_f", 1)
        else:
            assert not KTR_set_int_param_by_name(kc, "hessopt", 3) # SR1
        # Compare performance of all the algorithms
        assert not KTR_set_int_param_by_name(kc, "algorithm", 4) # SQP
        # assert not KTR_set_int_param_by_name(kc, "linsolver", 3) # Non-sparse
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
        x = x_init.copy()
        lambda_ = np.zeros(m + n)
        obj = bo.acquisition(x)[0]
        try:
            nStatus = KTR_solve(kc, x, lambda_, 0, obj, None, None, None, None, None,
                                None)
            nIter = KTR_get_number_iters(kc)
            fcEvals = KTR_get_number_FC_evals(kc)
            gaEvals = KTR_get_number_GA_evals(kc)
            hEvals = KTR_get_number_H_evals(kc)
        finally:
            KTR_free(kc)
        Result = namedtuple('Result', 'status nit fEvals gEvals hEvals success message')
        return x, obj, Result(status=nStatus, nit=nIter,
            fEvals=fcEvals, gEvals=gaEvals, hEvals=hEvals,
            success=(nStatus==0), message=str(nStatus)
        )
except OSError:
    logging.getLogger('').debug(
        "Couldn't load knitro. Make sure it is properly installed."
    )
except ModuleNotFoundError:
    logging.getLogger('').warning(
        "Couldn't find knitro. Make sure it is installed and visible."
    )

def solve(X_init, bounds, hessian, bo, solver):
    '''
    Perform non-linear optimization on bo.acquisition_function with either L-BFGS-B (Pseudo-Newton)
    or Knitro (Sequential Quadratic Programming)
    '''
    x_init = X_init.flatten()

    if solver == 'bfgs':
        x, y, status = bfgs_solve(x_init=x_init, bounds=bounds,
                                   hessian=hessian, bo=bo)
    elif solver == 'knitro':
        try:
            x, y, status = knitro_solve(x_init=x_init, bounds=bounds,
                                        hessian=hessian, bo=bo)
        except NameError:
            logging.getLogger('').critical(
                'Please install knitro to use it.'
            )
            raise
    else:
        assert False, 'Invalid nonlinear solver choice!'

    k = x.size // bo.dim
    X = x.reshape(k, bo.dim)
    return X, y, status
