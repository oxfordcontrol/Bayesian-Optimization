import numpy as np
import numdifftools as nd
from .create_model import create_model
from methods import sdp

# Batch size
K = 3


def derivatives_numerical(x, model):
    '''
    Returns the gradient and hessian of the optimal value of
    the SDP with respect to x.
    Beware, the hessian is based on the analytical derivative,
    for accuracy and performance reasons.
    '''
    def opt_val(y):
        return model.acquisition(y)[0]

    def gradient(y):
        return model.acquisition(y)[1]

    gradient_numerical = nd.Gradient(opt_val)(x)
    hessian_numerical = nd.Hessian(opt_val)(x)

    return gradient_numerical, hessian_numerical


def derivatives_analytical(x, model):
    '''
    Returns the gradient and hessian of the optimal value of
    the SDP with respect to x.
    '''
    return model.acquisition(x)[1], model.acquisition_hessian(x)


def sensitivity_numerical(omega, direction, model):
    '''
    Returns the derivative of
    1) optimal value of SDP
    2) derivative primal and dual solution:
        d/dx([vec(M); vec(Y_0); ...; vec(Y_k)])
    at omega when perturbing the second moment matrix across 
    the given direction
    '''
    y_min = np.min(model.Y.value)

    def solution(om):
        '''
        Return [vec(M); vec(Y_0); ...; vec(Y_k)] at omega=om
        '''
        M, Y = sdp(om, y_min)[1:3]
        solution = M.flatten()
        for i in range(len(Y)):
            solution = np.concatenate((solution, Y[i].flatten()))
        return solution

    d_opt_val = nd.Derivative(
        lambda x: sdp(omega + x*direction, y_min)[0]
    )(0)

    d_solution = nd.Derivative(
        lambda x: solution(omega + x*direction)
    )(0)

    return d_opt_val, d_solution


def sensitivity_analytical(omega, direction, model):
    M = sdp(omega, np.min(model.Y.value))[1]
    d_opt_val = np.trace(M.T.dot(direction))

    LU = model.factor(omega)
    dM, dY = model.solve(LU, direction)

    d_solution = dM.flatten()
    for i in range(len(dY)):
        d_solution = np.concatenate((d_solution, dY[i].flatten()))

    return d_opt_val, d_solution


def test_sensitivity():
    '''
    Checks sensitivity results of the SDP when perturbing
    omega(X) at a direction D. X and D are chosen uniformly at random.
    '''
    bo = create_model(batch_size=K)
    X = bo.random_sample(bo.bounds, K)
    D = 1e-2*np.random.rand(K + 1, K + 1)
    D = D + D.T

    omega = bo.omega(X)

    d_opt_val, d_solution = sensitivity_analytical(omega, D, bo)
    d_opt_val_n, d_solution_n = sensitivity_numerical(omega, D, bo)

    # Check derivative of optimum value 
    np.testing.assert_allclose(d_opt_val, d_opt_val_n, rtol=1e-2)
    # Check derivative of primal and dual solution
    np.testing.assert_allclose(d_solution, d_solution_n, rtol=1e-1)


def test_derivatives():
    '''
    Tests gradient and hessian of the acquisition function.
    '''
    bo = create_model(batch_size=K)
    X = bo.random_sample(bo.bounds, K)

    gradient, hessian = derivatives_analytical(X.flatten(), bo)
    gradient_n, hessian_n = derivatives_numerical(X.flatten(), bo)

    # The following two work better when no warm starting is present
    # Maybe warm starting confuses the numerical differentiation?
    np.testing.assert_allclose(gradient, gradient_n, rtol=1e-1)
    np.testing.assert_allclose(hessian, hessian_n, rtol=2*1e-1)
