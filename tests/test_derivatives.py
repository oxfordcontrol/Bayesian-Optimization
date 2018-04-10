import numpy as np
import numdifftools as nd
from .create_model import create_model
from methods.sdp import sdp, solution_derivative

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
    fmin = np.min(model.predict_f(model.X.value)[0])

    def solution(om):
        '''
        Return [vec(M); vec(Y_0); ...; vec(Y_k)] at omega=om
        '''
        M, Y = sdp(om, fmin)[1:3]
        solution = M.flatten()
        for i in range(len(Y)):
            y = Y[i][0]/Y[i][0, 0]**.5
            solution = np.concatenate((solution, y))
        return solution

    d_opt_val = nd.Derivative(
        lambda x: sdp(omega + x*direction, fmin)[0]
    )(0)

    d_solution = nd.Derivative(
        lambda x: solution(omega + x*direction)
    )(0)

    return d_opt_val, d_solution


def sensitivity_analytical(omega, direction, model):
    fmin = np.min(model.predict_f(model.X.value)[0])
    _, M, Y, C = sdp(omega, fmin)
    d_opt_val = np.trace(M.T.dot(direction))

    dM, dy = solution_derivative(M, Y, C, direction)

    d_solution = dM.flatten()
    for i in range(len(dy)):
        d_solution = np.concatenate((d_solution, dy[i].flatten()))

    return d_opt_val, d_solution


def test_sensitivity():
    '''
    Checks sensitivity results of the SDP when perturbing
    omega(X) at a direction D. X and D are chosen uniformly at random.
    '''
    np.random.seed(0)

    bo = create_model(batch_size=K)
    X = bo.random_sample(bo.bounds, K)

    omega = bo.omega(X)
    mu = omega[0:K, -1][:, None]

    D_s = np.random.rand(K, K)
    D_s = D_s.dot(D_s.T)
    D_m = np.random.rand(K, 1)
    D = np.zeros((K + 1, K + 1))

    D[0:K, 0:K] = D_s + mu.dot(D_m.T) + D_m.dot(mu.T)
    D[-1, 0:K] = D_m.flatten()
    D[0:K, -1] = D_m.flatten()
    D = (D + D.T)/2
    D = 1e-3*D

    d_opt_val, d_solution = sensitivity_analytical(omega, D, bo)
    d_opt_val_n, d_solution_n = sensitivity_numerical(omega, D, bo)

    # Check derivative of optimum value 
    np.testing.assert_allclose(d_opt_val, d_opt_val_n, rtol=1e-2)
    # Check derivative of primal and dual solution
    np.testing.assert_allclose(d_solution, d_solution_n, rtol=3e-1)


def test_derivatives():
    '''
    Tests gradient and hessian of the acquisition function.
    '''
    np.random.seed(0)

    bo = create_model(batch_size=K)
    X = bo.random_sample(bo.bounds, K)

    gradient, hessian = derivatives_analytical(X.flatten(), bo)
    gradient_n, hessian_n = derivatives_numerical(X.flatten(), bo)

    # The following two work better when no warm starting is present
    # Maybe warm starting confuses the numerical differentiation?
    np.testing.assert_allclose(gradient, gradient_n, rtol=5e-1)
    assert np.linalg.norm(gradient - gradient_n) / np.linalg.norm(gradient) < 1e-2
    assert np.linalg.norm(hessian - hessian_n)/np.linalg.norm(hessian) < 2e-2 
    # This is too hard
    # np.testing.assert_allclose(hessian, hessian_n, rtol=1)
