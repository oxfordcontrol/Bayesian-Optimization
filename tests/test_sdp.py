from methods import sdp
import numpy as np
import cvxpy as cvx

def sdp_mosek(omega, fmin):
    k_ = omega.shape[0]

    Y = []
    C = []

    C.append(np.zeros((k_, k_)))
    C[0][-1, -1] = 0
    Y.append(cvx.Semidef(k_))
    cost_sum = Y[0]*C[0]

    for i in range(1, k_):
        Y.append(cvx.Semidef(k_))
        C.append(np.zeros((k_, k_)))
        C[i][-1, i - 1] = 1/2
        C[i][i - 1, -1] = 1/2
        C[i][-1, -1] = -fmin
        cost_sum += Y[i]*C[i]

    constraints = [sum(Y) == omega]

    objective = cvx.Minimize(cvx.trace(cost_sum))

    prob = cvx.Problem(objective, constraints)
    opt_val = prob.solve(solver=cvx.MOSEK, verbose=0)

    # Assert a valid solution is returned
    assert (isinstance(opt_val, np.ndarray) or isinstance(opt_val, float))\
        and np.isfinite(opt_val)

    M = -constraints[0].dual_value
    M = np.asarray((M + M.T)/2)  # From matrix to array

    Y_return = []
    for y in Y:
        Y_return.append(np.asarray(y.value))

    return opt_val, M, Y_return, C

def test_sdp():
    np.random.seed(0)
    k = 5  # Batch size

    for i in range(10):
        # Generate problem data
        tmp = np.random.randn(k, k)
        # TODO: Change the way the sigma is constructed
        # Creating a matrix as below results in a very bad condition number
        sigma = tmp.dot(tmp.T) + 0.01*np.eye(tmp.shape[0], tmp.shape[1])
        mu = np.random.randn(k, 1)

        omega = np.zeros((k + 1, k + 1))
        omega[0:k, 0:k] = sigma + mu.dot(mu.T)
        omega[-1, 0:k] = mu.flatten()
        omega[0:k, -1] = mu.flatten()
        omega[-1, -1] = 1

        fmin = np.random.randn(1)

        # Solve SDP with SCS and compare with MOSEK
        opt_val_scs, M_scs = sdp.sdp(omega, fmin, warm_start=False)[0:2]
        opt_val_mosek, M_mosek = sdp_mosek(omega, fmin)[0:2]

        # Test the different formulations
        np.testing.assert_allclose(opt_val_scs, opt_val_mosek, rtol=1e-4)

        # This is coarse, hence it shouldn't fail
        assert np.linalg.norm(M_scs - M_mosek)/np.linalg.norm(M_mosek) < 1e-2 
        # This might fail occasionaly, as SCS is of limited accuracy
        np.testing.assert_allclose(M_scs, M_mosek, rtol=5e-2)