from methods import sdps
import numpy as np


def test_sdps():
    k = 3  # Batch size

    for i in range(10):
        # Generate problem data
        tmp = np.random.randn(k, k)
        sigma = tmp.dot(tmp.T) + 0.01*np.eye(tmp.shape[0], tmp.shape[1])
        mu = np.random.randn(k, 1)

        omega = np.zeros((k + 1, k + 1))
        omega[0:k, 0:k] = sigma + mu.dot(mu.T)
        omega[-1, 0:k] = mu.flatten()
        omega[0:k, -1] = mu.flatten()
        omega[-1, -1] = 1

        fmin = 5*np.random.randn(1)

        # Solve SDPs
        opt_val, M = sdps.sdp_primal(omega, fmin)[0:2]
        opt_val_new, M_new = sdps.sdp_primal_new(omega, fmin)[0:2]
        opt_val_dual, M_dual = sdps.sdp_dual(omega, fmin)[0:2]
        opt_val_dual_new, M_dual_new = sdps.sdp_dual_new(omega, fmin)[0:2]
        opt_val_scs, M_scs = sdps.sdp_scs(omega, fmin)[0:2]

        # Test the different formulations
        np.testing.assert_allclose(opt_val, opt_val_new, rtol=1e-4)
        np.testing.assert_allclose(opt_val_dual, opt_val_dual_new, rtol=1e-4)
        np.testing.assert_allclose(opt_val_dual, opt_val, rtol=1e-4)
        np.testing.assert_allclose(opt_val, opt_val_scs, rtol=1e-4)
        np.testing.assert_allclose(M_scs, M_dual_new, rtol=1e-2)