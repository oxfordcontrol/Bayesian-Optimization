import cvxpy as cvx
import numpy as np
import scs
import collections
import logging
import scipy as sp

OUTPUT_LEVEL = 0


def sdp(omega, fmin):
    '''
    Set the default option
    '''
    return sdp_scs(omega, fmin)


def sdp_dual(omega, fmin):
    '''
    Solves the SDP, the solution of which is the acquisiton function.
    Inputs:
        omega: Second order moment matrix
        fmin: min value achieved so far, i.e. min(y0)
    Outpus:
        opt_val: Optimal value of the SDP
        M: Solution of the SDP
        Y: Dual Solution of the SDP
        C: List of auxiliary matrices used in the cone constraints

    The dual formulation is used, as this appears to be faster:

    minimize \sum_{i=0}^{k} <Y_i, C_i> - fmin
    s.t.     Y_i positive semidefinite for all i = 0...k
                \sum_{i=0}^{k} Y_i = \omega
    '''
    k = omega.shape[1] - 1

    Y = []
    C = []

    C.append(np.zeros((k + 1, k + 1)))
    C[0][-1, -1] = fmin
    Y.append(cvx.Semidef(k+1))
    cost_sum = Y[0]*C[0]

    for i in range(1, k + 1):
        Y.append(cvx.Semidef(k+1))
        C.append(np.zeros((k + 1, k + 1)))
        C[i][-1, i - 1] = 1/2
        C[i][i - 1, -1] = 1/2
        cost_sum += Y[i]*C[i]

    constraints = [sum(Y) == omega]

    objective = cvx.Minimize(cvx.trace(cost_sum) - fmin)

    prob = cvx.Problem(objective, constraints)
    # Use only one thread for MOSEK
    params = {'MSK_IPAR_NUM_THREADS': 1,
              'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-11,
              'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-11,
              'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-11}
    opt_val = prob.solve(solver=cvx.MOSEK, verbose=OUTPUT_LEVEL,
                         mosek_params=params)

    # Assert a valid solution is returned
    assert (isinstance(opt_val, np.ndarray) or isinstance(opt_val, float))\
        and np.isfinite(opt_val)

    M = -constraints[0].dual_value
    M = np.asarray((M + M.T)/2)  # From matrix to array

    Y_return = []
    for y in Y:
        Y_return.append(np.asarray(y.value))

    logging.getLogger('opt').debug(
        'omega:' + str(omega)
    )

    if prob.status != 'optimal':
        logging.getLogger('opt').warning(
            'SDP:' + str(prob.status) + ' It:' +
            str(prob.solver_stats.num_iters)
        )

    return opt_val, M, Y_return, C


def sdp_dual_new(omega, fmin):
    k = omega.shape[1] - 1

    Y = []
    C = []

    C.append(np.zeros((k + 1, k + 1)))
    C[0][-1, -1] = 0
    Y.append(cvx.Semidef(k+1))
    cost_sum = Y[0]*C[0]

    for i in range(1, k + 1):
        Y.append(cvx.Semidef(k+1))
        C.append(np.zeros((k + 1, k + 1)))
        C[i][-1, i - 1] = 1/2
        C[i][i - 1, -1] = 1/2
        C[i][-1, -1] = -fmin
        cost_sum += Y[i]*C[i]

    constraints = [sum(Y) == omega]

    objective = cvx.Minimize(cvx.trace(cost_sum))

    prob = cvx.Problem(objective, constraints)
    # Use only one thread for MOSEK
    params = {'MSK_IPAR_NUM_THREADS': 1,
              'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-11,
              'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-11,
              'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-11}
    opt_val = prob.solve(solver=cvx.MOSEK, verbose=OUTPUT_LEVEL,
                         mosek_params=params)

    # Assert a valid solution is returned
    assert (isinstance(opt_val, np.ndarray) or isinstance(opt_val, float))\
        and np.isfinite(opt_val)

    M = -constraints[0].dual_value
    M = np.asarray((M + M.T)/2)  # From matrix to array

    Y_return = []
    for y in Y:
        Y_return.append(np.asarray(y.value))

    logging.getLogger('opt').debug(
        'omega:' + str(omega)
    )

    if prob.status != 'optimal':
        logging.getLogger('opt').warning(
            'SDP:' + str(prob.status) + ' It:' +
            str(prob.solver_stats.num_iters)
        )

    return opt_val, M, Y_return, C


def sdp_primal(omega, fmin):
    k = omega.shape[1] - 1

    M = cvx.Symmetric(k+1)

    C_i = np.zeros((k + 1, k + 1))
    C_i[-1, -1] = fmin

    constraints = [C_i - M == cvx.Semidef(k + 1)]
    for i in range(0, k):
        C_i = np.zeros((k + 1, k + 1))
        C_i[-1, i] = 1/2
        C_i[i, -1] = 1/2
        constraints += [C_i - M == cvx.Semidef(k + 1)]

    objective = cvx.Maximize(cvx.trace(omega * M) - fmin)

    prob = cvx.Problem(objective, constraints)
    opt_val = prob.solve(solver=cvx.MOSEK, verbose=OUTPUT_LEVEL)

    return opt_val, M


def sdp_primal_new(omega, fmin):
    k = omega.shape[1] - 1

    M = cvx.Symmetric(k+1)

    C_i = np.zeros((k + 1, k + 1))

    constraints = [C_i - M == cvx.Semidef(k + 1)]
    for i in range(0, k):
        C_i = np.zeros((k + 1, k + 1))
        C_i[-1, i] = 1/2
        C_i[i, -1] = 1/2
        C_i[-1, -1] = -fmin
        constraints += [C_i - M == cvx.Semidef(k + 1)]

    objective = cvx.Maximize(cvx.trace(omega * M))

    prob = cvx.Problem(objective, constraints)
    opt_val = prob.solve(solver=cvx.MOSEK, verbose=OUTPUT_LEVEL)

    return opt_val, M


def pack(Z, n):
    '''
    Auxiliary function for 'packing' a matrix into
    a vector format, as required by scs.
    '''
    Z = np.copy(Z)
    tidx = np.triu_indices(n)
    tidx = (tidx[1], tidx[0])
    didx = np.diag_indices(n)

    Z = Z * np.sqrt(2.)
    Z[didx] = Z[didx] / np.sqrt(2.)
    z = Z[tidx]

    return z


def unpack(z, n):
    '''
    Auxiliary function for 'unpacking' a packed matrix of
    a vector format, as required by scs.
    '''
    z = np.copy(z)
    tidx = np.triu_indices(n)
    tidx = (tidx[1], tidx[0])
    didx = np.diag_indices(n)

    Z = np.zeros((n, n))
    Z[tidx] = z
    Z = (Z + np.transpose(Z)) / np.sqrt(2.)
    Z[didx] = Z[didx] / np.sqrt(2.)

    return Z


omega_list = collections.deque(maxlen=10)
x_list = collections.deque(maxlen=10)
y_list = collections.deque(maxlen=10)
s_list = collections.deque(maxlen=10)
A = None
b = None
C = None


def sdp_scs(omega, fmin):
    '''
    Solves the SDP, with the first order solver SCS,
    the solution of which is the acquisiton function.
    Inputs:
        omega: Second order moment matrix
        fmin: min value achieved so far, i.e. min(y0)
    Outpus:
        opt_val: Optimal value of the SDP
        M: Optimizer of the SDP
    The dual formulation is used, as this appears to be faster:
    minimize \sum_{i=0}^{k} <Y_i, C_i> - fmin
    s.t.     Y_i positive semidefinite for all i = 0...k
                \sum_{i=0}^{k} Y_i = \omega
    '''
    k_ = omega.shape[0]  # abbreviation: k_ = k + 1

    c = -pack(omega, k_)

    global omega_list, x_list, y_list, s_list, A, b, C
    if len(omega_list) > 0 and omega_list[0].shape[0] == k_:
        for i in range(1, k_):
            # Update fmin in C, b
            n = k_ * (k_ + 1) // 2
            C[i][-1, -1] = -fmin
            b[n*(i + 1) - 1] = -fmin

        # Warm starting - do not recalculate the problem data again
        # Find nearest neighbour solution
        def sort_func(X):
            return np.linalg.norm(X - omega)
        min_score = float('inf')
        idx = -1
        for i in range(len(omega_list)):
            score = sort_func(omega_list[i])
            if min_score > score:
                min_score = score
                idx = i
        assert idx != -1

        data = {'A': A, 'b': b, 'c': c,
                'x': x_list[idx], 'y': y_list[idx],
                's': s_list[idx]}
    else:
        # First run. Create static problem data: b, A
        # Reset warm starting
        omega_list = collections.deque(maxlen=10)
        x_list = collections.deque(maxlen=10)
        y_list = collections.deque(maxlen=10)
        s_list = collections.deque(maxlen=10)

        # Create b
        b = np.array([])
        C = []
        C.append(np.zeros((k_, k_)))
        b = np.append(b, pack(C[0], k_))
        for i in range(1, k_):
            C.append(np.zeros((k_, k_)))
            C[i][-1, i - 1] = 1/2
            C[i][i - 1, -1] = 1/2
            C[i][-1, -1] = -fmin
            b = np.append(b, pack(C[i], k_))

        # Create A
        n = k_ * (k_ + 1) // 2
        A = np.zeros((k_*n, n))
        row_ind = np.array([])
        col_ind = np.array([])
        z = np.array([])
        for i in range(n):
            row_ind = np.append(row_ind, np.arange(i, k_*n, n))
            col_ind = np.append(col_ind, np.repeat(i, k_))
            z = np.append(z, np.repeat(1, k_))
        A = sp.sparse.csc_matrix((z, (row_ind, col_ind)), shape=(k_*n, n))

        # Gather data
        data = {'A': A, 'b': b, 'c': c}

    cone = {'s': [k_]*k_}

    sol = scs.solve(data, cone, eps=1e-5, verbose=False)
    if omega is not None and not np.isnan(omega).any():
        # If solver returned a valid solution, then:
        # Save solution for warm starting
        omega_list.append(omega)
        x_list.append(sol['x'])
        y_list.append(sol['y'])
        s_list.append(sol['s'])

    if sol['info']['status'] != 'Solved':
        print('Solver: solution status ', sol['info']['status'])

    opt_val = -(sol['info']['pobj'] + sol['info']['dobj'])/2
    M = unpack(sol['x'], k_)
    Y = []
    for i in range(k_):
        n = k_ * (k_ + 1) // 2
        Y.append(
            unpack(sol['y'][i*n:(i+1)*n], k_)
        )

    return opt_val, M, Y, C
