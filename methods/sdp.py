import numpy as np
import scs
import collections
import logging
import scipy.sparse as sp
from pypardiso import spsolve, factorized

OUTPUT_LEVEL = 0

def sdp(omega, fmin, warm_start=True):
    '''
    Solves the following SDP with the first-order primal-dual solver SCS:
    ---------Primal Form----------
    minimize <M, \omega>
    s.t.     M - C_i = positive semidefinite for all i = 0...k
    Dual Form:
    ---------Dual Form------------
    minimize \sum_{i=1}^{k} <Y_i, C_i>
    s.t.     Y_i positive semidefinite for all i = 0...k
                \sum_{i=0}^{k} Y_i = \omega
    ------------------------------
    in order to get the value and the gradient of the acquisiton function.
    Inputs:
        omega: Second order moment matrix
        fmin: min value achieved so far, i.e. min(y0)
        warm_start: Whether or not to warm-start the solution
    Outpus:
        opt_val: Optimal value of the SDP
        M: Optimizer of the SDP
        Y: List of the Optimal Lagrange Multipliers (Dual Optimizers) of the SDP
        C: List of C_i, i = 0 ... k
    '''
    omega = (omega + omega.T)/2

    # Express the problem in the format required by scs
    data = create_scs_data(omega, fmin)
    k_ = omega.shape[0]
    cone = {'s': [k_]*k_}
    if not 'past_omegas' in globals() or len(past_omegas) == 0 or \
        past_omegas[0].shape[0] != k_:
        # Clear the saved solutions, as they are of different size
        # than the one we are currently trying to solve.
        reset_warm_starting()
    elif warm_start:
        # Update data with warm-started solution
        data = get_warm_start(omega, data)

    # Call SCS
    sol = scs.solve(data, cone, eps=1e-5, use_indirect=False, verbose=OUTPUT_LEVEL==1)
    # print(sol['info']['solveTime']) # Prints solution time for SDP.
    if sol['info']['status'] != 'Solved':
        logging.getLogger('opt').warning(
            'SCS solution status:' + sol['info']['status']
        )

    # Extract solution from SCS' structures
    M, Y = unpack_solution(sol['x'], sol['y'], k_)
    objective = -sol['info']['pobj']
    sol['C'] = data['C']

    if warm_start:
        # Save solution for warm starting
        past_solutions.append(sol); past_omegas.append(omega)

    return objective, M, Y, data['C']

def reset_warm_starting():
    '''
    Clears list of previous solutions that are used for warm-starting SCS
    '''
    global past_omegas, past_solutions
    past_omegas = collections.deque(maxlen=20)
    past_solutions = collections.deque(maxlen=20)

def get_warm_start(omega, data):
    '''
    Updates SCS' data structure with a warm-started solution
    '''
    k_ = omega.shape[0]

    # Search on the list of past solved problems for the one
    # that had the most similar omega matrix as the one we
    # are trying to solve now
    def sort_func(X):
        return np.linalg.norm(X - omega)
    min_score = float('inf')
    idx = -1
    for i in range(len(past_omegas)):
        score = sort_func(past_omegas[i])
        if min_score > score:
            min_score = score
            idx = i

    # Copy the solution from the closest match
    x = past_solutions[idx]['x'].copy()
    y = past_solutions[idx]['y'].copy()
    s = past_solutions[idx]['s'].copy()

    # Improve warm starting by moving the solution in the direction of dM, dy
    domega = -(past_omegas[idx] - omega)
    M, Y = unpack_solution(x, y, k_)
    C = past_solutions[idx]['C']
    dM, dY = solution_derivative(M, Y, C, domega, return_dY=True)

    x += pack(dM[:,:,0], k_)
    n = len(x)
    for i in range(0, k_):
        y[i*n:(i+1)*n] += pack(dY[i], k_)
        s[i*n:(i+1)*n]-= pack(dM[:,:,0], k_)

    # Add warm starting to the problem data
    data['x'] = x; data['y'] = y; data['s'] = s

    return data

def create_scs_data(omega, fmin):
    '''
    Returns a data structure that containts the problem data as required by SCS. 
    See github.com/cvxgrp/scs 
    '''
    k_ = omega.shape[0]

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
    A = sp.csc_matrix((z, (row_ind, col_ind)), shape=(k_*n, n))

    # Create b and C
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

    # Create c
    c = -pack(omega, k_)

    return {'A': A, 'b': b, 'c': c, 'C': C}

def solution_derivative(M, Y, C, domegas, return_dY=False):
    '''
    Calculates the derivatives of M and Y across perturbations of the cost matrix
    defined in domegas.
    Inputs: M: Primal Solution of the SDP
            Y: List of Dual Solutions of the SDP
            C: List of the auxiliary matrices in the definition of the SDP
            domegas: 3D-ndarray with each domegas[:, :, i] defining a perturbation
    Outputs: dM: 3D-ndarray with each dM[:, :, i] being the derivative of M when
                 perturbing omega across domega[:, :, i]
             dY: list of 3D-ndarrays having, similarly to dM,
                 the derivatives of Y when perturbing omega
    '''

    H = create_matrix(M, Y, C)
    assert len(domegas.shape) == 2 or len(domegas.shape) == 3
    if len(domegas.shape) == 2:
        domegas = domegas[:, :, None]

    n = domegas.shape[0]
    k = domegas.shape[-1]

    # z a vectorized copy of the upper triangular part of domegas
    z = np.zeros((n**2 + n*(n+1)//2, k))
    z[-n*(n+1)//2:, :] = domegas[np.triu_indices(n)]
    x = H(z)

    def convert_to_matrix(x):
        G = np.zeros((n, n, k))
        G[np.triu_indices(n)] = x[-n*(n+1)//2:, :]
        G = G + np.transpose(G, [1, 0, 2]) 
        G[np.diag_indices(n)] = G[np.diag_indices(n)]/2
        return G

    dM = convert_to_matrix(x[-n*(n+1)//2:, :])

    if return_dY:
        assert k == 1  # This part has been only implemented for the case where k = 1
        dy = np.reshape(x[0:n**2], (n, n, k))
        dY = []
        for i in range(n):
            eig_Y_i = np.linalg.eigh(Y[i]) 
            y_i = (eig_Y_i[0][-1]**.5 * eig_Y_i[1][:, -1])[:, None]
            dY.append(y_i.dot(dy[i].T) + dy[i].dot(y_i.T))

        return dM, dY
    else:
        return dM

def create_matrix(M, Y, C):
    '''
    Creates the left-hand side matrix of the linear system that gives dM, dy
    '''
    k = len(Y)
    y = []
    for i in range(len(Y)):
        eig_Y_i = np.linalg.eigh(Y[i]) 
        y_i = (eig_Y_i[0][-1]**.5 * eig_Y_i[1][:, -1])[:, None]
        y.append(y_i)

    S = M - C
    P = get_P(k)
    P_ = get_P_(k)
    H1 = sp.block_diag(S, format='csc')
    I = sp.eye(k, format='csc')
    for i in range(len(y)):
        h2 = sp.kron(y[i].T, I).dot(P_)
        h3 = P.dot(sp.kron(y[i], I) + sp.kron(I, y[i]))
        if i == 0:
            H2 = h2
            H3 = h3
        else:
            H2 = sp.vstack((H2, h2))
            H3 = sp.hstack((H3, h3))
    H = sp.bmat([[H1, H2], [H3, None]])
    return factorized(H)

def unpack_solution(x, y, k_):
    '''
    Creates matrices M and Y from scs solution structure
    '''
    M = unpack(x, k_)
    Y = []
    for i in range(k_):
        n = k_ * (k_ + 1) // 2
        Y.append(
            unpack(y[i*n:(i+1)*n], k_)
        )
    return M, Y

def pack(Z, n):
    '''
    Auxiliary function for 'packing' a matrix into
    a vector format, as required by scs.
    See github.com/cvxgrp/scs 
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
    Auxiliary function for 'unpacking' a packed matrix of a vector format,
    as required by scs. See github.com/cvxgrp/scs 
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

def get_P_(k):
    A = sp.csc_matrix((0, k*(k+1)//2))
    for i in range(k):
        B = sp.csc_matrix((0, 0))
        for j in range(i):
            tmp = sp.csc_matrix(([1], ([0], [i-j])), shape=(1, k - j))
            B = sp.block_diag((B, tmp))

        B = sp.block_diag((B, sp.eye(k - i, format='csc')))
        B = sp.hstack([B, sp.csc_matrix((k, k*(k+1)//2 - B.shape[1]))])
        A = sp.vstack([A, B])

    return A

def get_P(k):
    A = sp.csc_matrix((0, 0))
    I = sp.eye(k, format='csc')
    for i in range(k):
        A = sp.block_diag((A, I[i:, :]))

    return A
