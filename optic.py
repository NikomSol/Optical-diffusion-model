import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import copy
import math
import matplotlib.pyplot as plt


# def solve(cnf, properties, old_T_glob, old_g_glob, boundary):
#     N = cnf['N']
#     M = cnf['M']
#     dn = cnf['dn']
#     dm = cnf['dm']
#
#     grid = np.mgrid[0:N, 0:M]
#
#     return np.zeros((N, M)), properties['Ic'](dn * grid[0], dm * grid[1]) * properties['mu_a']

def solve(cnf, properties, old_T_glob, old_g_glob, boundary):
    N = cnf['M']
    M = cnf['N']

    hn = cnf["dm"]
    hm = cnf["dn"]

    old_T = np.transpose(old_T_glob)
    old_g = np.transpose(old_g_glob)

    D = properties["diffusion_opt"](old_T, old_g)
    mu_s = properties["mu_s"](old_T, old_g)
    mu_a = properties["mu_a"](old_T, old_g)
    boundary_local = boundary(old_T, old_g)

    Ic = get_Ic(cnf, properties, old_T_glob, old_g_glob).transpose()

    web = np.full((N, M), 0)
    b = np.zeros(N * M)
    W = np.zeros((N * M, N * M))

    for i in range(0, M):
        boundaryArray = boundary_local['rmin']
        W[i][i] = boundaryArray[0] + boundaryArray[1] / hn
        W[i][(1) * M + i] = -boundaryArray[1] / hn
        b[i] = boundaryArray[2]
    for i in range(0, M):
        boundaryArray = boundary_local['rmax']
        W[M + i][(N - 1) * M + i] = boundaryArray[0] + boundaryArray[1] / hn
        W[M + i][(N - 2) * M + i] = -boundaryArray[1] / hn
        b[M + i] = boundaryArray[2]
    for i in range(0, M - 2):
        boundaryArray = boundary_local['zmax']
        W[2 * M + i][(i + 1) * M + M - 1] = boundaryArray[0] + boundaryArray[1] / hn
        W[2 * M + i][(i + 1) * M + M - 2] = -boundaryArray[1] / hn
        b[2 * M + i] = boundaryArray[2]
    for i in range(0, M - 2):
        boundaryArray = boundary_local['zmin']
        W[2 * M + M - 2 + i][(i + 1) * M] = boundaryArray[0] + boundaryArray[1][i + 1] / hn
        W[2 * M + M - 2 + i][(i + 1) * M + 1] = -boundaryArray[1][i + 1] / hn
        b[2 * M + M - 2 + i] = boundaryArray[2]

    flag = 0
    for n in range(1, N - 1):
        for m in range(1, M - 1):
            W[2 * M + M - 2 + M - 2 + flag][(n + 1) * M + m] = D[n][m] / hn ** 2 * (1 + 1 / (2 * n))
            W[2 * M + M - 2 + M - 2 + flag][(n) * M + m] = -(D[n][m] * 2 / hn ** 2 + 2 * D[n][m] / hm ** 2) - mu_a[n,m]
            W[2 * M + M - 2 + M - 2 + flag][(n - 1) * M + m] = D[n][m] / hn ** 2 * (1 - 1 / (2 * n))
            W[2 * M + M - 2 + M - 2 + flag][(n) * M + m + 1] = D[n][m] / hm ** 2
            W[2 * M + M - 2 + M - 2 + flag][(n) * M + m - 1] = D[n][m] / hm ** 2
            b[2 * M + M - 2 + M - 2 + flag] = - mu_s[n, m] * Ic[n, m];
            flag = flag + 1

    sW = sp.csr_matrix(W)
    sol = spsolve(sW.tocsr(), b)

    for n in range(0, N):
        for m in range(0, M):
            web[n][m] = 10**-10 * sol[n * M + m]

    Id = web * 10**10
    Iall = Id + Ic
    sourses = mu_a * Iall

    return Iall.transpose(), sourses.transpose()


def get_Ic(cnf, properties, T, g):
    N = cnf['N']
    M = cnf['M']
    dn = cnf['dn']
    dm = cnf['dm']

    mu_a = properties['mu_a'](T, g)
    mu_s = properties['mu_s'](T, g)
    mu_t_total = np.zeros((N, M))

    for n in range(N - 1):
        mu_t_total[n + 1] = mu_t_total[n] + dn * (mu_a[n] + mu_s[n])

    w0 = properties['w0']
    lam = properties['lam']
    Ic0 = properties['Power'] / w0 ** 2

    waist = np.array([w0 * (1 + (lam * (n + 0.5) * dn / (math.pi * w0)) ** 2) ** (0.5) for n in range(N)])

    Ic = np.zeros((N, M))
    for n in range(N):
        r = np.array([m * dm for m in range(M)])
        Ic[n] = Ic0 * (w0 / waist) ** 2 * np.exp(
            - mu_t_total[n] - 2 * r ** 2 / waist ** 2)

    return Ic


if __name__ == "__main__":
    from input import cnf, properties, condition_boundary_optic

    N = cnf['N']
    M = cnf['M']

    T = np.full((N, M), 273)
    g = np.full((N, M), 1)

    Ic = get_Ic(cnf, properties, T, g)
    Iall, Q = solve(cnf, properties, T, g, condition_boundary_optic)

    plt.matshow(Ic)
    plt.matshow(Iall)
    plt.matshow(Q)
    plt.show()
