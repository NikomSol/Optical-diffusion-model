import numpy as np
import copy
import scipy.sparse as sp


def solve(cnf, properties, old_T_glob, new_Q_glob, boundary):
    N = cnf['M']
    M = cnf['N']

    hn = cnf["dm"]
    hm = cnf["dn"]
    ht = cnf['dk']

    old_T = np.transpose(old_T_glob)
    new_Q = np.transpose(new_Q_glob)

    rho = properties["rho"]
    c = properties["c"]

    b = np.zeros(N * M)
    vecT = old_T.reshape(1, N * M)[0]
    W = np.zeros((N * M, N * M))

    D = properties["kappa"]

    for n in range(0, N):
        boundaryArray = boundary['zmin']
        vecT[n * M] = hm * (boundaryArray[2] + boundaryArray[1] * vecT[n * M + 1] / hm) / (
                (boundaryArray[0] + boundaryArray[1] / hm) * hm)  # (vecT[n*M+1]+T0*h*hm)/(1+h*hm)
        boundaryArray = boundary['zmax']
        vecT[n * M + M - 1] = hm * (boundaryArray[2] - boundaryArray[1] * vecT[n * M + M - 2] / hm) / (
                (boundaryArray[0] - boundaryArray[1] / hm) * hm)
    for m in range(0, N):
        boundaryArray = boundary['rmin']
        vecT[m] = hn * (boundaryArray[2] + boundaryArray[1] * vecT[N + m] / hn) / (
                (boundaryArray[0] + boundaryArray[1] / hn) * hn)
        boundaryArray = boundary['rmax']
        vecT[(N - 1) * M + m] = hn * (boundaryArray[2] - boundaryArray[1] * vecT[(N - 2) * M + m] / hn) / (
                (boundaryArray[0] - boundaryArray[1] / hn) * hn)

    coef = ht * 1 / (rho * c)
    coefN = coef / hn ** 2
    coefM = coef / hm ** 2

    for n in range(1, N - 1):
        for m in range(1, M - 1):
            W[(n) * M + m][(n + 1) * M + m] = D * coefN * (1 + 1 / (2 * n))
            W[(n) * M + m][(n) * M + m] = -(D * 2 * coefN + 2 * D * coefM)
            W[(n) * M + m][(n - 1) * M + m] = D * coefN * (1 - 1 / (2 * n))
            W[(n) * M + m][(n) * M + m + 1] = D * coefM
            W[(n) * M + m][(n) * M + m - 1] = D * coefM
            b[(n) * M + m] = new_Q[n, m] * coef

    new_vecT = vecT + (np.dot(W, vecT) + b)

    for n in range(0, N):
        boundaryArray = boundary['zmin']
        new_vecT[n * M] = hm * (boundaryArray[2] + boundaryArray[1] * new_vecT[n * M + 1] / hm) / (
                (boundaryArray[0] + boundaryArray[1] / hm) * hm)  # (vecT[n*M+1]+T0*h*hm)/(1+h*hm)
        boundaryArray = boundary['zmax']
        new_vecT[n * M + M - 1] = hm * (boundaryArray[2] - boundaryArray[1] * new_vecT[n * M + M - 2] / hm) / (
                (boundaryArray[0] - boundaryArray[1] / hm) * hm)
    for m in range(0, N):
        boundaryArray = boundary['rmin']
        new_vecT[m] = hn * (boundaryArray[2] + boundaryArray[1] * new_vecT[N + m] / hn) / (
                (boundaryArray[0] + boundaryArray[1] / hn) * hn)
        boundaryArray = boundary['rmax']
        new_vecT[(N - 1) * M + m] = hn * (boundaryArray[2] - boundaryArray[1] * new_vecT[(N - 2) * M + m] / hn) / (
                (boundaryArray[0] - boundaryArray[1] / hn) * hn)


    res = new_vecT.reshape(N, M)
    return np.transpose(res)
