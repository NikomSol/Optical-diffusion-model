import numpy as np
from math import exp


def solve(cnf, properties, old_g, new_T):
    lnA = properties['lnA']
    Tdeg = properties['Tdeg']
    N = cnf['N']
    M = cnf['M']
    dt = cnf['dk']

    new_g = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            old_gnm = old_g[n, m]
            """
            Если ткань почти продеградировала на предыдущем шаге, 
            то принимаем на следующем шаге строго степень деградации сторого 0.
            Иначе продолжаем ее деградировать.
            Так избежим работы с большими числами. 
            """
            if old_gnm > 0.001:
                new_g[n, m] = old_gnm * exp(- dt * exp(lnA * (1 - Tdeg / new_T[n, m])))

    return new_g
