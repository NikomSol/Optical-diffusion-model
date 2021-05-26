import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class ElectroDiffusionEquation():

    @classmethod
    def static_1D_real(self, cnf, properties, bound_conditions):
        N = cnf['N']
        dn = cnf['dn']
        inv_dn = 1 / dn
        eps0 = cnf['eps0']

        zeros_N = np.zeros(N)
        ones_2N = np.ones(2 * N)

        sigma = properties['sigma']
        epsilon = properties['epsilon']
        diffusion = properties['diffusion']

        sigma_diffusion = sigma / diffusion
        inv_eps_dn2 = (1 / eps0) * (1 / epsilon) * dn * dn

        potential_start = bound_conditions['N_start_potential']
        potential_end = bound_conditions['N_end_potential']
        charge_start = bound_conditions['N_start_charge']
        charge_end = bound_conditions['N_end_charge']

        matrix_potential_equations = sp.dia_matrix((
            [ones_2N, -2 * ones_2N, ones_2N, np.concatenate((zeros_N, inv_eps_dn2))],
            [0, 1, 2, N + 1]),
            shape=(N - 2, 2 * N))
        matrix_charge_equations = sp.dia_matrix((
            [ones_2N,
             -2 * ones_2N - np.concatenate((zeros_N, sigma_diffusion * inv_eps_dn2)),
             ones_2N],
            [N, N + 1, N + 2]),
            shape=(N - 2, 2 * N))

        matrix_potential_start = sp.coo_matrix((
            [potential_start[0] - potential_start[1] * inv_dn, potential_start[1] * inv_dn],
            ([0, 0], [0, 1])), shape=(1, 2 * N))
        matrix_potential_end = sp.coo_matrix((
            [-potential_end[1] * inv_dn, potential_end[0] + potential_end[1] * inv_dn],
            ([0, 0], [N - 2, N - 1])), shape=(1, 2 * N))

        matrix_charge_start = sp.coo_matrix((
            [-sigma_diffusion[0] * charge_start[1],
             sigma_diffusion[0] * charge_start[1],
             -charge_start[1] + dn * charge_start[0] / diffusion[0],
             charge_start[1]],
            ([0, 0, 0, 0], [0, 1, N, N + 1])), shape=(1, 2 * N))
        matrix_charge_end = sp.coo_matrix((
            [-sigma_diffusion[-1] * charge_end[1],
             sigma_diffusion[-1] * charge_end[1],
             -charge_end[1] + dn * charge_end[0] / diffusion[-1],
             charge_end[1]],
            ([0, 0, 0, 0], [N - 2, N - 1, 2 * N - 2, 2 * N - 1])), shape=(1, 2 * N))

        matrix = sp.vstack([matrix_potential_start, matrix_potential_end,
                            matrix_charge_start, matrix_charge_end,
                            matrix_potential_equations, matrix_charge_equations])

        vector = np.zeros(2 * N)
        vector[0] = potential_start[2]
        vector[1] = potential_end[2]
        vector[2] = charge_start[2] * dn / diffusion[0]
        vector[3] = charge_end[2] * dn / diffusion[-1]

        print(matrix.toarray())
        solution = spsolve(matrix.tocsr(), vector)
        return solution
