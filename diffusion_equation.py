import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class DiffusionEquation():

    @classmethod
    def static_1D(self, cnf, properties, bound_conditions, sources):
        N = cnf['N']
        dn = cnf['dn']

        matrix_equations = sp.dia_matrix(([np.ones(N), -2 * np.ones(N), np.ones(N)], [0, 1, 2]), shape=(N - 2, N))

        matrix_start = sp.coo_matrix(([dn * bound_conditions['Nstart'][0] - bound_conditions['Nstart'][1],
                                       bound_conditions['Nstart'][1]],
                                      ([0, 0], [0, 1])), shape=(1, N))
        matrix_end = sp.coo_matrix(([-bound_conditions['Nend'][1],
                                     dn * bound_conditions['Nend'][0] + bound_conditions['Nend'][1]],
                                    ([0, 0], [N - 2, N - 1])), shape=(1, N))
        matrix = sp.vstack([matrix_start, matrix_equations, matrix_end])

        vector = -dn * dn * sources / properties
        vector[0] = dn * bound_conditions['Nstart'][2]
        vector[-1] = dn * bound_conditions['Nend'][2]

        solution = spsolve(matrix.tocsr(), vector)
        return solution

    @classmethod
    def static_2D(self, cnf, properties, bound_conditions, sources):
        pass
        # N = cnf['N']
        # dn = cnf['dn']
        # M = cnf['M']
        # dm = cnf['dm']
        #
        # matrix_equation = sp.coo_matrix(([dm * dm, dm * dm, dn * dn, dn * dn, -2 * dn * dn + -2 * dm * dm],
        #                                  ([0, 0, 0, 0, 0],
        #                                   [M * (n - 1) + m, M * (n + 1) + m, M * n + m - 1, M * n + m + 1, M * n + m])),
        #                                 shape=(1, N*M))

        # solution = spsolve(matrix.tocsr(), vector)
        return solution

    def time_dependence_1D(self):
        pass

    def static_1D_axial(self):
        pass

    def time_dependence_1D_axial(self):
        pass

    def time_dependence_2D(self):
        pass

    def static_2D_axial(self):
        pass

    def time_dependence_2D_axial(self):
        pass
