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


# cnf = {
#     'K': 3,
#     'N': 10,
#     'M': 3,
#     'dk': 0.1,
#     'dn': 0.1,
#     'dm': 0.1
#
#     'eps0': 8.85 * 10 ** (-12)
# }
# properties = {'sigma': 10 ** (-3) * np.ones[cnf['N']],
#               'epsilon': 1 * np.ones[cnf['N']],
#               'diffusion': 10 ** (-9) * np.ones[cnf['N']]}
#
# sources = 0 * np.ones((cnf['K'], cnf['N'], cnf['M']))
# start_conditions = np.ones((cnf['N'], cnf['M']))
# bound_conditions = cnf['K'] * [
#     {'Nstart': cnf['M'] * [[1, 0, 1]],
#      'Nend': cnf['M'] * [[1, 0, 1]],
#      'Mstart': cnf['N'] * [[1, 0, 1]],
#      'Mend': cnf['N'] * [[1, 0, 1]]}
# ]
#
# properties_static_1D = 0.6 / 4200 / 1000 * np.ones(cnf['N'])
# sources_static_1D = 0 * (10 ** (-6)) * np.ones(cnf['N'])
# bound_conditions_static_1D = {'Nstart': [1, -1, 0], 'Nend': [1, 0, 1]}
#
# solution_static_1D = DiffusionEquation.static_1D(cnf, properties_static_1D, bound_conditions_static_1D,
#                                                  sources_static_1D)
# properties_static_2D = 0.6 / 4200 / 1000 * np.ones(cnf['N'], cnf['M'])
# sources_static_2D = 0 * (10 ** (-6)) * np.ones(cnf['N'], cnf['M'])
# bound_conditions_static_2D = {
#     'Nstart': cnf['M'] * [[1, 0, 1]],
#     'Nend': cnf['M'] * [[1, 0, 1]],
#     'Mstart': cnf['N'] * [[1, 0, 1]],
#     'Mend': cnf['N'] * [[1, 0, 1]]}
#
# solution_static_2D = DiffusionEquation.static_2D(cnf, properties_static_2D, bound_conditions_static_2D,
#                                                  sources_static_2D)
# print(solution_static_2D)
