import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

cnf = {
    'symmetry': 'descart',
    'step_number': (9,9),
    'step_size': (10 ** -10,10 ** -10),

    'eps0': 8.85 * 10 ** (-12),
    'R': 8.31
}


properties = {
    'sigma': np.concatenate((np.zeros(len(geometry[0])), np.ones(len(geometry[1])) * 10 ** (-3))),
    'epsilon': np.concatenate((np.ones(len(geometry[0])), np.ones(len(geometry[1])) * 81)),
    'diffusion': np.concatenate((np.zeros(len(geometry[0])), np.ones(len(geometry[1])) * 10 ** (-9)))
}

bound_conditions = {
    'N_start_potential': [1, 0, 0],
    'N_start_charge': [1, 0, 0],

    'N_bound_potential_1': [[1, 0, -1, 0, 0]],
    'N_bound_potential_2': [[0, 1, 0, -81, 0]],
    'N_bound_charge_1': [[1, 0, 0, 0, 0]],
    'N_bound_charge_2': [[0, 0, 0, 1, 0]],

    'N_end_potential': [1, 0, 1],
    'N_end_charge': [0, 1, 0]
}


# class ElectroDiffusionEquation():
#     def __init__(self, cnf, geometry, properties, bound_conditions):
#         self.step_number = cnf['step_number']
#         self.step_size = cnf['step_size']
#
#         self.symmetry = cnf['symmetry']
#         self.dimension = cnf['step_number'].length()
#
#         self.geometry = geometry
#         self.areas_number = geometry.length()
#
#         self.properties = properties
#         self.diff_coef = self.calc_diff_coef(properties['diffusion'])
#
#         self.bound_conditions = bound_conditions
#
#         var_numbers = 2
#         for i in cnf['step_number']:
#             var_numbers *= i
#         self.sys_matrix_shape = (var_numbers, var_numbers)
#
#     def solve(self):
#         dimension = self.dimension
#         symmetry = self.symmetry
#         if dimension == 1:
#             if symmetry == 'axial':
#                 sys_matrix = self.create_sys_matrix_1d_axial()
#             elif symmetry == 'descart':
#                 sys_matrix = self.create_sys_matrix_1d_descartes()
#             else:
#                 raise ValueError("Unknown symmetry")
#             sys_vector = self.create_sys_vector_1d()
#         elif dimension == 2:
#             if symmetry == 'axial':
#                 sys_matrix = self.create_sys_matrix_2d_axial()
#             elif symmetry == 'descart':
#                 sys_matrix = self.create_sys_matrix_2d_descartes()
#             else:
#                 raise ValueError("Unknown symmetry")
#             sys_vector = self.create_sys_vector_2d()
#         elif dimension == 3:
#             if symmetry == 'axial':
#                 sys_matrix = self.create_sys_matrix_3d_axial()
#             elif symmetry == 'descart':
#                 sys_matrix = self.create_sys_matrix_3d_descartes()
#             else:
#                 raise ValueError("Unknown symmetry")
#             sys_vector = self.create_sys_vector_3d()
#         else:
#             raise ValueError("Unknown dimension")
#
#         solution = spsolve(sys_matrix.tocsr(), sys_vector)
#         return solution
#
#     def create_sys_matrix_1d_descartes(self):
#
#         geometry = self.geometry
#         diff_coef = self.diff_coef
#         bound_conditions = self.bound_conditions
#         N = self.step_number
#
#         # add external bound conditions
#         sys_matryx = (self.d0('u', [0], bound_conditions['N_start_potential'][0]) +
#                       self.d1('u', 'x', [0], bound_conditions['N_start_potential'][1], 'right') +
#                       self.d0('v', [0], bound_conditions['N_start_potential'][0]) +
#                       self.d1('vj', 'x', [0], bound_conditions['N_start_potential'][1], 'right') +
#
#                       self.d0('u', [N - 1], bound_conditions['N_end_potential'][0]) +
#                       self.d1('u', 'x', [N - 1], bound_conditions['N_end_potential'][1], 'left') +
#                       self.d0('v', [N - 1], bound_conditions['N_end_potential'][0]) +
#                       self.d1('vj', 'x', [N - 1], bound_conditions['N_end_potential'][1], 'left')
#                       )
#
#         # add equations
#         for i, area in enumerate(geometry):
#             sys_matryx += (self.d2('u', 'x', area[1:-2], np.ones((3, area[1] - area[0] - 2))) +
#                            self.d2('v', 'x', area[1:-2], diff_coef[area[0] + 1:area[1] - 1]))
#
#         # add internal bound conditions
#         for i in range(len(geometry) - 1):
#             sys_matryx += (self.d0('u', [geometry[i][1]], bound_conditions['N_bound_potential_1'][0]) +
#                            self.d1('u', 'x', [geometry[i][1]], bound_conditions['N_bound_potential_1'][1], 'left') +
#                            self.d0('u', [geometry[i + 1][0]], bound_conditions['N_bound_potential_1'][2]) +
#                            self.d1('u', 'x', [geometry[i + 1][0]], bound_conditions['N_bound_potential_1'][3],
#                                    'right') +
#
#                            self.d0('u', [geometry[i][1]], bound_conditions['N_bound_potential_2'][0]) +
#                            self.d1('u', 'x', [geometry[i][1]], bound_conditions['N_bound_potential_2'][1], 'left') +
#                            self.d0('u', [geometry[i + 1][0]], bound_conditions['N_bound_potential_2'][2]) +
#                            self.d1('u', 'x', [geometry[i + 1][0]], bound_conditions['N_bound_potential_2'][3],
#                                    'right') +
#
#                            self.d0('v', [geometry[i][1]], bound_conditions['N_bound_charge_1'][0]) +
#                            self.d1('vj', 'x', [geometry[i][1]], bound_conditions['N_bound_charge_1'][1], 'left') +
#                            self.d0('v', [geometry[i + 1][0]], bound_conditions['N_bound_charge_1'][2]) +
#                            self.d1('vj', 'x', [geometry[i + 1][0]], bound_conditions['N_bound_charge_1'][3],
#                                    'right') +
#
#                            self.d0('v', 'x', [geometry[i][1]], bound_conditions['N_bound_charge_2'][0]) +
#                            self.d1('vj', 'x', [geometry[i][1]], bound_conditions['N_bound_charge_2'][1], 'left') +
#                            self.d0('v', 'x', [geometry[i + 1][0]], bound_conditions['N_bound_charge_2'][2]) +
#                            self.d1('vj', 'x', [geometry[i + 1][0]], bound_conditions['N_bound_charge_2'][3],
#                                    'right')
#                            )
#
#         return sys_matryx
#
#     def calc_diff_coef(difusion):
#         pass
#
#     def d2(self, var, dif_var, area, koef):
#         pass
#
#     def d1(self, var, dif_var, area, koef, scheme):
#         pass
#
#     def d0(self, var, area, koef):
#         step_number = self.step_number
#         sys_matrix_shape = self.sys_matrix_shape
#
#         return d0_matrix
