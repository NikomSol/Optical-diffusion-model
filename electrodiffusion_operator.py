import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

cnf = {
    'symmetry':'descart',
    'step_number':[6],
    'step_size':[10**-10],

    'eps0': 8.85 * 10 ** (-12)
}

bound_positions = [int(cnf['step_number']/2)]
geometry = [[0, bound_positions[0]-1],[bound_positions[0] ,cnf['step_number']-1]]

properties = {
    'sigma': np.concatenate((np.zeros(len(geometry[0])), np.ones(len(geometry[1])) * 10 ** (-3))),
    'epsilon': np.concatenate((np.ones(len(geometry[0])), np.ones(len(geometry[1])) * 81)),
    'diffusion': np.concatenate((np.zeros(len(geometry[0])), np.ones(len(geometry[1])) * 10 ** (-9)))
}

bound_conditions = {
    'N_start_potential': [1, 0, 0],
    'N_start_charge': [1, 0, 0],

    'N_bound_potential_1':[[1,0,-1,0,0]],
    'N_bound_potential_2':[[0,1,0,-81,0]],
    'N_bound_charge_1':[[1,0,0,0,0]],
    'N_bound_charge_2':[[0,0,0,1,0]],

    'N_end_potential': [1, 0, 1],
    'N_end_charge': [0, 1, 0]
}


class ElectroDiffusionEquation():
    def __init__(self, cnf, geometry, properties, bound_conditions):
        self.step_number = cnf['step_number']
        self.step_size = cnf['step_size']

        self.symmetry = cnf['symmetry']
        self.dimension = cnf['step_number'].length()

        self.geometry = geometry
        self.areas_number = geometry.length()

        self.properties = properties
        self.diff_coef = self.calc_diff_coef(properties['diffusion'])

        self.bound_conditions = bound_conditions

    def solve(self):
        dimension = self.dimension
        symmetry = self.symmetry
        if dimension == 1:
            if symmetry == 'axial':
                sys_matrix = self.create_sys_matrix_1d_axial()
            elif symmetry == 'descart':
                sys_matrix = self.create_sys_matrix_1d_descartes()
            else:
                raise ValueError("Unknown symmetry")
            sys_vector = self.create_sys_vector_1d()
        elif dimension == 2:
            if symmetry == 'axial':
                sys_matrix = self.create_sys_matrix_2d_axial()
            elif symmetry == 'descart':
                sys_matrix = self.create_sys_matrix_2d_descartes()
            else:
                raise ValueError("Unknown symmetry")
            sys_vector = self.create_sys_vector_2d()
        elif dimension == 3:
            if symmetry == 'axial':
                sys_matrix = self.create_sys_matrix_3d_axial()
            elif symmetry == 'descart':
                sys_matrix = self.create_sys_matrix_3d_descartes()
            else:
                raise ValueError("Unknown symmetry")
            sys_vector = self.create_sys_vector_3d()
        else:
            raise ValueError("Unknown dimension")

        solution = spsolve(sys_matrix.tocsr(), sys_vector)
        return solution

    def create_sys_matrix_1d_descartes(self):
        geometry = self.geometry
        diff_coef = self.diff_coef
        bound_conditions = self.bound_conditions

        for i, area in enumerate(geometry):
            sys_matryx = (self.d2('u', 'x', area[1:-2], np.ones((3, area[1] - area[0] - 2))) +
                          self.d2('v', 'x', area[1:-2], diff_coef) +
                          self.d0('u')
                          )

    def calc_diff_coef(difusion):
        pass


def d1(self, nArea, mArea, dif_var, scheme, koef):
    pass


def d0(self, nArea, mArea, koef):
    pass
