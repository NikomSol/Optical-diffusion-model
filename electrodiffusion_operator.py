import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class ElectroDiffusionEquation():
    def __init__(self, cnf, geometry, properties, bound_conditions):
        self.step_number = cnf['step_number']
        self.step_length = cnf['step_length']

        self.symmetry = cnf['symmetry']
        self.dimension = cnf['step_number'].length()

        self.geometry = geometry
        self.areas_number = geometry.length()

        self.properties = properties

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
        properties = self.properties
        unhomo_diffusion_coefficient = self.unhomo_diffusion_coefficient(properties['diffusion_coefficient'])
        for area in geometry:
            sys_matryx = (self.d2('u', 'x', area, np.ones((3,area[1]-area[0]-2))) +
                          self.d2('v', 'x', area, unhomo_diffusion_coefficient))

    def create_sys_matrix_2d(self):
        pass

    def create_sys_vector_1d(self):
        pass

    def create_sys_vector_2d(self):
        pass

    def d2(self, area dif_var, koef):
    pass


def d1(self, nArea, mArea, dif_var, scheme, koef):
    pass


def d0(self, nArea, mArea, koef):
    pass
