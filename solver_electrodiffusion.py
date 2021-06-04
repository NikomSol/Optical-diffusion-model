import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class Solver_Electrodiffusion():
    def __init__(self, geometry, boundary_condition, properties):
        self.geometry = geometry
        self.boundary_condition = boundary_condition

    def solve(self):
        system_matrix = self.calc_system_matrix()
        free_vector = self.calc_free_vector()

        solution = spsolve(system_matrix.tocsr(), free_vector)
        return solution

    def calc_system_matrix(self):
        internal_areas = self.geometry_get_interal_areas()

        system_matrix = self.poisson_equation(internal_areas)

        return system_matrix
