import numpy as np
import scipy.sparse as sp


class Geometry():
    def __init__(self, cnf, area_function):
        step_number = cnf['step_number']
        step_size = cnf['step_size']

        x = np.arange(step_number[0])  # np.linspace(0, step_number[0] * step_size[0], step_number[0])
        y = np.arange(step_number[1])  # np.linspace(0, step_number[1] * step_size[1], step_number[1])

        [x_mesh, y_mesh] = np.meshgrid(x, y)
        self.area_matrix = area_function(x_mesh, y_mesh)


if __name__ == '__main__':
    cnf = {
        'symmetry': 'descart',
        'step_number': (9, 9),
        'step_size': (10 ** -10, 10 ** -10),

        'eps0': 8.85 * 10 ** (-12),
        'R': 8.31
    }


    def area_function(x, y):
        return (x >= 3) & (y >= 3)


    geometry = Geometry(cnf, area_function)
    print(geometry.area_matrix)
