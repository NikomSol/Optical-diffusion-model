import numpy as np
import scipy.sparse as sp

class Config():
    cnf = {
        'symmetry': 'descart',
        'step_number': (9, 9),
        'step_size': (10 ** -10, 10 ** -10),

        'eps0': 8.85 * 10 ** (-12),
        'R': 8.31
    }

class Geometry():
    def __init__(self, area_function):
        cnf = Config.cnf

        step_number = cnf['step_number']
        step_size = cnf['step_size']

        x_array = np.arange(step_number[0])  # np.linspace(0, step_number[0] * step_size[0], step_number[0])
        y_array = np.arange(step_number[1])  # np.linspace(0, step_number[1] * step_size[1], step_number[1])
                                             # [x_mesh, y_mesh] = np.meshgrid(x_array, y_array)
        self.area_matrix = [[area_function(x, y) for x in x_array] for y in y_array]

class Properties():
    air_properties = {
        'eps': 1,
        'mobility': 0,
        'ion_concentration': 0,

        'absorbtion_koef': 0,
        'scattering_koef': 0,
        'scattering_anisotropy': 1,

        'density_native': 0,
        'thermal_conductivity': 0,
        'heat_capacity': 0,
    }

    potato_native = {
        'eps': 81,
        'mobility': 10 ** (-5),
        'ion_concentration': 1,

        'absorbtion_koef': 1,
        'scattering_koef': 1000,
        'scattering_anisotropy': 0.9,

        'density_native': 1000,
        'thermal_conductivity': 0.6,
        'heat_capacity': 4200,
    }

    potato_degraded = {
        'eps': 81,
        'mobility': 10 ** (-5),
        'ion_concentration': 1,

        'absorbtion_koef': 1,
        'scattering_koef': 1000,
        'scattering_anisotropy': 0.9,

        'density_native': 1000,
        'thermal_conductivity': 0.6,
        'heat_capacity': 4200,
    }

    potato_Arrenius = {
        'arr_velocity': 1,
        'arr_energy': 1
    }

    def __init__(self, geometry, temperature, conditions):
        pass


if __name__ == '__main__':

    def area_function(x, y):
        return 'potata' if (x >= 3) & (y >= 3)  else 'air'

    print(Properties.potato_degraded['eps'])

    geometry = Geometry(area_function)
    print(geometry.area_matrix)
