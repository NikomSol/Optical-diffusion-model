import numpy as np
import scipy.sparse as sp


class Config():
    eps0 = 8.85 * 10 ** (-12)
    R = 8.31
    F = 96485.3

    def __init__(self, symmetry='decart', T0=273,
                 K=10, dk=10 ** (-6),
                 N=9, dn=10 ** (-6),
                 M=9, dm=10 ** (-6)):
        self.K = K
        self.N = N
        self.M = M
        self.dk = dk
        self.dn = dn
        self.dm = dm
        self.symmetry = symmetry
        self.T0 = T0


class Geometry():
    def __init__(self, cnf, type="slice", nTissueStart=5):
        self.cnf = cnf
        self.type = type
        self.nTissueStart = nTissueStart

        N = cnf.N
        M = cnf.M

        if type == 'slice':
            self.domain_matrix = nTissueStart * [M * ['air']] + (N - nTissueStart) * [M * ['tissue']]
        else:
            raise ValueError("Unknown geometry type")

    def get_bound(self, domain, bound, only=False, without=[]):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        nTissueStart = self.nTissueStart

        if domain == 'air':
            if bound == 'nStart':
                normal = [-1, 0]
                if only:
                    coordinates = np.array([[0, m] for m in only])
                else:
                    coordinates = np.array([[0, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)
            if bound == 'nEnd':
                normal = [1, 0]
                if only:
                    coordinates = np.array([[nTissueStart - 1, m] for m in only])
                else:
                    coordinates = np.array([[nTissueStart - 1, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)

            if bound == 'mStart':
                normal = [0, -1]
                if only:
                    coordinates = np.array([[n, 0] for n in only])
                else:
                    coordinates = np.array([[n, 0] for n in np.delete(np.arange(nTissueStart), without)])
                return Boundary(coordinates, normal)
            if bound == 'mEnd':
                normal = [0, 1]
                if only:
                    coordinates = np.array([[n, M - 1] for n in only])
                else:
                    coordinates = np.array([[n, M - 1] for n in np.delete(np.arange(nTissueStart), without)])
                return Boundary(coordinates, normal)

        if domain == 'tissue':
            if bound == 'nStart':
                normal = [-1, 0]
                if only:
                    coordinates = np.array([[nTissueStart, m] for m in only])
                else:
                    coordinates = np.array([[nTissueStart, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)
            if bound == 'nEnd':
                normal = [1, 0]
                if only:
                    coordinates = np.array([[N - 1, m] for m in only])
                else:
                    coordinates = np.array([[N - 1, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)

            if bound == 'mStart':
                normal = [0, -1]
                if only:
                    coordinates = np.array([[n, 0] for n in only])
                else:
                    coordinates = np.array([[n, 0] for n in np.delete(np.arange(nTissueStart, N), without)])
                return Boundary(coordinates, normal)
            if bound == 'mEnd':
                normal = [0, 1]
                if only:
                    coordinates = np.array([[n, M - 1] for n in only])
                else:
                    coordinates = np.array([[n, M - 1] for n in np.delete(np.arange(nTissueStart, N), without)])
                return Boundary(coordinates, normal)

    def get_domain_matrix(self, name):
        domain_matrix = self.domain_matrix
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        return np.array([[1 if domain_matrix[i][j] == name else 0 for j in range(M)] for i in range(N)])

    def get_domain_coordinates(self, name, internal=False):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        domain_matrix = self.domain_matrix

        coordinate = []
        if internal:
            for n in range(1, N - 1):
                for m in range(1, M - 1):
                    if ((domain_matrix[n][m] == name) &
                            (domain_matrix[n + 1][m] == name) & (domain_matrix[n - 1][m] == name) &
                            (domain_matrix[n][m + 1] == name) & (domain_matrix[n][m - 1] == name)):
                        coordinate += [[n, m]]
        else:
            for n in range(N):
                for m in range(M):
                    if domain_matrix[n][m] == name:
                        coordinate += [[n, m]]
        return np.array(coordinate)

    def get_tissue_shape(self):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        nTissueStart = self.nTissueStart
        return N - nTissueStart, M

    def get_air_shape(self):
        cnf = self.cnf
        M = cnf.M
        nTissueStart = self.nTissueStart
        return nTissueStart, M


class Boundary():
    def __init__(self, coordinates, normal):
        self.coordinates = coordinates
        self.normal = normal


class Properties():
    air_properties = {
        'eps': 1,
        'mobility': 10 ** (-10),
        'ion_concentration': 0,

        'absorbtion_koef': 0,
        'scattering_koef': 0,
        'scattering_anisotropy': 1,

        'density_native': 0,
        'thermal_conductivity': 0,
        'heat_capacity': 0,
    }

    tissue_properties_native = {
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

    tissue_properties_degraded = {
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

    tissue_properties_Arrenius = {
        'arr_velocity': 1,
        'arr_energy': 1
    }

    def __init__(self, cnf, geometry):
        self.cnf = cnf
        self.geometry = geometry

    def get_property_table(self, property_name, temperature, state):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        F = cnf.F
        R = cnf.R
        T0 = cnf.T0
        geometry = self.geometry
        # temperature = np.concatenate((np.zeros((geometry.get_air_shape())), temperature))
        # state = np.concatenate((np.zeros((geometry.get_air_shape())), state))

        # tissue_matrix = self.geometry.get_domain_matrix('tissue')
        # air_matrix = np.ones((N, M)) - tissue_matrix
        air_matrix = np.ones((geometry.get_air_shape()))
        tissue_matrix = np.ones((geometry.get_tissue_shape()))
        # TODO: create temperature and condition dependence
        air_properties = self.air_properties
        tissue_properties = self.tissue_properties_native

        if property_name == 'sigma':
            air_properties_matrix = F * F * air_properties['mobility'] * air_properties[
                'ion_concentration'] * air_matrix
            tissue_properties_matrix = F * F * tissue_properties['mobility'] * tissue_properties[
                'ion_concentration'] * tissue_matrix
        elif property_name == 'diff_ion':
            air_properties_matrix = R * air_properties['mobility'] * T0 * air_matrix
            tissue_properties_matrix = R * tissue_properties['mobility'] * temperature
        else:
            air_properties_matrix = air_properties[property_name] * air_matrix
            tissue_properties_matrix = tissue_properties[property_name] * tissue_matrix

        return np.concatenate((air_properties_matrix, tissue_properties_matrix))

    def get_property_table_unhomo(self, property, temperature, state):
        property_table_homo = self.get_property_table(property, temperature, state)
        property_table_unhomo = np.array(
            [0.25 * (property_table_homo[1:-1, 2:] + property_table_homo[1:-1, :-2] +
                     property_table_homo[2:, 1:-1] + property_table_homo[:-2, 1:-1]),
             0.5 * (property_table_homo[1:-1, 1:-1] + property_table_homo[:-2, 1:-1]),
             0.5 * (property_table_homo[1:-1, 1:-1] + property_table_homo[2:, 1:-1]),
             0.5 * (property_table_homo[1:-1, 1:-1] + property_table_homo[1:-1, :-2]),
             0.5 * (property_table_homo[1:-1, 1:-1] + property_table_homo[1:-1, 2:])]
        )
        return property_table_unhomo


class Conditions():
    def __init__(self, cnf, geometry, properties):
        self.cnf = cnf
        N = cnf.N
        M = cnf.M
        self.geometry = geometry
        nTissueStart = geometry.nTissueStart
        self.properties = properties

        self.termo_start = cnf.T0 * np.ones((N - nTissueStart, M))
        self.state_start = np.ones((N - nTissueStart, M))

        self.boundary_matrix_electrodiffusion = sp.coo_matrix(([], ([], [])), shape=(0, 2 * N * M))
        self.boundary_vector_electrodiffusion = np.array([])
        self.boundary_matrix_heat_transfer = sp.coo_matrix(([], ([], [])), shape=(0, (N - nTissueStart) * M))
        self.boundary_vector_heat_transfer = np.array([])
        self.boundary_matrix_light_transfer = sp.coo_matrix(([], ([], [])), shape=(0, (N - nTissueStart) * M))
        self.boundary_vector_light_transfer = np.array([])

    def get_start_temperature(self):
        return self.termo_start

    def get_start_state(self):
        return self.state_start

    def get_boundary_matrix_electrodiffusion(self):
        return self.boundary_matrix_electrodiffusion

    def Dirichlet(self, equa_name, bound, value):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        nTissueStart = self.geometry.nTissueStart
        coordinates = bound.coordinates

        #TODO vector len != matrix len

        equa_num = len(coordinates)
        row = np.arange(equa_num)
        col = np.array([coo[0] * M + coo[1] for coo in coordinates])
        data = np.ones(equa_num)

        if equa_name == 'potential':
            matrix = sp.coo_matrix((data, (row, col)), shape=(equa_num, 2 * N * M))
            vector = value
            self.boundary_matrix_electrodiffusion = sp.vstack([self.boundary_matrix_electrodiffusion, matrix])
            self.boundary_vector_electrodiffusion = np.concatenate([self.boundary_vector_electrodiffusion, vector])
        elif equa_name == 'charge':
            matrix = sp.coo_matrix((data, (row, col + N * M)), shape=(equa_num, 2 * N * M))
            vector = value
            self.boundary_matrix_electrodiffusion = sp.vstack([self.boundary_matrix_electrodiffusion, matrix])
            self.boundary_vector_electrodiffusion = np.concatenate([self.boundary_vector_electrodiffusion, vector])
        elif equa_name == 'heat_transfer':
            matrix = sp.coo_matrix((data, (row, col)), shape=(equa_num, (N - nTissueStart) * M))
            vector = value
            self.boundary_matrix_heat_transfer = sp.vstack([self.boundary_matrix_heat_transfer, matrix])
            self.boundary_vector_heat_transfer = np.concatenate([self.boundary_vector_heat_transfer, vector])
        elif equa_name == 'light_transfer':
            matrix = sp.coo_matrix((data, (row, col)), shape=(equa_num, (N - nTissueStart) * M))
            vector = value
            self.boundary_matrix_light_transfer = sp.vstack([self.boundary_matrix_light_transfer, matrix])
            self.boundary_vector_light_transfer = np.concatenate([self.boundary_vector_light_transfer, vector])
        else:
            raise ValueError('unknown equation name')

    def Neumann(self, equa_name, bound, value):
        pass

    def Newton(self, equa_name, bound, value, h='ones'):
        pass

    def Continious(self, equa_name, bound1, bound2, coef_values='ones', coef_differences='ones'):
        pass


class Sources():
    def __init__(self, cnf, geometry,
                 optic=lambda x, y, t: 0,
                 termal=lambda x, y, t: 0):
        self.cnf = cnf
        self.geometry = geometry


if __name__ == '__main__':
    cnf = Config(K=10, dk=10 ** (-6),
                 N=9, dn=10 ** (-6),
                 M=9, dm=10 ** (-6),
                 symmetry='decart',
                 T0=273)
    # print(cnf.N)

    geometry = Geometry(cnf, type="slice", nTissueStart=5)
    # print(geometry.domain_matrix)
    # print(geometry.get_bound('tissue', 'mEnd', without=[7]).coordinates)

    property = Properties(cnf, geometry)
    temp = cnf.T0 * np.ones((cnf.N, cnf.M))
    cond = np.ones((cnf.N, cnf.M))
    print(property.get_property_table('eps', temp, cond))
    print(property.get_property_table('sigma', temp, cond))
    print(property.get_property_table('diff_ion', temp, cond))
    # print(property.get_property_table_unhomo('eps', temp, cond))
    # print(geometry.get_domain_coordinates('tissue', internal=True))

    sources = Sources(cnf, geometry)
