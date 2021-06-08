import numpy as np
import scipy.sparse as sp


class Config():
    eps0 = 8.85 * 10 ** (-12),
    R = 8.31

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

        if domain is 'air':
            if bound is 'nStart':
                normal = [-1, 0]
                if only:
                    coordinates = np.array([[0, m] for m in only])
                else:
                    coordinates = np.array([[0, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)
            if bound is 'nEnd':
                normal = [1, 0]
                if only:
                    coordinates = np.array([[nTissueStart - 1, m] for m in only])
                else:
                    coordinates = np.array([[nTissueStart - 1, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)

            if bound is 'mStart':
                normal = [0, -1]
                if only:
                    coordinates = np.array([[n, 0] for n in only])
                else:
                    coordinates = np.array([[n, 0] for n in np.delete(np.arange(nTissueStart), without)])
                return Boundary(coordinates, normal)
            if bound is 'mEnd':
                normal = [0, 1]
                if only:
                    coordinates = np.array([[n, M - 1] for n in only])
                else:
                    coordinates = np.array([[n, M - 1] for n in np.delete(np.arange(nTissueStart), without)])
                return Boundary(coordinates, normal)

        if domain is 'tissue':
            if bound is 'nStart':
                normal = [-1, 0]
                if only:
                    coordinates = np.array([[nTissueStart, m] for m in only])
                else:
                    coordinates = np.array([[nTissueStart, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)
            if bound is 'nEnd':
                normal = [1, 0]
                if only:
                    coordinates = np.array([[N - 1, m] for m in only])
                else:
                    coordinates = np.array([[N - 1, m] for m in np.delete(np.arange(M), without)])
                return Boundary(coordinates, normal)

            if bound is 'mStart':
                normal = [0, -1]
                if only:
                    coordinates = np.array([[n, 0] for n in only])
                else:
                    coordinates = np.array([[n, 0] for n in np.delete(np.arange(nTissueStart, N), without)])
                return Boundary(coordinates, normal)
            if bound is 'mEnd':
                normal = [0, 1]
                if only:
                    coordinates = np.array([[n, M - 1] for n in only])
                else:
                    coordinates = np.array([[n, M - 1] for n in np.delete(np.arange(nTissueStart, N), without)])
                return Boundary(coordinates, normal)

    def get_domain(self, name):
        domain_matrix = self.domain_matrix
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        return np.array([[1 if domain_matrix[i][j] == name else 0 for j in range(M)] for i in range(N)])


class Boundary():
    def __init__(self, coordinates, normal):
        self.coordinates = coordinates
        self.normal = normal


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

    def get_property_table(self, property, temperature, condition):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M

        tissue_matrix = self.geometry.get_domain('tissue')
        air_matrix = np.ones((N, M)) - tissue_matrix
        # TODO: create temperature and condition dependence
        air_properties = self.air_properties[property] * condition
        tissue_properties = self.tissue_properties_native[property] * condition

        return tissue_matrix * tissue_properties + air_properties * air_matrix


class Conditions():
    def __init__(self, cnf, geometry,
                 termo_start=[],
                 termo_boundary=[], optic_boundaty=[], charge_boundary=[], potential_boundary=[]):
        self.cnf = cnf
        self.geometry = geometry

    @classmethod
    def Start(self, domain, value):
        pass

    @classmethod
    def Dirichlet(self, bound, value):
        pass

    @classmethod
    def Neumann(self, bound, value):
        pass

    @classmethod
    def Newton(self, bound, value, h='ones'):
        pass

    @classmethod
    def Continious(self, bound1, bound2, coef_values='ones', coef_differences='ones'):
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

    sources = Sources(cnf, geometry)
