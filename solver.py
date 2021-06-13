import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class Solver():
    def __init__(self, cnf, geometry, property, conditions, sources):
        self.cnf = cnf
        # N = cnf.N
        # M = cnf.M

        self.geometry = geometry
        # tissue_N, tissue_M = geometry.get_tissue_shape()

        self.property = property
        self.conditions = conditions
        self.sources = sources
        self.result = {}

    def solve(self, equation={'O', 'T', 'E'}):
        K = self.cnf.K
        result = self.result
        conditions = self.conditions
        if equation == {'O'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            result['I'] = np.array([self.solve_light_transfer()])
        elif equation == {'E'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            buf_u, buf_v = self.solve_electrodiffusion()
            result['u'] = np.array([buf_u])
            result['v'] = np.array([buf_v])
        elif equation == {'T'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'], [self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'], [self.step_degradation()]])
        elif equation == {'O', 'T'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            result['I'] = np.array([self.solve_light_transfer()])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'], [self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'], [self.step_degradation()]])
                result['I'] = np.concatenate([result['I'], [self.step_degradation()]])
        elif equation == {'E', 'T'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            buf_u, buf_v = self.solve_electrodiffusion()
            result['u'] = np.array([buf_u])
            result['v'] = np.array([buf_v])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'], [self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'], [self.step_degradation()]])
                buf_u, buf_v = self.solve_electrodiffusion()
                result['u'] = np.concatenate([result['u'], [buf_u]])
                result['v'] = np.concatenate([result['v'], [buf_v]])
        elif equation == {'E', 'T', 'O'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            result['I'] = np.array([self.solve_light_transfer()])
            buf_u, buf_v = self.solve_electrodiffusion()
            result['u'] = np.array([buf_u])
            result['v'] = np.array([buf_v])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'], [self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'], [self.step_degradation()]])
                result['I'] = np.concatenate([result['I'], [self.step_degradation()]])
                buf_u, buf_v = self.solve_electrodiffusion()
                result['u'] = np.concatenate([result['u'], [buf_u]])
                result['v'] = np.concatenate([result['v'], [buf_v]])

    def solve_electrodiffusion(self):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        system_matrix = self.create_electrodiffusion_system_matrix()
        free_vector = self.create_electrodiffusion_free_vector()

        uv = spsolve(system_matrix.tocsr(), free_vector).reshape(2, N, M)
        return uv[0], uv[1]

    def create_electrodiffusion_system_matrix(self):
        cnf = self.cnf
        eps0 = cnf.eps0
        div_eps0 = 1 / eps0
        N = cnf.N
        M = cnf.M

        geometry = self.geometry
        domain_air_coordinates = geometry.get_domain_coordinates('air', internal=True)
        domain_tissue_coordinates = geometry.get_domain_coordinates('tissue', internal=True)

        result = self.result
        temp = result['temp'][-1]
        status = result['state'][-1]

        property = self.property
        eps = property.get_property_table('eps', temp, status)
        sigma = property.get_property_table('sigma', temp, status)
        diff_ion = property.get_property_table_unhomo('diff_ion', temp, status)

        equa_num = len(domain_air_coordinates)
        row = np.arange(equa_num)
        col = np.array([coo[0] * M + coo[1] for coo in domain_air_coordinates])
        data = np.ones(equa_num)
        data_plus = div_eps0 / eps[domain_air_coordinates[:, 0], domain_air_coordinates[:, 1]]

        poison_air_matrix = (
                sp.coo_matrix((-4 * data, (row, col)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col + 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col - 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col + M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col - M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data_plus, (row, col + M * N)), shape=(equa_num, 2 * N * M))
        )

        equa_num = len(domain_air_coordinates)
        row = np.arange(equa_num)
        col = np.array([coo[0] * M + coo[1] for coo in domain_air_coordinates]) + M * N
        data = diff_ion[:, domain_air_coordinates[:, 0] - 1, domain_air_coordinates[:, 1] - 1]
        data_plus = div_eps0 * (sigma[domain_air_coordinates[:, 0], domain_air_coordinates[:, 1]] /
                                eps[domain_air_coordinates[:, 0], domain_air_coordinates[:, 1]])
        germgolc_air_matrix = (
                sp.coo_matrix((-4 * data[0] - data_plus, (row, col)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[4], (row, col + 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[3], (row, col - 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[2], (row, col + M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[1], (row, col - M)), shape=(equa_num, 2 * N * M))
        )

        equa_num = len(domain_tissue_coordinates)
        row = np.arange(equa_num)
        col = np.array([coo[0] * M + coo[1] for coo in domain_tissue_coordinates])
        data = np.ones(equa_num)
        data_plus = div_eps0 / eps[domain_tissue_coordinates[:, 0], domain_tissue_coordinates[:, 1]]

        poison_tissue_matrix = (
                sp.coo_matrix((-4 * data, (row, col)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col + 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col - 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col + M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data, (row, col - M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data_plus, (row, col + M * N)), shape=(equa_num, 2 * N * M))
        )

        equa_num = len(domain_tissue_coordinates)
        row = np.arange(equa_num)
        col = np.array([coo[0] * M + coo[1] for coo in domain_tissue_coordinates]) + M * N
        data = diff_ion[:, domain_tissue_coordinates[:, 0] - 1, domain_tissue_coordinates[:, 1] - 1]
        data_plus = div_eps0 * (sigma[domain_tissue_coordinates[:, 0], domain_tissue_coordinates[:, 1]] /
                                eps[domain_tissue_coordinates[:, 0], domain_tissue_coordinates[:, 1]])
        germgolc_tissue_matrix = (
                sp.coo_matrix((-4 * data[0] - data_plus, (row, col)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[4], (row, col + 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[3], (row, col - 1)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[2], (row, col + M)), shape=(equa_num, 2 * N * M)) +
                sp.coo_matrix((data[1], (row, col - M)), shape=(equa_num, 2 * N * M))
        )

        system_matrix = sp.vstack(
            [poison_air_matrix, germgolc_air_matrix, poison_tissue_matrix, germgolc_tissue_matrix])
        return sp.dia_matrix((np.ones(2 * N * M), 0), shape=(2 * N * M, 2 * N * M))

    def create_electrodiffusion_free_vector(self):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        return np.ones(2 * N * M)

    def solve_light_transfer(self):
        return np.zeros((self.geometry.get_tissue_shape()))

    def step_heat_transfer(self):
        return self.result['temp'][-1]

    def step_degradation(self):
        return self.result['state'][-1]

    def plot3D(self, z='T', y='n', x='k', slices=[1]):
        pass

    def export(self, x='n', y='T', slices=[1, 2], file_name="result.csv"):
        pass
