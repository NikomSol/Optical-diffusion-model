import numpy as np
import scipy.sparse as sp


class Solver():
    def __init__(self, cnf, geometry, property, conditions, sources):
        self.cnf = cnf
        N = cnf.N
        M = cnf.M

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
                result['temp'] = np.concatenate([result['temp'],[self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'],[self.step_degradation()]])
        elif equation == {'O', 'T'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            result['I'] = np.array([self.solve_light_transfer()])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'],[self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'],[self.step_degradation()]])
                result['I'] = np.concatenate([result['I'],[self.step_degradation()]])
        elif equation == {'E', 'T'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            buf_u, buf_v = self.solve_electrodiffusion()
            result['u'] = np.array([buf_u])
            result['v'] = np.array([buf_v])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'],[self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'],[self.step_degradation()]])
                buf_u, buf_v = self.solve_electrodiffusion()
                result['u'] = np.concatenate([result['u'],[buf_u]])
                result['v'] = np.concatenate([result['v'],[buf_v]])
        elif equation == {'E', 'T', 'O'}:
            result['temp'] = np.array([conditions.get_start_temperature()])
            result['state'] = np.array([conditions.get_start_state()])
            result['I'] = np.array([self.solve_light_transfer()])
            buf_u, buf_v = self.solve_electrodiffusion()
            result['u'] = np.array([buf_u])
            result['v'] = np.array([buf_v])
            for k in range(K - 1):
                result['temp'] = np.concatenate([result['temp'],[self.step_heat_transfer()]])
                result['state'] = np.concatenate([result['state'],[self.step_degradation()]])
                result['I'] = np.concatenate([result['I'],[self.step_degradation()]])
                buf_u, buf_v = self.solve_electrodiffusion()
                result['u'] = np.concatenate([result['u'],[buf_u]])
                result['v'] = np.concatenate([result['v'],[buf_v]])

    def solve_electrodiffusion(self):
        cnf = self.cnf
        N = cnf.N
        M = cnf.M
        return np.zeros((N, M)), np.zeros((N, M))

    def solve_light_transfer(self):
        return np.zeros((self.geometry.get_tissue_shape()))

    def step_heat_transfer(self):
        return np.zeros((self.geometry.get_tissue_shape()))

    def step_degradation(self):
        return np.zeros((self.geometry.get_tissue_shape()))

    def plot3D(self, z='T', y='n', x='k', slices=[1]):
        pass

    def export(self, x='n', y='T', slices=[1, 2], file_name="result.csv"):
        pass
