import numpy as np
from diffusion_equation import DiffusionEquation
from electrodiffusion_equation import ElectroDiffusionEquation
import matplotlib.pyplot as plt

cnf = {
    'K': 3,
    'N': 4*10 ** 0,
    'M': 3,
    'dk': 0.1,
    'dn': 10 ** (-10),
    'dm': 0.1,

    'eps0': 8.85 * 10 ** (-12)
}
properties = {
    'sigma': np.ones(cnf['N']) * 10 ** (-3),
    'epsilon': np.ones(cnf['N']) * 1,
    'diffusion': np.ones(cnf['N']) * 10 ** (-9)
}

# sources = 0 * np.ones((cnf['K'], cnf['N'], cnf['M']))
# start_conditions = np.ones((cnf['N'], cnf['M']))

bound_conditions = {
    'N_start_potential': [1, 0, 0],
    'N_end_potential': [1, 0, 1],
    'N_start_charge': [1, 0, 0],
    'N_end_charge': [0, 1, 0]
}

solution_static_1D = ElectroDiffusionEquation.static_1D_real(cnf, properties, bound_conditions)
u, v = solution_static_1D[:cnf['N']], solution_static_1D[cnf['N']:]

# print(v[0])
# print(v[-1])

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(u)
plt.subplot(2, 1, 2)
plt.plot(v)
plt.show()
