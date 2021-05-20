import numpy as np
from diffusion_equation import DiffusionEquation

cnf = {
    'K': 3,
    'N': 10,
    'M': 3,
    'dk': 0.1,
    'dn': 0.1,
    'dm': 0.1
}
properties = 0.6 / 4200 / 1000 * np.ones((cnf['K'], cnf['N'], cnf['M']))
sources = 0 * np.ones((cnf['K'], cnf['N'], cnf['M']))
start_conditions = np.ones((cnf['N'], cnf['M']))
bound_conditions = cnf['K'] * [
    {'Nstart': cnf['M'] * [[1, 0, 1]],
     'Nend': cnf['M'] * [[1, 0, 1]],
     'Mstart': cnf['N'] * [[1, 0, 1]],
     'Mend': cnf['N'] * [[1, 0, 1]]}
]

properties_static_1D = 0.6 / 4200 / 1000 * np.ones(cnf['N'])
sources_static_1D = 0 * (10 ** (-6)) * np.ones(cnf['N'])
bound_conditions_static_1D = {'Nstart': [1, -1, 0], 'Nend': [1, 0, 1]}

solution_static_1D = DiffusionEquation.static_1D(cnf, properties_static_1D, bound_conditions_static_1D,
                                                 sources_static_1D)
print(solution_static_1D)

# properties_static_2D = 0.6 / 4200 / 1000 * np.ones(cnf['N'], cnf['M'])
# sources_static_2D = 0 * (10 ** (-6)) * np.ones(cnf['N'], cnf['M'])
# bound_conditions_static_2D = {
#     'Nstart': cnf['M'] * [[1, 0, 1]],
#     'Nend': cnf['M'] * [[1, 0, 1]],
#     'Mstart': cnf['N'] * [[1, 0, 1]],
#     'Mend': cnf['N'] * [[1, 0, 1]]}
#
# solution_static_2D = DiffusionEquation.static_2D(cnf, properties_static_2D, bound_conditions_static_2D,
#                                                  sources_static_2D)
# print(solution_static_2D)
