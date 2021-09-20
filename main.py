from input import cnf, properties, condition_start  # dict
from input import condition_boundary_optic, condition_boundary_electric, condition_boundary_termal  # func(T,g)
from combiner import solver, step_solver, calc_impedance

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

N = cnf['N']
M = cnf['M']
K = cnf['K']
dn = cnf['dn']
dm = cnf['dm']
dk = cnf['dk']

# npsol_step = step_solver(cnf, properties,
#                        condition_start['T'],
#                        condition_start['g'],
#                        condition_boundary_optic,
#                        condition_boundary_electric,
#                        condition_boundary_termal)

# sol_all = solver(cnf, properties,
#                  condition_start,
#                  condition_boundary_optic,
#                  condition_boundary_electric,
#                  condition_boundary_termal)

# for k in range(K):
#     plt.matshow(sol_all['g'][k])
#     plt.colorbar()


# print(sol_all['u'][:,cnf['NAir'],cnf['el2_M']])
# plt.plot(sol_all['u'][-1,:,cnf['el1_M']])
# plt.show()

timeGrid = dk * np.arange(K+1)
# u_t = sol_all['u'][:, cnf['NAir'], cnf['el2_M']]
#
# df = pd.DataFrame(np.array([timeGrid, u_t]).transpose())
# df.to_csv('u(t) T.csv', index=False)

df_u_Tg = pd.read_csv('u(t) Tg.csv', delimiter=',')
df_u_T = pd.read_csv('u(t) T.csv', delimiter=',')
df_u_g = pd.read_csv('u(t) g.csv', delimiter=',')

plt.plot(timeGrid, df_u_Tg['1'])
plt.plot(timeGrid, df_u_T['1'])
plt.plot(timeGrid, df_u_g['1'])
plt.show()
