from helper_classes import Config, Geometry, Properties, Conditions, Sources
from solver import Solver
import numpy as np

cnf = Config(K=10, dk=10 ** (-6),
             N=9, dn=10 ** (-6),
             M=9, dm=10 ** (-6),
             symmetry='decart',
             T0=273)
N = cnf.N
M = cnf.M

# 'slice' -  воздух, потом ткань по слоям, 'internal' - одна область в другой
nTissueStart = 5
geometry = Geometry(cnf, type="slice", nTissueStart=nTissueStart)

# Границы задаются в отдельном формате, можно просто как список координат и вектор нормали
bound_air_nStart = geometry.get_bound('air', 'nStart')
bound_air_nEnd = geometry.get_bound('air', 'nEnd')
bound_air_mStart = geometry.get_bound('air', 'mStart')
bound_air_mEnd = geometry.get_bound('air', 'mEnd')

bound_tissue_nStart = geometry.get_bound('tissue', 'nStart')
bound_tissue_nEnd = geometry.get_bound('tissue', 'nEnd')
bound_tissue_mStart = geometry.get_bound('tissue', 'mStart')
bound_tissue_mEnd = geometry.get_bound('tissue', 'mEnd')

bound_air_electrode_1 = geometry.get_bound('air', 'nEnd', only=[3])
bound_air_electrode_2 = geometry.get_bound('air', 'nEnd', only=[5])
bound_air_non_electrode = geometry.get_bound('air', 'nEnd', without=[3, 5])
bound_tissue_electrode_1 = geometry.get_bound('tissue', 'nStart', only=[3])
bound_tissue_electrode_2 = geometry.get_bound('tissue', 'nStart', only=[5])
bound_tissue_non_electrode = geometry.get_bound('tissue', 'nStart', without=[3, 5])

domain_tissue = geometry.get_domain_matrix('tissue')
# Свойства различны у разных областей
property = Properties(cnf, geometry)

# Граничные условия могут быть Дирихле, Нейман, Ньютон-Рихман, Непрерывность
conditions = Conditions(cnf, geometry, property)

conditions.termo_start = cnf.T0 * np.ones((geometry.get_tissue_shape()))

conditions.state_start = np.ones((geometry.get_tissue_shape()))

conditions.Newton('heat_transfer', bound_tissue_nStart, cnf.T0 * np.ones(M), h='h')
conditions.Neumann('heat_transfer', bound_tissue_nEnd, np.zeros(M))
conditions.Neumann('heat_transfer', bound_tissue_mStart, np.zeros(N))
conditions.Neumann('heat_transfer', bound_tissue_mEnd, np.zeros(N))

conditions.Newton('light_transfer', bound_tissue_nStart, np.zeros(M), h='A')
conditions.Newton('light_transfer', bound_tissue_nEnd, np.zeros(M), h='D')
conditions.Neumann('light_transfer', bound_tissue_mStart, np.zeros(N))
conditions.Newton('light_transfer', bound_tissue_mEnd, np.zeros(N), h='D')

conditions.Neumann('charge', bound_tissue_nStart, np.zeros(M))
conditions.Dirichlet('charge', bound_tissue_nEnd, np.zeros(M))
conditions.Neumann('charge', bound_tissue_mStart, np.zeros(N))
conditions.Dirichlet('charge', bound_tissue_mEnd, np.zeros(N))

conditions.Continious('potential', bound_tissue_non_electrode, bound_air_non_electrode, 'ones', 'eps')
conditions.Dirichlet('potential', bound_tissue_electrode_1, [1])
conditions.Dirichlet('potential', bound_air_electrode_1, [1])
conditions.Dirichlet('potential', bound_tissue_electrode_2, [-1])
conditions.Dirichlet('potential', bound_air_electrode_2, [-1])

conditions.Neumann('potential', bound_tissue_nEnd, np.zeros(M))
conditions.Neumann('potential', bound_tissue_mStart, np.zeros(N))
conditions.Neumann('potential', bound_tissue_mEnd, np.zeros(N))

conditions.Neumann('potential', bound_air_nStart, np.zeros(M))
conditions.Neumann('potential', bound_air_mStart, np.zeros(N))
conditions.Neumann('potential', bound_air_mEnd, np.zeros(N))


def gauss(x, y, t):
    # TODO: gauss function
    return x + y


sources = Sources(cnf, geometry, optic=gauss)

solution = Solver(cnf, geometry, property, conditions, sources)
# equations {'E': elecrtodiffusion, 'O': light transfer, 'T': heat transfer}
solution.solve(equation={'E'})

print(solution.result['u'])

solution.plot3D(z='T', y='n', x='k', slices=[1])
solution.export(x='n', y='T', slices=[1, 2], file_name="result.csv")
