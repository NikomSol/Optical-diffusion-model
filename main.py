from helper_classes import Config, Geometry, Properties, Conditions
from solver import Solver

cnf = Config(K=10, dk=10 ** (-6),
             N=9, dn=10 ** (-6),
             M=9, dm=10 ** (-6),
             symmetry='decart',
             T0=273)


def internal_area(n, m):
    return 'potata' if (n >= 4) else 'air'


geometry = Geometry(cnf, internal_area)
property = Properties(cnf, geometry)

m_electrode_1 = 3
m_electrode_2 = 5
bound_nStart_electrode_1 = geometry.get_bound('internal', 'nStart', only=[m_electrode_1])
bound_nStart_electrode_2 = geometry.get_bound('internal', 'nStart', only=[m_electrode_2])
bound_nStart_non_electrode = geometry.get_bound('internal', 'nStart', without=[m_electrode_1, m_electrode_2])

conditions = Conditions(
    cnf, geometry,

    termo_start=[Conditions.Start(cnf.T0)],

    termo_boundary=[Conditions.Dirichlet('external', cnf.T0), Conditions.Continuity('internal')],

    optic_boundaty=[Conditions.Dirichlet('internal', 'nStart', 0)],

    charge_boundary=[Conditions.Dirichlet('internal', 'nStart', 0)],

    potential_boundary=[Conditions.Dirichlet('internal', 'nStart', 0)]
)


def gauss(x, y):
    pass


sources = Sources(optic=gauss(x, y))

solution = Solver(cnf, geometry, property, conditions, sources)

solution.plot3D(z='T', y='n', x='k', par=[1])
solution.export(x='n', y='T', par=[1, 2], "result.csv")
