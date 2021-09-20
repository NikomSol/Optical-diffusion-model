import numpy as np
import math

"Параметры сетки"
cnf = {
    "N": 60,
    "dn": 1 * 10 ** (-5),
    "M": 60,
    "dm": 5 * 10 ** (-5),
    "K": 200,
    "dk": 3 * 10 ** (-4)
}
cnf['NAir'] = int(cnf['N'])
cnf["el1_M"] = int(1 * cnf['M'] / 3)
cnf['el2_M'] = int(2 * cnf['M'] / 3)

M = cnf['M']
N = cnf['N']
NAir = cnf['NAir']

"Свойства ткани и константы, каждое свойство - функция температуры и степени деградации"
properties = {
    # electro
    "eps": lambda T, g: 10 ** 5 * T / T,
    # "sigma": lambda T, g: (10 ** (0) * (1 - g) + 10 ** (-1) * g) * (1 +  0.02 * (T - 23)),
    "sigma": lambda T, g: 10 ** (-1) * (1 +  0.02 * (T - 23)),
    "diffusion": lambda T, g: 10 ** (-5) * T,
    # thermo
    "rho": 1095,
    "c": 3375,
    "kappa": 0.46,
    # opto 850 nm
    "mu_a": lambda T, g: 1400 * T / T,
    "mu_s": lambda T, g: (45 * (1 - g) + 10 * g) * 1000,
    "g_o": lambda T, g: 0.95 * T / T,
    "n1": 1,
    "n2": 1.34,
    # beam
    'lam': 0.850 * 10 ** (-6),
    "w0": 1 * 10 ** (-3),
    "Power": 20,
    # degrad
    "lnA": 161,
    "Tdeg": 80,
    # constant
    'eps0': 8.85 * 10 ** (-12)
}
mu_a = properties['mu_a']
mu_s = properties['mu_s']
g_o = properties['g_o']
properties["diffusion_opt"] = lambda T, g: 1 / (3 * (mu_a(T, g) + mu_s(T, g) * (1 - g_o(T, g))))


def pre_A(n1, n2):
    def rs(th):
        a = n1 * math.cos(th) - n2 * math.cos(math.asin(n1 / n2 * math.sin(th)))
        b = n1 * math.cos(th) + n2 * math.cos(math.asin(n1 / n2 * math.sin(th)))
        return a / b

    def rp(th):
        a = n2 * math.cos(th) - n1 * math.cos(math.asin(n1 / n2 * math.sin(th)))
        b = n2 * math.cos(th) + n1 * math.cos(math.asin(n1 / n2 * math.sin(th)))
        return a / b

    def R(th):
        return 0.5 * abs(rs(th)) ** 2 + 0.5 * abs(rp(th)) ** 2

    def R_f(th):
        return 2 * math.sin(th) * math.cos(th) * R(math.cos(th))

    R_f = np.vectorize(R_f)

    def R_J(th):
        return 3 * math.sin(th) * math.cos(th) ** 2 * R(math.cos(th))

    R_J = np.vectorize(R_J)

    th = np.arange(0, math.pi / 2, math.pi / 2000)
    R_f = np.trapz(R_f(th), dx=math.pi / 2000)
    R_J = np.trapz(R_J(th), dx=math.pi / 2000)

    return 2 * (1 + R_J) / (1 - R_f)


properties['pre_A'] = pre_A(properties['n1'], properties['n2'])

"Начальные условия в ткани"
condition_start = {
    'T': 23 * np.ones((N, M)),
    'g': 1 * np.ones((N, M))
}


class Boundary():
    """
    Класс граничных условий.
    объект класса содержит в себе информация о граничном условии на одной определенной границе
    """

    def __init__(self, variable_name, condition_type, coordinates, external_normal, coefficients):
        """
        :param variable_name: Имя переменной к которой относится условие T,q,u...
        :param condition_type: Тип условия 'Dirichlet', 'Newmann', 'Newton', 'Continiosly'
        :param coordinates: Список координат точек в которых задано условие
        :param external_normal: Направление внешней нормали к данной границе (нужно для задания производной)
        :param coefficients: Список необходимых для рассчета коэффициентов, например значения переменной или
        производной в данной точке
        """
        self.variable_name = variable_name
        self.condition_type = condition_type
        self.coefficients = coefficients
        self.coordinates = coordinates
        self.external_normal = external_normal


"Граничные условия в ткани"
coo_n_start = np.array([[0, m] for m in range(M)])  # Координаты границы z = 0
coo_n_stop = np.array([[N - 1, m] for m in range(M)])  # Координаты границы z = Z
coo_m_start = np.array([[n, 0] for n in range(0, N)])  # Координаты границы r = 0
coo_m_stop = np.array([[n, M - 1] for n in range(0, N)])  # Координаты границы r = R

condition_boundary_termal = dict(rmax=[1., 0., 23.],
                                 rmin=[0., 1., 0.],
                                 zmax=[1., 0., 23.],
                                 zmin=[-10., 1, -10. * 23])


# def condition_boundary_termal(T,g):
# return [Boundary('T', 'Newton', coo_n_start, [-1, 0], [np.ones(M), np.ones(M)]),
# Boundary('T', 'Dirichlet', coo_n_stop, [1, 0], np.full(M,273)),
# Boundary('T', 'Newmann', coo_m_start[1:-1], [0, -1], np.zeros(N - 2)),
# Boundary('T', 'Dirichlet', coo_m_stop[1:-1], [0, 1], np.zeros(N - 2))]


def condition_boundary_optic(T, g):
    return dict(rmax=[0., 1., 0.],
                rmin=[0., 1., 0.],
                zmax=[1., 0., 0.],
                zmin=[1., properties['pre_A'] * properties['diffusion_opt'](T, g)[0], 0.])
    # Boundary('I', 'Newton', coo_n_start, [-1, 0], [np.ones(M), np.ones(M)]),
    # Boundary('I', 'Dirichlet', coo_n_stop, [1, 0], np.zeros(M)),
    # Boundary('I', 'Newmann', coo_m_start[1:-1], [0, -1], np.zeros(N - 2)),
    # Boundary('I', 'Dirichlet', coo_m_stop[1:-1], [0, 1], np.zeros(N - 2))]


coo_air_n_start = np.array([[0, m] for m in range(M)])
coo_air_n_stop = np.array([[NAir - 1, m] for m in range(M)])
coo_air_m_start = np.array([[n, 0] for n in range(0, NAir)])
coo_air_m_stop = np.array([[n, M - 1] for n in range(0, NAir)])

coo_tissue_n_start = np.array([[NAir, m] for m in range(M)])
coo_tissue_n_stop = np.array([[N + NAir - 1, m] for m in range(M)])
coo_tissue_m_start = np.array([[n, 0] for n in range(NAir, N + NAir)])
coo_tissue_m_stop = np.array([[n, M - 1] for n in range(NAir, N + NAir)])

el1_M = cnf["el1_M"]
el2_M = cnf["el2_M"]
ne_M = np.delete(np.arange(M), el1_M)
coo_air_el1 = np.array([[NAir - 1, el1_M]])
# coo_air_el2 = np.array([[NAir - 1, el2_M]])
coo_air_n_stop_ne = np.array([[NAir - 1, m] for m in ne_M])
coo_tissue_el1 = np.array([[NAir, el1_M]])
# coo_tissue_el2 = np.array([[NAir, el2_M]])
coo_tissue_n_start_ne = np.array([[NAir, m] for m in ne_M])


def condition_boundary_electric(T, g):
    return [
        Boundary('u', 'Dirichlet', coo_air_n_start, [-1, 0], np.zeros(M)),
        Boundary('u', 'Newmann', coo_air_m_start[1:-1], [0, -1], np.zeros(NAir - 2)),
        Boundary('u', 'Newmann', coo_air_m_stop[1:-1], [0, 1], np.zeros(NAir - 2)),

        Boundary('q', 'Dirichlet', coo_air_n_start, [-1, 0], np.zeros(M)),
        Boundary('q', 'Newmann', coo_air_m_start[1:-1], [0, -1], np.zeros(NAir - 2)),
        Boundary('q', 'Newmann', coo_air_m_stop[1:-1], [0, 1], np.zeros(NAir - 2)),

        Boundary('u', 'Dirichlet', coo_tissue_n_stop, [1, 0], np.zeros(M)),
        Boundary('u', 'Newmann', coo_tissue_m_start[1:-1], [0, -1], np.zeros(N - 2)),
        Boundary('u', 'Newmann', coo_tissue_m_stop[1:-1], [0, 1], np.zeros(N - 2)),

        Boundary('q', 'Dirichlet', coo_tissue_n_stop, [1, 0], np.zeros(M)),
        Boundary('j', 'Newmann', coo_tissue_m_start[1:-1], [0, -1], np.zeros(N - 2)),
        Boundary('j', 'Newmann', coo_tissue_m_stop[1:-1], [0, 1], np.zeros(N - 2)),

        # bound air-tissue
        Boundary('u', 'Continiosly', [coo_air_n_stop_ne, coo_tissue_n_start_ne], [[0, 1], [0, -1]],
                 [np.ones(M - 1), properties['eps'](T, g)[0, ne_M]]),
        Boundary('u', 'Dirichlet', coo_air_el1, [1, 0], [1]),
        # Boundary('u', 'Dirichlet', coo_air_el2, [1, 0], [-1]),
        Boundary('u', 'Dirichlet', coo_tissue_el1, [-1, 0], [1]),
        # Boundary('u', 'Dirichlet', coo_tissue_el2, [-1, 0], [-1]),

        Boundary('q', 'Dirichlet', coo_air_n_stop, [0, 1], np.zeros(M)),
        Boundary('j', 'Newmann', coo_tissue_n_start, [-1, 0], np.zeros(M))
    ]
# el1_M = cnf["el1_M"]
# el2_M = cnf["el2_M"]
# ne_M = np.delete(np.arange(M), [el1_M, el2_M])
# coo_air_el1 = np.array([[NAir - 1, el1_M]])
# coo_air_el2 = np.array([[NAir - 1, el2_M]])
# coo_air_n_stop_ne = np.array([[NAir - 1, m] for m in ne_M])
# coo_tissue_el1 = np.array([[NAir, el1_M]])
# coo_tissue_el2 = np.array([[NAir, el2_M]])
# coo_tissue_n_start_ne = np.array([[NAir, m] for m in ne_M])
# def condition_boundary_electric(T, g):
#     return [
#         Boundary('u', 'Dirichlet', coo_air_n_start, [-1, 0], np.zeros(M)),
#         Boundary('u', 'Newmann', coo_air_m_start[1:-1], [0, -1], np.zeros(NAir - 2)),
#         Boundary('u', 'Dirichlet', coo_air_m_stop[1:-1], [0, 1], np.zeros(NAir - 2)),
#
#         Boundary('q', 'Dirichlet', coo_air_n_start, [-1, 0], np.zeros(M)),
#         Boundary('q', 'Newmann', coo_air_m_start[1:-1], [0, -1], np.zeros(NAir - 2)),
#         Boundary('q', 'Dirichlet', coo_air_m_stop[1:-1], [0, 1], np.zeros(NAir - 2)),
#
#         Boundary('u', 'Dirichlet', coo_tissue_n_stop, [1, 0], np.zeros(M)),
#         Boundary('u', 'Newmann', coo_tissue_m_start[1:-1], [0, -1], np.zeros(N - 2)),
#         Boundary('u', 'Dirichlet', coo_tissue_m_stop[1:-1], [0, 1], np.zeros(N - 2)),
#
#         Boundary('q', 'Dirichlet', coo_tissue_n_stop, [1, 0], np.zeros(M)),
#         Boundary('j', 'Newmann', coo_tissue_m_start[1:-1], [0, -1], np.zeros(N - 2)),
#         Boundary('j', 'Newmann', coo_tissue_m_stop[1:-1], [0, 1], np.zeros(N - 2)),
#
#         # bound air-tissue
#         Boundary('u', 'Continiosly', [coo_air_n_stop_ne, coo_tissue_n_start_ne], [[0, 1], [0, -1]],
#                  [np.ones(M - 2), properties['eps'](T, g)[0, 1:-1]]),
#         Boundary('u', 'Dirichlet', coo_air_el1, [1, 0], [1]),
#         Boundary('u', 'Dirichlet', coo_air_el2, [1, 0], [-1]),
#         Boundary('u', 'Dirichlet', coo_tissue_el1, [-1, 0], [1]),
#         Boundary('u', 'Dirichlet', coo_tissue_el2, [-1, 0], [-1]),
#
#         Boundary('q', 'Dirichlet', coo_air_n_stop, [0, 1], np.zeros(M)),
#         Boundary('j', 'Newmann', coo_tissue_n_start, [-1, 0], np.zeros(M))
#     ]

# def condition_boundary_electric(T,g):
#     return [
#         Boundary('u','Dirichlet',coo_air_n_start,[-1,0],np.zeros(M)),
#         Boundary('u','Dirichlet',coo_air_m_start[1:-1],[0,-1],np.zeros(NAir-2)),
#         Boundary('u','Dirichlet',coo_air_m_stop[1:-1],[0,1],np.zeros(NAir-2)),
#
#         Boundary('q','Dirichlet',coo_air_n_start,[-1,0],np.zeros(M)),
#         Boundary('q','Dirichlet',coo_air_m_start[1:-1],[0,-1],np.zeros(NAir-2)),
#         Boundary('q','Dirichlet',coo_air_m_stop[1:-1],[0,1],np.zeros(NAir-2)),
#
#         Boundary('u','Newmann',coo_tissue_n_stop,[1,0],np.zeros(M)),
#         Boundary('u','Dirichlet',coo_tissue_m_start[1:-1],[0,-1],np.ones(N-2)),
#         Boundary('u','Dirichlet',coo_tissue_m_stop[1:-1],[0,1],-np.ones(N-2)),
#
#         Boundary('j','Newmann',coo_tissue_n_stop,[1,0],np.zeros(M)),
#         Boundary('j','Newmann',coo_tissue_m_start[1:-1],[0,-1],np.zeros(N-2)),
#         Boundary('j','Newmann',coo_tissue_m_stop[1:-1],[0,1],np.zeros(N-2)),
#
#         Boundary('u','Continiosly',[coo_air_n_stop,coo_tissue_n_start],[[0,1],[0,-1]],
#                  [np.ones(M),properties['eps'](T,g)[0]]),
#         Boundary('q','Dirichlet',coo_air_n_stop,[0,1],np.zeros(M)),
#         Boundary('j','Newmann',coo_tissue_n_start,[-1,0],np.zeros(M))
#     ]
