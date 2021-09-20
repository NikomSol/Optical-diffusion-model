import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


def solve(cnf, properties, T, g, bound_electric):
    """Пользуюсь далее только свойствами электрическими причем от функций перехожу к матрицам,
    рассчитанным по известным температурам и степеням деградации"""
    properties_local = {
        'eps': properties['eps'](T, g),
        'sigma': properties['sigma'](T, g),
        'diffusion': properties['diffusion'](T, g),
        'eps0': properties['eps0']
    }

    "Рассчитываю матрицу и свободный вектор для уравнений внутренних областей"
    system_matrix = get_system_matrix_internal_cylinde(cnf, properties_local)
    free_vector = np.zeros(system_matrix.shape[0])

    "Добавляю уравнения для граничных областей"
    for bound_i in bound_electric(T, g):
        matrix, vector = get_bound_matrix(cnf, properties_local, bound_i)
        system_matrix = sp.vstack([system_matrix, matrix])
        free_vector = np.concatenate([free_vector, vector])

    "Решаем все вместе"
    uq = spsolve(system_matrix.tocsr(), free_vector).reshape(2, cnf['N'] + cnf['NAir'], cnf['M'])
    return uq[0], uq[1]


def get_system_matrix_internal_cylinde(cnf, properties):
    # creating poison_air_matrix
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    dn = cnf['dn']
    dm = cnf['dm']

    div_dn = 1 / dn
    div_dm = 1 / dm

    eps0 = properties['eps0']
    div_eps0 = 1 / eps0

    eps1 = 1 * np.ones((NAir, M))
    # sigma1 = 0 * np.ones((NAir, M))
    # diff1 = 0 * np.ones((NAir, M))

    eps2 = properties['eps']
    sigma2 = properties['sigma']
    diff2 = properties['diffusion']

    equa_num = (NAir - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(1, NAir - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    dataM1 = np.array([1. / (m + 1) for m in range(equa_num)])
    data2 = (div_eps0 / eps1[1:-1, 1: - 1]).reshape(equa_num)

    poison_air_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1, (row, col)),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1 + dataM1), (row, col + 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1 - dataM1), (row, col - 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col + M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col - M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))
    )

    # creating germgolc_air_matrix (v = 0)
    germgolc_air_matrix = sp.coo_matrix((data1, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))

    # creating poison_tissue_matrix
    equa_num = (N - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(NAir + 1, (N + NAir) - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    dataM1 = np.array([1. / (m + 1) for m in range(equa_num)])
    data2 = (div_eps0 / eps2[1:-1, 1:-1]).reshape(equa_num)

    poison_tissue_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1, (row, col)),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1 + dataM1), (row, col + 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1 - dataM1), (row, col - 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col + M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col - M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))
    )

    # data1 = np.array(
    #     [diff2[1:-1, 1:-1].reshape(equa_num),
    #      0.5 * (2 * diff2[1:-1, 1:-1] - diff2[2:, 1:-1] + diff2[:-2, 1:-1]).reshape(equa_num),
    #      0.5 * (2 * diff2[1:-1, 1:-1] + diff2[2:, 1:-1] - diff2[:-2, 1:-1]).reshape(equa_num),
    #      0.5 * (2 * diff2[1:-1, 1:-1] - diff2[1:-1, 2:] + diff2[1:-1, :-2]).reshape(equa_num),
    #      0.5 * (2 * diff2[1:-1, 1:-1] + diff2[1:-1, 2:] - diff2[1:-1, :-2]).reshape(equa_num)]
    # )
    data1 = np.array(
        [diff2[1:-1, 1:-1].reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] + diff2[2:, 1:-1] - diff2[:-2, 1:-1]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] - diff2[2:, 1:-1] + diff2[:-2, 1:-1]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] + diff2[1:-1, 2:] - diff2[1:-1, :-2]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] - diff2[1:-1, 2:] + diff2[1:-1, :-2]).reshape(equa_num)]
    )

    # data1 = np.array(
    #     [diff2[1:-1, 1:-1].reshape(equa_num),
    #      diff2[1:-1, 1:-1].reshape(equa_num),
    #      diff2[1:-1, 1:-1].reshape(equa_num),
    #      diff2[1:-1, 1:-1].reshape(equa_num),
    #      diff2[1:-1, 1:-1].reshape(equa_num)]
    # )
    # data1 = np.array(
    #     [diff2[1:-1, 1:-1].reshape(equa_num),
    #      diff2[2:, 1:-1].reshape(equa_num),
    #      diff2[:-2, 1:-1].reshape(equa_num),
    #      diff2[1:-1, 2:].reshape(equa_num),
    #      diff2[1:-1, :-2].reshape(equa_num)]
    # )

    dataM1 = diff2[1:-1, 1:-1].reshape(equa_num) * np.array([1. / (m + 1) for m in range(equa_num)])
    koef = 1
    sigma2mid = (koef * sigma2[1:-1, 1:-1] + (1 - koef) * (
                sigma2[2:, 1:-1] + sigma2[:-2, 1:-1] + sigma2[1:-1, 2:] + sigma2[1:-1, :-2]) / 4)
    data2 = div_eps0 * (sigma2mid / eps2[1:-1, 1:-1]).reshape(equa_num)

    germgolc_tissue_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1[0] - data2, (row, col + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1[4] + dataM1), (row, col + 1 + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * (data1[3] - dataM1), (row, col - 1 + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1[2], (row, col + M + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1[1], (row, col - M + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M))
    )

    return sp.vstack([poison_air_matrix, germgolc_air_matrix, poison_tissue_matrix, germgolc_tissue_matrix])


def get_system_matrix_internal_decart(cnf, properties):
    # creating poison_air_matrix
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    dn = cnf['dn']
    dm = cnf['dm']

    div_dn = 1 / dn
    div_dm = 1 / dm

    eps0 = properties['eps0']
    div_eps0 = 1 / eps0

    eps1 = 1 * np.ones((NAir, M))
    # sigma1 = 0 * np.ones((NAir, M))
    # diff1 = 0 * np.ones((NAir, M))

    eps2 = properties['eps']
    sigma2 = properties['sigma']
    diff2 = properties['diffusion']

    equa_num = (NAir - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(1, NAir - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    data2 = (div_eps0 / eps1[1:-1, 1: - 1]).reshape(equa_num)

    poison_air_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1, (row, col)),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1, (row, col + 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1, (row, col - 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col + M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col - M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))
    )

    # creating germgolc_air_matrix (v = 0)
    germgolc_air_matrix = sp.coo_matrix((data1, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))

    # creating poison_tissue_matrix
    equa_num = (N - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(NAir + 1, (N + NAir) - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    data2 = (div_eps0 / eps2[1:-1, 1:-1]).reshape(equa_num)

    poison_tissue_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1, (row, col)),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1, (row, col + 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1, (row, col - 1)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col + M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1, (row, col - M)), shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N + NAir))), shape=(equa_num, 2 * (N + NAir) * M))
    )

    data1 = np.array(
        [diff2[1:-1, 1:-1].reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] - diff2[2:, 1:-1] + diff2[:-2, 1:-1]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] + diff2[2:, 1:-1] - diff2[:-2, 1:-1]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] - diff2[1:-1, 2:] + diff2[1:-1, :-2]).reshape(equa_num),
         0.5 * (2 * diff2[1:-1, 1:-1] + diff2[1:-1, 2:] - diff2[1:-1, :-2]).reshape(equa_num)]
    )

    data2 = div_eps0 * (sigma2[1:-1, 1:-1] / eps2[1:-1, 1:-1]).reshape(equa_num)

    germgolc_tissue_matrix = (
            sp.coo_matrix((-2 * (div_dn ** 2 + div_dm ** 2) * data1[0] - data2, (row, col + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1[4], (row, col + 1 + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dm ** 2 * data1[3], (row, col - 1 + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1[2], (row, col + M + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M)) +
            sp.coo_matrix((div_dn ** 2 * data1[1], (row, col - M + M * (N + NAir))),
                          shape=(equa_num, 2 * (N + NAir) * M))
    )

    return sp.vstack([poison_air_matrix, germgolc_air_matrix, poison_tissue_matrix, germgolc_tissue_matrix])


def get_bound_matrix(cnf, properties, bound_electric):
    """
    Ищем матрицу и свободный вектор для уравнения заданного гран условием
    :return: Матрица, свободный вектор
    """
    condition_type = bound_electric.condition_type
    variable_name = bound_electric.variable_name
    value = bound_electric.coefficients
    coordinates = bound_electric.coordinates
    external_normal = bound_electric.external_normal

    "В зависимости от типа гран условия перенаправляем конкретной функции"
    if condition_type == 'Dirichlet':
        return Dirichlet(cnf, properties, variable_name, coordinates, value)
    elif condition_type == 'Newmann':
        return Newmann(cnf, properties, variable_name, coordinates, external_normal, value)
    elif condition_type == 'Continiosly':
        return Continiosly(cnf, properties, variable_name, coordinates[0], coordinates[1], value[0], value[1])
    else:
        raise ValueError('unknown conditions')


def Dirichlet(cnf, properties, variable_name, coordinates, value):
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']

    equa_num = len(coordinates)
    row = np.arange(equa_num)
    col = np.array([coo[0] * M + coo[1] for coo in coordinates])
    data = np.ones(equa_num)

    vector = value
    if variable_name == 'u':
        matrix = sp.coo_matrix((data, (row, col)), shape=(equa_num, 2 * (N + NAir) * M))
    elif variable_name == 'q':
        matrix = sp.coo_matrix((data, (row, col + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M))
    else:
        raise ValueError('unknown variable_name')
    return matrix, vector


def Newmann(cnf, properties, variable_name, coordinates, external_normal, value):
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    dn = cnf['dn']
    dm = cnf['dm']

    eps1 = 1 * np.ones((NAir, M))
    sigma1 = 0 * np.ones((NAir, M))
    diff1 = 0 * np.ones((NAir, M))

    eps2 = properties['eps']
    sigma2 = properties['sigma']
    diff2 = properties['diffusion']

    eps_all = np.concatenate([eps1, eps2])
    sigma_all = np.concatenate([sigma1, sigma2])
    diff_all = np.concatenate([diff1, diff2])

    equa_num = len(coordinates)
    row = np.arange(equa_num)
    col = np.array([coo[0] * M + coo[1] for coo in coordinates])
    data = np.ones(equa_num)

    normal = external_normal
    vector = value * (normal[0] * dn + normal[1] * dm)
    if variable_name == 'u':
        matrix = (sp.coo_matrix((data * (normal[0] + normal[1]), (row, col)),
                                shape=(equa_num, 2 * (N + NAir) * M)) +
                  sp.coo_matrix((-data * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1])),
                                shape=(equa_num, 2 * (N + NAir) * M)))
    elif variable_name == 'q':
        matrix = (sp.coo_matrix((data * (normal[0] + normal[1]), (row, col + (N + NAir) * M)),
                                shape=(equa_num, 2 * (N + NAir) * M)) +
                  sp.coo_matrix(
                      (-data * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1] + (N + NAir) * M)),
                      shape=(equa_num, 2 * (N + NAir) * M)))
    elif variable_name == 'j':
        data_sigma = sigma_all[coordinates[:, 0], coordinates[:, 1]]
        data_diff = diff_all[coordinates[:, 0], coordinates[:, 1]]
        matrix = (sp.coo_matrix((data_sigma * (normal[0] + normal[1]), (row, col)),
                                shape=(equa_num, 2 * (N + NAir) * M)) +
                  sp.coo_matrix((-data_sigma * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1])),
                                shape=(equa_num, 2 * (N + NAir) * M)) +
                  sp.coo_matrix((data_diff * (normal[0] + normal[1]), (row, col + (N + NAir) * M)),
                                shape=(equa_num, 2 * (N + NAir) * M)) +
                  sp.coo_matrix((-data_diff * (normal[0] + normal[1]),
                                 (row, col - normal[0] * M - normal[1] + (N + NAir) * M)),
                                shape=(equa_num, 2 * (N + NAir) * M)))
    else:
        raise ValueError('unknown variable_name')

    return matrix, vector


def Continiosly(cnf, properties, variable_name, coordinates_in, coordinates_out, ratio, ratio_dif):
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']

    equa_num = len(coordinates_in)
    row = np.arange(equa_num)
    col1 = np.array([coo[0] * M + coo[1] for coo in coordinates_in])
    col2 = np.array([coo[0] * M + coo[1] for coo in coordinates_out])

    data = np.ones(equa_num)
    vector = np.zeros(2 * equa_num)

    if variable_name == 'u':
        matrix = sp.vstack([
            (sp.coo_matrix((data, (row, col1)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-ratio, (row, col2)), shape=(equa_num, 2 * (N + NAir) * M))),
            (sp.coo_matrix((data, (row, col1)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-data, (row, col1 - M)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((ratio_dif, (row, col2)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-ratio_dif, (row, col2 + M)), shape=(equa_num, 2 * (N + NAir) * M)))
        ])

    elif variable_name == 'q':
        matrix = sp.vstack([
            (sp.coo_matrix((data, (row, col1 + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-ratio, (row, col2 + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M))),
            (sp.coo_matrix((data, (row, col1 + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-data, (row, col1 - M + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((ratio_dif, (row, col2 + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M)) +
             sp.coo_matrix((-ratio_dif, (row, col2 + M + (N + NAir) * M)), shape=(equa_num, 2 * (N + NAir) * M)))
        ])
    else:
        raise ValueError('unknown variable_name')

    return matrix, vector
