import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import matplotlib.pyplot as plt

"""
Решатель мой:
"""

def solve(cnf, properties, T, g, bound_electric):
# Создаем матрицу для внутренних точек и вектор свободных коэффициентов заодно
    system_matrix = get_system_matrix_internal(cnf, properties, T, g)
    free_vector = np.zeros(system_matrix.shape[0])
#     Поочередно добавляем в матрицу условия на границе, такой подход позволяет не париться на каких там местах стоят эти уравнения, можно хоть в каждой точке свое условие задать
    for bound_electric_i in bound_electric(T, g):
        matrix_i, vector_i = get_bound_matrix(cnf, bound_electric_i)
        system_matrix = sp.vstack([system_matrix,matrix_i])
        free_vector = np.concatenate([free_vector,vector_i])
#         Решаем все уравнения вместе
    uv = spsolve(system_matrix.tocsr(), free_vector).reshape(2, cnf['N'] + cnf['NAir'], cnf['M'])
    return uv[0], uv[1]

"""
Тут получение внутренней матрицы. Там их 4, для воздуха, для ткани и в каждом случае на потенциал и заряды
"""
def get_system_matrix_internal(cnf, properties, T, g):
#     Тут как раз на счет локальности N и M, до этого они все валялись в cnf теперь достаю, чтобы каждый раз не прописывать обращение к словарю
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    dn = cnf['dn']
    dm = cnf['dm']
    
    div_dn = 1/dn
    div_dm = 1/dm
#     Деление долгая оперция, делю один раз тут, а не каждый раз
    eps0 = properties['eps0']
    div_eps0 = 1 / eps0
    
    eps1 = 1 * np.ones((NAir, M))
    sigma1 = 0 * np.ones((NAir, M))
    diff1 = 0 * np.ones((NAir, M))
    
    eps2 = properties['eps'](T,g)
    sigma2 = properties['sigma'](T,g)
    diff2 = properties['diffusion'](T,g)
    
    #creating poison_air_matrix
    equa_num = (NAir - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(1, NAir - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    data2 = (div_eps0 / eps1[1:-1, 1: - 1]).reshape(equa_num)

    poison_air_matrix = (
            sp.coo_matrix((-2 * (div_dn**2 + div_dm**2) * data1, (row, col)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1, (row, col + 1)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1, (row, col - 1)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1, (row, col + M)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1, (row, col - M)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M))
    )

    #creating germgolc_air_matrix (v = 0)
    germgolc_air_matrix = sp.coo_matrix((data1, (row, col + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M))

    #creating poison_tissue_matrix
    equa_num = (N - 2) * (M - 2)
    row = np.arange(equa_num)
    col = np.array([[n * M + m for m in range(1, M - 1)] for n in range(NAir+1, (N+NAir) - 1)]).reshape(equa_num)
    data1 = np.ones(equa_num)
    data2 = (div_eps0 / eps2[1:-1, 1:-1]).reshape(equa_num)

    poison_tissue_matrix = (
            sp.coo_matrix((-2 * (div_dn**2 + div_dm**2) * data1, (row, col)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1, (row, col + 1)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1, (row, col - 1)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1, (row, col + M)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1, (row, col - M)), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((data2, (row, col + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M))
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
            sp.coo_matrix((-2 * (div_dn**2 + div_dm**2) * data1[0] - data2, (row, col + M * (N+NAir))), 
                          shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1[4], (row, col + 1 + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dm**2 * data1[3], (row, col - 1 + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1[2], (row, col + M + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M)) +
            sp.coo_matrix((div_dn**2 * data1[1], (row, col - M + M * (N+NAir))), shape=(equa_num, 2 * (N+NAir) * M))
    )
    
    return sp.vstack([poison_air_matrix, germgolc_air_matrix, poison_tissue_matrix, germgolc_tissue_matrix])

# Доставалка граничных условий, преобрзует тот формат, который в main в тот который я писал давно уже (хотя они похожи)
def get_bound_matrix(cnf, bound_electric_i):
    
    condition_type = bound_electric_i.condition_type
    variable_name = bound_electric_i.variable_name
    value = bound_electric_i.coefficients
    coordinates = bound_electric_i.coordinates
    external_normal = bound_electric_i.external_normal 
       
    if condition_type == 'Dirichlet':
        matrix, vector = Dirichlet(cnf, variable_name, coordinates, value)
    elif condition_type == 'Newmann':
        matrix, vector = Newmann(cnf, variable_name, coordinates, external_normal, value)
    elif condition_type == 'Continiosly':
        matrix, vector = Continiosly(cnf, variable_name, coordinates[0], coordinates[1], value[0], value[1])
    else:
        raise ValueError('unknown conditions')
        
    return matrix, vector


def Dirichlet(cnf, variable_name, coordinates, value):
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    
    equa_num = len(coordinates)
    row = np.arange(equa_num)
    col = np.array([coo[0] * M + coo[1] for coo in coordinates])
    data = np.ones(equa_num)

    vector = value
    if variable_name == 'u':
#         matrix = sp.coo_matrix((data, (row, col)), shape=(equa_num, 2 * (N+NAir) * M))
        return sp.coo_matrix((data, (row, col)), shape=(equa_num, 2 * (N+NAir) * M)), vector
    elif variable_name == 'v':
#         matrix = sp.coo_matrix((data, (row, col + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M))
        return sp.coo_matrix((data, (row, col + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)), vector

#     return matrix, vector

"""
Тут есть приколы через нормаль, хотя тут я уже сам ничего не помню, но можно вывести и самому, нормаль определяет 0 и 1 элементы или N-1 N элементы тебе нужно взять в производной.
"""
def Newmann(cnf, variable_name, coordinates, external_normal, value):
    N = cnf['N']
    NAir = cnf['NAir']
    M = cnf['M']
    
    equa_num = len(coordinates)
    row = np.arange(equa_num)
    col = np.array([coo[0] * M + coo[1] for coo in coordinates])
    data = np.ones(equa_num)

    normal = external_normal
    vector = value * (normal[0] * dn + normal[1] * dm)
    if variable_name == 'u':
        matrix = (sp.coo_matrix((data * (normal[0] + normal[1]), (row, col)),
                                shape=(equa_num, 2 * (N+NAir) * M)) +
                  sp.coo_matrix((-data * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1])),
                                shape=(equa_num, 2 * (N+NAir) * M)))
    elif variable_name == 'v':
        matrix = (sp.coo_matrix((data * (normal[0] + normal[1]), (row, col + (N+NAir) * M)),
                                shape=(equa_num, 2 * (N+NAir) * M)) +
                  sp.coo_matrix((-data * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1] + (N+NAir) * M)),
                                shape=(equa_num, 2 * (N+NAir) * M)))
    elif variable_name == 'j':
        data_sigma = sigma_all[coordinates[:, 0], coordinates[:, 1]]
        data_diff = diff_all[coordinates[:, 0], coordinates[:, 1]]
        matrix = (sp.coo_matrix((data_sigma * (normal[0] + normal[1]), (row, col)),
                                shape=(equa_num, 2 * (N+NAir) * M)) +
                  sp.coo_matrix((-data_sigma * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1])),
                                shape=(equa_num, 2 * (N+NAir) * M)) +
                  sp.coo_matrix((data_diff * (normal[0] + normal[1]), (row, col + (N+NAir) * M)),
                                shape=(equa_num, 2 * (N+NAir) * M)) +
                  sp.coo_matrix((-data_diff * (normal[0] + normal[1]), (row, col - normal[0] * M - normal[1] + (N+NAir) * M)),
                                shape=(equa_num, 2 * (N+NAir) * M)))

    return matrix, vector

def Continiosly(cnf, variable_name, coordinates_in, coordinates_out, ratio, ratio_dif):
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
            (sp.coo_matrix((data, (row, col1)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-ratio, (row, col2)), shape=(equa_num, 2 * (N+NAir) * M))),
            (sp.coo_matrix((data, (row, col1)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-data, (row, col1 - M)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((ratio_dif, (row, col2)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-ratio_dif, (row, col2 + M)), shape=(equa_num, 2 * (N+NAir) * M)))
        ])

    elif variable_name == 'v':
        matrix = sp.vstack([
            (sp.coo_matrix((data, (row, col1 + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-ratio, (row, col2 + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M))),
            (sp.coo_matrix((data, (row, col1 + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-data, (row, col1 - M + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((ratio_dif, (row, col2 + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)) +
             sp.coo_matrix((-ratio_dif, (row, col2 + M + (N+NAir) * M)), shape=(equa_num, 2 * (N+NAir) * M)))
        ])

    return matrix, vector


