import optic
import thermal
import degrading
import electric

import numpy as np


def step_solver(cnf, properties, old_T, old_g, bound_optic, bound_electric, bound_thermal):
    """
    Решаем совместо все задачи на одном временном шаге
    :param cnf: Настройки сетки
    :param properties: Словарь со свойствами тканей, как функции температуры и степени деградации
    :param old_T: Температура на предыдущем шаге
    :param old_g: Степень деградации на предыдущем шаге
    :param bound_optic: Гран условия оптики, как список элементов класс Boundary
    :param bound_electric: Гран условия электричества, как список элементов класс Boundary
    :param bound_thermal: Гран условия теплопроводности, как список элементов класс Boundary
    :return: {I,Q,u,q,T,g} - интенсивности излучения, тепловые источники при разогреве излучением,
    потенциалы, плотности зарядов, температуры, степени повреждения соотвтственно на новом шаге
    """
    new_I, new_Q = optic.solve(cnf, properties, old_T, old_g, bound_optic)
    new_u, new_q = electric.solve(cnf, properties, old_T, old_g, bound_electric)

    new_T = thermal.solve(cnf, properties, old_T, new_Q, bound_thermal)
    new_g = degrading.solve(cnf, properties, old_g, new_T)

    return {"I": new_I, "Q": new_Q, "u": new_u, "q": new_q, "T": new_T, "g": new_g}


def solver(cnf, properties, condition_start, bound_optic, bound_electric, bound_thermal):
    """
    Решаем совместо все задачи на всем временном промежутке
    :param cnf: Настройки сетки
    :param properties: Словарь со свойствами тканей, как функции температуры и степени деградации
    :param condition_start: Начальные условия, как словарь таблиц с начальными температурами и степенями деградации
    :param bound_optic: Гран условия оптики, как список элементов класс Boundary
    :param bound_electric: Гран условия электричества, как список элементов класс Boundary
    :param bound_thermal: Гран условия теплопроводности, как список элементов класс Boundary
    :return: [{I,Q,u,q,T,g},...] - интенсивности излучения, тепловые источники при разогреве излучением,
    потенциалы, плотности зарядов, температуры, степени повреждения соотвтственно на каждом временном шаге
    Каждая "переменная" в виде массива [t,z,r]
    """
    step_solution = step_solver(cnf, properties,
                                condition_start['T'], condition_start['g'],
                                bound_optic, bound_electric, bound_thermal)
    solution = {
        "T": np.array([condition_start['T']]),
        "g": np.array([condition_start['g']]),
        'I': np.array([step_solution['I']]),
        'Q': np.array([step_solution['Q']]),
        'u': np.array([step_solution['u']]),
        'q': np.array([step_solution['q']])
    }
    for k in range(cnf['K']):
        step_solution = step_solver(cnf, properties, solution['T'][-1], solution['g'][-1],
                                    bound_optic, bound_electric, bound_thermal)

        solution['I'] = np.concatenate((solution['I'], [step_solution['I']]))
        solution['Q'] = np.concatenate((solution['Q'], [step_solution['Q']]))
        solution['u'] = np.concatenate((solution['u'], [step_solution['u']]))
        solution['q'] = np.concatenate((solution['q'], [step_solution['q']]))
        solution['T'] = np.concatenate((solution['T'], [step_solution['T']]))
        solution['g'] = np.concatenate((solution['g'], [step_solution['g']]))

    return solution


def calc_impedance(cnf, solution, f, coo_1, coo_2):
    u = solution['u']
    q = solution['q']
    dk = cnf['dk']
    dn = cnf['dn']
    dm = cnf['dm']

    u1 = u[:, coo_1[0], coo_1[1]]
    u2 = u[:, coo_2[0], coo_2[1]]
    q1 = q[:, coo_1[0], coo_1[1]] * 2 * 3.1416 * coo_1[1] * dn *dm
    q2 = q[:, coo_2[0], coo_2[1]] * 2 * 3.1416 * coo_2[1] * dn *dm

    impedance1 = (u2 - u1) / (q1 * f)
    impedance2 = - (u2 - u1) / (q2 * f)
    return impedance1, impedance2
