import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
import scenario_reduction as scen_red
from scipy.stats import beta


def windPow(ws):
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    if ws < vin or ws > vout:
        return 0
    elif vin <= ws <= vrated:
        return (prated * (ws - vin)) / (vrated - vin)
    else:
        return prated


def createScenarios(timeInd: int):
    from scenarioGeneration3 import windPow
    loadMean = 0.015
    loadSD = loadMean * 0.1
    reference_density = norm.pdf(loadMean, loadMean, loadSD)
    n = 100
    # timeInd = 8

    bSet = set(range(0, 30))
    drSet = {3, 5, 9, 11, 27, 29}
    rSet = {4, 10, 28}
    demandSet = bSet - (drSet | rSet)
    dn = len(demandSet)

    darr = np.arange(0, n)
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    shape, scale = 7.5, 3.5

    l, p = dict(), dict()

    for j in range(0, (dn + len(rSet)) * timeInd):
        if j % (dn + len(rSet)) < dn:
            a = norm(loadMean, loadSD).rvs(n)
            darr = np.column_stack((darr, a))
        elif j % (dn + len(rSet)) >= dn:
            p = np.array(weibull_min(shape).rvs(n) * scale)
            darr = np.column_stack((darr, p))

    darr = np.delete(darr, 0, 1)
    probs = []
    for i in range(0, len(darr)):
        prob = 1
        for j in range(0, (dn + len(rSet)) * timeInd):
            if j % (dn + len(rSet)) < dn:
                if darr[i, j] > 0:
                    prob *= norm.pdf(darr[i, j], loadMean, loadSD) / reference_density
                elif darr[i, j] <= 0:
                    prob *= norm.cdf(0, loadMean, loadSD)
            elif j % (dn + len(rSet)) >= dn:
                prob *= weibull_min.pdf(darr[i, j] / scale, shape)
            else:
                print("ERROR")
        probs.append(prob)

    probabilities = probs / sum(probs)
    scenarios = darr.T
    S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm',
                                   scen0=np.zeros(len(scenarios)))
    # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
    S.fast_forward_sel(n_sc_red=6)

    # get reduced scenarios
    scenarios_reduced = S.scenarios_reduced

    # get reduced probabilities
    probabilities_reduced = S.probabilities_reduced
    probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)
    s_reduced = scenarios_reduced.T

    for i in range(0, len(s_reduced)):
        for j in range(0, len(s_reduced[0])):
            if j % (dn + len(rSet)) >= 21:
                s_reduced[i, j] = windPow(s_reduced[i, j])

    s_res = np.reshape(s_reduced, (timeInd, 6, 24))  ## Time - Scenario -  Bus

    return s_res, np.array(probabilities_reduced.tolist())


def createScenarios2(timeInd, pars_for_scenarios: dict):
    loadMean = 0.015
    loadSD = loadMean * 0.1
    reference_density = norm.pdf(loadMean, loadMean, loadSD)
    n = 1000
    bSet = set(range(0, 30))
    drSet = pars_for_scenarios['drSet']
    rSet = pars_for_scenarios['renSet']
    demandSet = bSet - (drSet | rSet)
    nB = len(bSet)

    darr = np.arange(0, n)
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    shape, scale = 7.5, 3.5

    l, p = dict(), dict()
    for j in range(0, 30 * timeInd):
        if j % nB in demandSet:
            a = norm(loadMean, loadSD).rvs(n)
            darr = np.column_stack((darr, a))
        elif j % nB in rSet:
            p = np.array(weibull_min(shape).rvs(n) * scale)
            darr = np.column_stack((darr, p))
        else:
            darr = np.column_stack((darr, np.zeros(n).reshape((-1, 1))))

    darr = np.delete(darr, 0, 1)
    probs = []
    for i in range(0, len(darr)):
        prob = 1
        for j in range(0, 30 * timeInd):
            if j % 30 in demandSet:
                if darr[i, j] > 0:
                    prob *= norm.pdf(darr[i, j], loadMean, loadSD) / reference_density
                elif darr[i, j] <= 0:
                    prob *= norm.cdf(0, loadMean, loadSD)
            elif j % 30 in rSet:
                prob *= weibull_min.pdf(darr[i, j] / scale, shape)
            else:
                prob *= 1
        probs.append(prob)

    probabilities = probs / sum(probs)
    scenarios = darr.T
    S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm',
                                   scen0=np.zeros(len(scenarios)))
    # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
    S.fast_forward_sel(n_sc_red=6)

    # get reduced scenarios
    scenarios_reduced = S.scenarios_reduced

    # get reduced probabilities
    probabilities_reduced = S.probabilities_reduced

    # probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

    s_reduced = scenarios_reduced.T

    for i in range(0, len(s_reduced)):
        for j in range(0, nB * timeInd):
            if j % 30 in rSet:
                s_reduced[i, j] = windPow(s_reduced[i, j])

    s_res = np.reshape(s_reduced, (timeInd, 6, nB))  # Time - Scenario -  Bus

    return s_res, np.array(probabilities_reduced.tolist())


def initialize33():
    pars33 = dict()
    pars33['drSet'] = [12, 14, 24, 32]
    pars33['eSet'] = {}
    pars33['renSet'] = [5, 9, 28]
    pars33['ren2Set'] = [19]
    pars33['caps'] = [3.0, 2.0, 2.0, 3.0]
    pars33['diSet'] = {}
    pars33['NS'] = 1
    pars33['diSet'] = {}
    pars33['N'] = 33
    pars33['lMean'] = 0.0015
    pars33['SGmax'] = dict(zip(pars33['drSet'], [pars33['caps'][i] for i in range(len(pars33['drSet']))]))
    pars33['lSdf'] = 0.1

    return pars33


pars_for_scenarios = initialize33()
timeInd = 1


def createScenarios2(timeInd, pars_for_scenarios: dict):
    loadMean = 0.0015
    loadSD = loadMean * 0.1
    reference_density = norm.pdf(loadMean, loadMean, loadSD)

    n = 1000
    bSet = set(range(0, 33))
    drSet = pars_for_scenarios['drSet']
    rSet = pars_for_scenarios['renSet']
    r2Set = pars_for_scenarios['ren2Set']
    demandSet = bSet - (drSet | rSet | r2Set)
    nB = len(bSet)

    darr = np.arange(0, n)
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    shape, scale = 7.5, 3.5
    A, B = 0.40, 8.56
    reference_density2 = beta.pdf(A / (A + B), A, B)
    l, p = dict(), dict()
    for j in range(0, 33 * timeInd):
        if j % nB in demandSet:
            a = norm(loadMean, loadSD).rvs(n)
            darr = np.column_stack((darr, a))
        elif j % nB in rSet:
            p = np.array(weibull_min(shape).rvs(n) * scale)
            darr = np.column_stack((darr, p))
        elif j % nB in r2Set:
            p = np.array(beta(A, B).rvs(n) * 66 * 0.0014)
            darr = np.column_stack((darr, p))
        else:
            darr = np.column_stack((darr, np.zeros(n).reshape((-1, 1))))

    darr = np.delete(darr, 0, 1)
    probs = []
    for i in range(0, len(darr)):
        prob = 1
        for j in range(0, 33 * timeInd):
            if j % 33 in demandSet:
                if darr[i, j] > 0:
                    prob *= norm.pdf(darr[i, j], loadMean, loadSD) / reference_density
                elif darr[i, j] <= 0:
                    prob *= norm.cdf(0, loadMean, loadSD)
            elif j % 33 in rSet:
                prob *= weibull_min.pdf(darr[i, j] / scale, shape)
            elif j % 33 in r2Set:
                prob *= beta.pdf(darr[i, j], A, B) / reference_density2
            else:
                prob *= 1
        probs.append(prob)

    probabilities = probs / sum(probs)
    scenarios = darr.T
    S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm',
                                   scen0=np.zeros(len(scenarios)))

    # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
    S.fast_forward_sel(n_sc_red=6)

    # get reduced scenarios
    scenarios_reduced = S.scenarios_reduced

    # get reduced probabilities
    probabilities_reduced = S.probabilities_reduced

    # probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

    s_reduced = scenarios_reduced.T

    for i in range(0, len(s_reduced)):
        for j in range(0, nB * timeInd):
            if j % nB in rSet:
                s_reduced[i, j] = windPow(s_reduced[i, j])

    s_res = np.reshape(s_reduced, (6, nB))  # Time - Scenario -  Bus

    return s_res, np.array(probabilities_reduced.tolist())


def initialize12():
    pars33 = dict()
    pars33['drSet'] = {3, 5, 9, 11}
    pars33['eSet'] = {}
    pars33['renSet'] = {4, 10}
    pars33['ren2Set'] = set({})
    pars33['sSet'] = set(range(0, 6))
    pars33['diSet'] = {}
    pars33['N'] = 12
    pars33['NS'] = 1
    pars33['lMean'] = 0.0015
    pars33['SGmax'] = dict(zip(pars33['drSet'], [1, 1, 1, 1]))
    # PRID = np.random.poisson(2, 30).copy()
    # PRID[np.argwhere(PRID == 0)] = 1
    # PRIS = np.random.poisson(4, 3).copy()
    # PRIS[np.argwhere(PRIS == 0)] = 1
    return pars33


pars_for_scenarios = initialize12()
timeInd = 1


def createScenariosN(timeInd, pars_for_scenarios: dict):
    loadMean = pars_for_scenarios['lMean']
    loadSD = loadMean * pars_for_scenarios['lSdf']
    reference_density = norm.pdf(loadMean, loadMean, loadSD)

    n = 1000
    bSet = set(range(0, pars_for_scenarios['N']))
    drSet = pars_for_scenarios['drSet']
    rSet = pars_for_scenarios['renSet']
    r2Set = pars_for_scenarios['ren2Set']
    demandSet = bSet - (set(drSet) | set(rSet) | set(r2Set))
    nB = len(bSet)

    darr = np.arange(0, n)
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    shape, scale = 7.5, 3.5
    A, B = 0.40, 8.56
    reference_density2 = beta.pdf(A / (A + B), A, B)
    l, p = dict(), dict()
    for j in range(0, nB * timeInd):
        if j % nB in demandSet:
            a = norm(loadMean, loadSD).rvs(n)
            darr = np.column_stack((darr, a))
        elif j % nB in rSet:
            p = np.array(weibull_min(shape).rvs(n) * scale)
            darr = np.column_stack((darr, p))
        elif j % nB in r2Set:
            p = np.minimum(np.array(beta(A, B).rvs(n) * 6600 * 0.0014), 0.0015)
            darr = np.column_stack((darr, p))
        else:
            darr = np.column_stack((darr, np.zeros(n).reshape((-1, 1))))

    darr = np.delete(darr, 0, 1)
    probs = []
    for i in range(0, len(darr)):
        prob = 1
        for j in range(0, nB * timeInd):
            if j % nB in demandSet:
                if darr[i, j] > 0:
                    prob *= norm.pdf(darr[i, j], loadMean, loadSD) / reference_density
                elif darr[i, j] <= 0:
                    prob *= norm.cdf(0, loadMean, loadSD)
            elif j % nB in rSet:
                prob *= weibull_min.pdf(darr[i, j] / scale, shape)
            elif j % nB in r2Set:
                prob *= beta.pdf(darr[i, j], A, B) / reference_density2
            else:
                prob *= 1
        probs.append(prob)

    probabilities = probs / sum(probs)
    scenarios = darr.T
    S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm',
                                   scen0=np.zeros(len(scenarios)))

    # use fast forward selection algrithm to reduce to 5 scenarios with 4 threads
    S.fast_forward_sel(n_sc_red=pars_for_scenarios['NS'])

    # get reduced scenarios
    scenarios_reduced = S.scenarios_reduced

    # get reduced probabilities
    probabilities_reduced = S.probabilities_reduced

    # probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

    s_reduced = scenarios_reduced.T

    for i in range(0, len(s_reduced)):
        for j in range(0, nB * timeInd):
            if j % nB in rSet:
                s_reduced[i, j] = windPow(s_reduced[i, j])

    s_res = np.reshape(s_reduced, (pars_for_scenarios['NS'], nB))  # Time - Scenario -  Bus

    return s_res, np.array(probabilities_reduced.tolist())


def initialize3():
    pars33 = dict()
    pars33['drSet'] = {2}
    pars33['eSet'] = {}
    pars33['renSet'] = {0}
    pars33['ren2Set'] = set({})
    pars33['NS'] = 2
    pars33['sSet'] = set(range(0, pars33['NS']))
    pars33['diSet'] = {}
    pars33['N'] = 3
    pars33['lMean'] = 0.1
    pars33['SGmax'] = dict(zip(pars33['drSet'], [i for i in range(len(pars33['drSet']))]))

    return pars33


def initialize6():
    pars33 = dict()
    pars33['drSet'] = [3, 5]
    pars33['eSet'] = {}
    pars33['renSet'] = [4]
    pars33['ren2Set'] = set({})
    pars33['NS'] = 200
    # pars33['sSet'] = set(range(0, pars33['NS']))
    pars33['diSet'] = []
    pars33['N'] = 6
    pars33['lMean'] = 0.1
    pars33['lSdf'] = 0.4
    pars33['SGmax'] = dict(zip(pars33['drSet'], [1.5 for i in range(len(pars33['drSet']))]))

    return pars33


def initialize12():
    pars33 = dict()
    pars33['drSet'] = [3, 5, 8, 11]
    pars33['caps'] = [1, 1.5, 1, 1.5]
    pars33['eSet'] = {}
    pars33['renSet'] = [4, 10]
    pars33['ren2Set'] = set({})
    pars33['NS'] = 50
    pars33['diSet'] = {}
    pars33['N'] = 12
    pars33['lMean'] = 0.015
    pars33['lSdf'] = 0.5
    pars33['SGmax'] = dict(zip(pars33['drSet'], [pars33['caps'][i] for i in range(len(pars33['drSet']))]))

    return pars33
