import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
import scenario_reduction as scen_red


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

def createScenarios2(timeInd, pars_for_scenarios:dict):
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
