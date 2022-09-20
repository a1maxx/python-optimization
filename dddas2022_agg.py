
import cmath
import pyomo.environ as pyo
import math
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.environ import value, NonNegativeReals
import numpy as np
from pyomo.environ import ConcreteModel, minimize, SolverFactory, Set, Param, Var, Constraint, Objective, \
    TransformationFactory
from pyomo.gdp import Disjunction
import pickle
import pandas as pd





def arrtoDict(arr):
    d = dict()
    for k, v in enumerate(arr):
        for j, l in enumerate(v):
            d[k, j] = l
    return d


# %%
def microgrid_model(dfR, dfX, red_scenes, red_probs, pars: dict):
    withCon = pars['withCon']
    PRID = pars['PRID']
    PRIS = pars['PRIS']
    drSet = pars['drSet']

    # diSet = pars['diSet']
    # sSet = pars['sSet']
    # pars['renSet']

    dict_complex = dict()
    dict_mag = dict()
    dict_the = dict()
    edges = []

    for i in range(0, dfR.shape[0]):
        for j in range(0, dfR.shape[1]):
            if i != j and dfR.loc[i, j] != 0:
                dict_complex[i, j] = - (1 / complex(dfR.loc[i, j], dfX.loc[i, j]))
                edges.append((i, j))
            else:
                dict_complex[i, j] = complex(0, 0)

    for i in range(0, dfR.shape[0]):
        total = complex(0, 0)
        for j in edges:
            if i == j[0]:
                total -= dict_complex[j]
        dict_complex[i, i] = total

    for i in range(0, dfR.shape[0]):
        for j in range(0, dfR.shape[1]):
            dict_mag[i, j], dict_the[i, j] = cmath.polar(dict_complex[i, j])

    del i, j

    model = ConcreteModel()
    model.N = pyo.Param(default=int(math.sqrt(len(dict_mag))))
    model.Z = Param(initialize=red_scenes.shape[0])  ## after changing the preprocessing of red_scenes this is the
    ## correct one
    # model.Z = Param(initialize=1)  ## For test purposes

    model.KPF = pyo.Param(initialize=1)
    model.KQF = pyo.Param(initialize=-1)
    model.alpha = pyo.Param(initialize=1)
    model.beta = pyo.Param(initialize=1)

    model.J = pyo.RangeSet(0, int(math.sqrt(len(dict_mag))) - 1)
    model.T = pyo.RangeSet(0, len(red_scenes) - 1)
    # model.drgenSet = pyo.Set(initialize={0, 1, 12, 21, 22, 26})
    model.drgenSet = pyo.Set(initialize=pars['drSet'])
    model.digenSet = pyo.Set(initialize=pars['diSet'])
    # model.S = pyo.Set(initialize={0, 1, 2, 3, 4, 5})
    model.S = pyo.Set(initialize=pars['sSet'])
    # model.renGen = pyo.Set(initialize={5, 9, 28})
    model.renGen = pyo.Set(initialize=pars['renSet'])
    # edge_servers = set(sorted([2, 10, 11]))
    edge_servers = pars['eSet']
    model.edSer = pyo.Set(initialize=edge_servers)

    # renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
    nScenarios = len(red_scenes[0])
    prenGen = dict()
    qrenGen = dict()

    ### Later reference purposes
    # a = np.arange(36).reshape(2, 3, 6)
    # np.append(a, np.zeros(6).reshape(2, 3, 1), axis=2)
    # ## or
    # np.insert(a, 6, 0, axis=2)

    ### When red_scenes is changed to be 3-dimensional
    for i in model.J:
        if i in model.drgenSet:
            if i < len(red_scenes[0, 0]):
                red_scenes = np.insert(red_scenes, i, np.zeros(nScenarios), axis=2)
            else:
                red_scenes = np.append(red_scenes, np.zeros(shape=(nScenarios, 1)), axis=1)

    # for i in model.J:
    #     if i in model.drgenSet:
    #         if i < len(red_scenes[0]):
    #             red_scenes = np.insert(red_scenes, i, a, axis=1)
    #         else:
    #             red_scenes = np.append(red_scenes, np.zeros(shape=(model.Z,nScenarios, 1)), axis=1)

    for t in model.T:
        for i in model.J:
            if i in model.renGen:
                for s in model.S:
                    prenGen[t, s, i] = red_scenes[t, s, i]
                    qrenGen[t, s, i] = red_scenes[t, s, i] * 0.6
                # red_scenes[:,:,i] = np.zeros(nScenarios) ### Could not understand what this is for now 9-14-22

    d1 = {}
    for t in model.T:
        for s in model.S:
            for i in model.J:
                d1[t, s, i] = red_scenes[t, s, i] + (i in model.edSer) * 0.01

    PF = 0.6
    d2 = {}
    for k, v in d1.items():
        d2[k] = v * PF

    d3 = {}
    j = 0
    ##   Below for loop needs to change
    for i in model.edSer:
        for t in model.T:
            d3[t, i] = withCon[j] * 1.5

        j += 1

    d4 = {}
    for k, v in d3.items():
        d4[k] = v * PF

    d5 = {}
    for k in zip(model.edSer, PRIS):
        d5[k[0]] = k[1]

    model.DPRI = pyo.Param(model.J, initialize=PRID)
    model.SPRI = pyo.Param(model.edSer, initialize=d5)
    model.EP = pyo.Param(model.T * model.edSer, initialize=d3)  ## There might be a need to add T
    model.EQ = pyo.Param(model.T, model.edSer, initialize=d4)  ## There might be a need to add T
    # model.SPROBS = pyo.Param(model.S, initialize=arrtoDict(red_probs))  ## There might be a need to add T
    model.SPROBS = pyo.Param(model.S, initialize=red_probs)  ## Either this or above need to think on it

    model.p0 = pyo.Param(model.T * model.S * model.J, initialize=d1)
    model.q0 = pyo.Param(model.T * model.S * model.J, initialize=d2)
    d3, d4 = dict(), dict()

    model.PR = pyo.Param(model.T * model.S * model.renGen, initialize=prenGen)
    model.QR = pyo.Param(model.T * model.S * model.renGen, initialize=qrenGen)
    model.w0 = pyo.Param(initialize=1.0)
    model.V0 = pyo.Param(initialize=1.01)
    model.SGmax = pyo.Param(model.drgenSet, initialize={0: 0.05, 1: 0.06, 12: 0.05, 21: 0.05, 22: 0.06, 26: 0.05})

    model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
    model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

    model.ql = pyo.Var(model.T * model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.pl = pyo.Var(model.T * model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.pg = pyo.Var(model.T * model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.qg = pyo.Var(model.T * model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

    model.r = pyo.Var(model.T * model.edSer, initialize=0, within=pyo.Binary)
    model.c = pyo.Var(model.T * model.J, initialize=1, within=pyo.Binary)

    model.v = pyo.Var(model.T * model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.95, 1.05))
    model.d = pyo.Var(model.T * model.S * model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))

    model.mp = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.03, bounds=(1e-10, 1))
    model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(1e-10, 1))

    model.w = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals, initialize=1, bounds=(0.995, 1.005))

    def obj_expression(m):
        return sum(model.SPROBS[s] * (sum(model.r[t, i] * model.SPRI[i] for t, i in model.T * m.edSer) + sum(
            model.c[t, i] * model.DPRI[i] for t, i in model.T * model.J)) for s in model.S)

    model.o = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    def ax_constraint_rule3(m, t, s, i):
        if i in m.renGen:
            return (m.pg[t, s, i] + m.PR[t, s, i] - m.pl[t, s, i]) - sum(
                m.v[t, s, i] * m.v[t, s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[t, s, j] - m.d[t, s, i]) for j
                in
                m.J) == 0
        else:
            return (m.pg[t, s, i] - m.pl[t, s, i]) - sum(
                m.v[t, s, i] * m.v[t, s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[t, s, j] - m.d[t, s, i]) for j
                in m.J) == 0

    def ax_constraint_rule4(m, t, s, i):
        if i in m.renGen:
            return (m.qg[t, s, i] + m.QR[t, s, i] - m.ql[t, s, i]) + sum(
                m.v[t, s, i] * m.v[t, s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[t, s, j] - m.d[t, s, i]) for j
                in m.J) == 0
        else:
            return (m.qg[t, s, i] - m.ql[t, s, i]) + sum(
                m.v[t, s, i] * m.v[t, s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[t, s, j] - m.d[t, s, i]) for j
                in m.J) == 0

    def ax_constraint_rule5(m, t, s, i):
        if i in m.drgenSet:
            # return m.pg[s, i] == (1 / m.mp[i]) * (m.w0 - m.w[s])
            return m.pg[t, s, i] == 0.5 * (
                    ((1 / model.mp[i]) * (model.w0 - model.w[t, s])) + (
                    (1 / model.nq[i]) * (model.V0 - model.v[t, s, i])))
        elif i in m.digenSet:
            return m.pg[t, s, i] <= 0.1
        else:
            return m.pg[t, s, i] == 0

    def ax_constraint_rule6(m, t, s, i):
        if i in m.drgenSet:
            # return m.qg[s, i] == (1 / m.nq[i]) * (m.V0 - m.v[s, i])
            return m.qg[t, s, i] == 0.5 * (
                    ((1 / model.nq[i]) * (model.V0 - model.v[t, s, i])) - (
                    (1 / model.mp[i]) * (model.w0 - model.w[t, s])))
        elif i in m.digenSet:
            return m.qg[t, s, i] <= 0.06
        else:
            return m.qg[t, s, i] == 0

    # Frequency & voltage dependent load constraints
    def ax_constraint_rule7(m, t, s, i):
        if i not in model.edSer:
            return m.pl[t, s, i] == m.c[t, i] * m.p0[t, s, i] * pow(m.v[t, s, i] / m.V0, m.alpha) * (
                    1 + m.KPF * (m.w[t, s] - m.w0))
        else:
            return m.pl[t, s, i] == ((model.r[t, i] * model.EP[t, i]) + m.p0[t, s, i]) * pow(m.v[t, s, i] / m.V0,
                                                                                             m.alpha) * (
                           1 + m.KPF * (m.w[t, s] - m.w0))

    def ax_constraint_rule8(m, t, s, i):
        if i not in model.edSer:
            return m.ql[t, s, i] == m.c[t, i] * m.q0[t, s, i] * pow(m.v[t, s, i] / m.V0, m.beta) * (
                    1 + m.KQF * (m.w[t, s] - m.w0))
        else:
            return m.ql[t, s, i] == ((model.r[t, i] * model.EQ[t, i]) + m.q0[t, s, i]) * pow(m.v[t, s, i] / m.V0,
                                                                                             m.beta) * (
                           1 + m.KQF * (m.w[t, s] - m.w0))

    # def maxGenCons(m, s, i):
    #     if i in m.drgenSet:
    #         return pyo.sqrt(pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2)) <= model.SGmax[i]
    #     else:
    #         return pyo.Constraint.Skip

    def maxGenCons(m, t, s, i):
        if i in m.drgenSet:
            return pow(m.pg[t, s, i], 2) + pow(m.qg[t, s, i], 2) <= pow(model.SGmax[i], 2)
        else:
            return pyo.Constraint.Skip

    model.cons3 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule3)
    model.cons4 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule4)
    model.cons5 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule5)
    model.cons6 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule6)
    model.cons7 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule7)
    model.cons8 = pyo.Constraint(model.T * model.S * model.J, rule=ax_constraint_rule8)
    model.cons20 = pyo.Constraint(model.T * model.S * model.J, rule=maxGenCons)

    model.name = "DroopControlledIMG"
    return model


# %%

def microgrid_solve(model):
    return SolverFactory('bonmin', executable="C:\\msys64\\home\\Administrator\\bonmin.exe").solve(model, tee=True)

def initialize30():
    pars30 = dict()
    pars30['drSet'] = {0, 1, 12, 21, 22, 26}
    # PRID = np.random.poisson(2, 30).copy()
    # PRID[np.argwhere(PRID == 0)] = 1
    # PRIS = np.random.poisson(4, 3).copy()
    # PRIS[np.argwhere(PRIS == 0)] = 1
    PRID = np.ones(30)
    PRIS = np.ones(30)
    pars30['PRID'] = PRID
    pars30['PRIS'] = PRIS
    pars30['withCon'] = [0.001, 0.001, 0.001]
    pars30['eSet'] = {2, 10, 11}
    pars30['renSet'] = {5, 9, 28}
    pars30['sSet'] = set(range(0,6))
    pars30['diSet']  = {}
    pars30['SGmax'] = dict(zip(pars30['drSet'], [0.05, 0.06, 0.05, 0.05, 0.06, 0.05]))


    return pars30

# %%
if __name__ == "__main__":
    from scenarioGeneration3 import createScenarios
    from dddas2022_agg import microgrid_model

    with open('datFiles\\tasks.txt', 'rb') as handle:
        data = handle.read()
    TASKS = pickle.loads(data)

    dfR = pd.read_csv('datFiles\\dat30R.csv', header=None, sep='\t')
    dfX = pd.read_csv('datFiles\\dat30X.csv', header=None, sep='\t')
    # red_scenes = np.loadtxt('datFiles\\red_scenes2.csv', delimiter=',')
    # red_probs = np.loadtxt('datFiles\\red_probs2.csv', delimiter=',')
    red_scenes, red_probs = createScenarios(1)

    P = len(red_scenes)
    # tiled_scenes = np.tile(red_scenes, (P, 1))  ## P is the number of periods
    # reshaped_tiled_scenes = np.reshape(tiled_scenes, (P, 6, 24))
    # tiled_probabilities = np.tile(red_probs, (P, 1))
    # reshaped_tiled_probabilities = np.reshape(tiled_probabilities, (P, 6))
    # results = sch2.jobshop(TASKS)
    # schedule = pd.DataFrame(results)
    # withCon = sch2.getConsumption(schedule)
    # modelM = microgrid_model(dfR, dfX, red_scenes, red_probs, withCon.apply(lambda x: x ** -1), PRID, PRIS)
    # modelM = microgrid_model(dfR, dfX, reshaped_tiled_scenes, reshaped_tiled_probabilities, withCon , PRID, PRIS)

    pars30 = initialize30()
    modelM = microgrid_model(dfR, dfX, red_scenes, red_probs, pars30)

    results = microgrid_solve(modelM)

    for i in modelM.r:
        print(" ", i, value(modelM.r[i]))

    for i in modelM.c:
        print(" ", i, value(modelM.c[i]))

    for parmobject in modelM.component_objects(pyo.Var, active=True):
        nametoprint = str(str(parmobject.name))
        print("Variable ", nametoprint)
        for index in parmobject:
            vtoprint = pyo.value(parmobject[index])
            print("   ", index, vtoprint)
