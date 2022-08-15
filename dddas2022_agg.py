import pandas as pd
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
import Scheduling2 as sch2


def microgrid_model(dfR, dfX, red_scenes, red_probs, withCon, PRID, PRIS):
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
    model.KPF = pyo.Param(initialize=1)
    model.KQF = pyo.Param(initialize=-1)
    model.alpha = pyo.Param(initialize=1)
    model.beta = pyo.Param(initialize=1)

    model.J = pyo.RangeSet(0, value(model.N) - 1)
    model.drgenSet = pyo.Set(initialize={0, 1, 12, 21, 22, 26})
    model.digenSet = pyo.Set(initialize={})
    model.S = pyo.Set(initialize={0, 1, 2, 3, 4, 5})
    model.renGen = pyo.Set(initialize={5, 9, 28})
    edge_servers = set(sorted([2, 10, 11]))
    model.edSer = pyo.Set(initialize=edge_servers)



    nScenarios = len(red_scenes)
    a = np.zeros(nScenarios)
    # renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
    prenGen = dict()
    qrenGen = dict()

    for i in model.J:
        if i in model.drgenSet:
            if i < len(red_scenes[0]):
                red_scenes = np.insert(red_scenes, i, a, axis=1)
            else:
                red_scenes = np.append(red_scenes, np.zeros(shape=(nScenarios, 1)), axis=1)

    for i in model.J:
        if i in model.renGen:
            for s in model.S:
                prenGen[s, i] = red_scenes[s, i]
                qrenGen[s, i] = red_scenes[s, i] * 0.6
            red_scenes[:, i] = np.zeros(nScenarios)

    # red_scenesPE = red_scenes[:, edge_servers]

    d1 = {}
    for i in range(0, 6):
        for j in range(0, 30):
            d1[i, j] = red_scenes[i, j] + (j in model.edSer) * 0

    PF = 0.6
    d2 = {}
    for k, v in d1.items():
        d2[k] = v * PF

    d3 = {}
    j = 0

    for i in model.edSer:
        d3[i] = withCon[j] * 1.5
        j += 1

    d4 = {}
    for k, v in d3.items():
        d4[k] = v * PF

    d5 = {}
    for k in zip(model.edSer, PRIS):
        d5[k[0]] = k[1]

    model.DPRI = pyo.Param(model.J, initialize=PRID)
    model.SPRI = pyo.Param(model.edSer, initialize=d5)
    model.EP = pyo.Param(model.edSer, initialize=d3)
    model.EQ = pyo.Param(model.edSer, initialize=d4)
    model.SPROBS = pyo.Param(model.S, initialize=red_probs)
    model.p0 = pyo.Param(model.S * model.J, initialize=d1)
    model.q0 = pyo.Param(model.S * model.J, initialize=d2)
    d3, d4 = dict(), dict()

    model.PR = pyo.Param(model.S * model.renGen, initialize=prenGen)
    model.QR = pyo.Param(model.S * model.renGen, initialize=qrenGen)
    model.w0 = pyo.Param(initialize=1.0)
    model.V0 = pyo.Param(initialize=1.01)
    model.SGmax = pyo.Param(model.drgenSet, initialize={0: 0.05, 1: 0.06, 12: 0.05, 21: 0.05, 22: 0.06, 26: 0.05})

    model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
    model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

    model.ql = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.pl = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.pg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.qg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

    model.r = pyo.Var(model.edSer, initialize=0, within=pyo.Binary)
    model.c = pyo.Var(model.J, initialize=1, within=pyo.Binary)

    model.v = pyo.Var(model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.95, 1.05))
    model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))

    model.mp = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.03, bounds=(1e-10, 1))
    model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(1e-10, 1))

    model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1, bounds=(0.995, 1.005))

    def obj_expression(m):
        return sum(model.r[i] * model.SPRI[i] for i in m.edSer) + sum(model.c[i] * model.DPRI[i] for i in model.J)

    model.o = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

    def ax_constraint_rule3(m, s, i):
        if i in m.renGen:
            return (m.pg[s, i] + m.PR[s, i] - m.pl[s, i]) - sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0
        else:
            return (m.pg[s, i] - m.pl[s, i]) - sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0

    def ax_constraint_rule4(m, s, i):
        if i in m.renGen:
            return (m.qg[s, i] + m.QR[s, i] - m.ql[s, i]) + sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0
        else:
            return (m.qg[s, i] - m.ql[s, i]) + sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0

    def ax_constraint_rule5(m, s, i):
        if i in m.drgenSet:
            # return m.pg[s, i] == (1 / m.mp[i]) * (m.w0 - m.w[s])
            return m.pg[s, i] == 0.5 * (
                    ((1 / model.mp[i]) * (model.w0 - model.w[s])) + ((1 / model.nq[i]) * (model.V0 - model.v[s, i])))
        elif i in m.digenSet:
            return m.pg[s, i] <= 0.1
        else:
            return m.pg[s, i] == 0

    def ax_constraint_rule6(m, s, i):
        if i in m.drgenSet:
            # return m.qg[s, i] == (1 / m.nq[i]) * (m.V0 - m.v[s, i])
            return m.qg[s, i] == 0.5 * (
                    ((1 / model.nq[i]) * (model.V0 - model.v[s, i])) - ((1 / model.mp[i]) * (model.w0 - model.w[s])))
        elif i in m.digenSet:
            return m.qg[s, i] <= 0.06
        else:
            return m.qg[s, i] == 0

    # Frequency & voltage dependent load constraints
    def ax_constraint_rule7(m, s, i):
        if i not in model.edSer:
            return m.pl[s, i] == m.c[i] * m.p0[s, i] * pow(m.v[s, i] / m.V0, m.alpha) * (1 + m.KPF * (m.w[s] - m.w0))
        else:
            return m.pl[s, i] == ((model.r[i] * model.EP[i]) + m.p0[s, i]) * pow(m.v[s, i] / m.V0, m.alpha) * (
                    1 + m.KPF * (m.w[s] - m.w0))

    def ax_constraint_rule8(m, s, i):
        if i not in model.edSer:
            return m.ql[s, i] == m.c[i] * m.q0[s, i] * pow(m.v[s, i] / m.V0, m.beta) * (1 + m.KQF * (m.w[s] - m.w0))
        else:
            return m.ql[s, i] == ((model.r[i] * model.EQ[i]) + m.q0[s, i]) * pow(m.v[s, i] / m.V0, m.beta) * (
                    1 + m.KQF * (m.w[s] - m.w0))

    # def maxGenCons(m, s, i):
    #     if i in m.drgenSet:
    #         return pyo.sqrt(pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2)) <= model.SGmax[i]
    #     else:
    #         return pyo.Constraint.Skip

    def maxGenCons(m, s, i):
        if i in m.drgenSet:
            return pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2) <= pow(model.SGmax[i], 2)
        else:
            return pyo.Constraint.Skip

    model.cons3 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule3)
    model.cons4 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule4)
    model.cons5 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule5)
    model.cons6 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule6)
    model.cons7 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule7)
    model.cons8 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule8)
    model.cons20 = pyo.Constraint(model.S * model.J, rule=maxGenCons)

    model.name = "DroopControlledIMG"
    return model


def microgrid_solve(model):
    return SolverFactory('bonmin').solve(model)


if __name__ == "__main__":
    with open('datFiles\\tasks.txt', 'rb') as handle:
        data = handle.read()
    TASKS = pickle.loads(data)
    results = sch2.jobshop(TASKS)
    schedule = pd.DataFrame(results)
    withCon = sch2.getConsumption(schedule)

    dfR = pd.read_csv('datFiles/dat30R.csv', header=None, sep='\t')
    dfX = pd.read_csv('datFiles/dat30X.csv', header=None, sep='\t')
    red_scenes = np.loadtxt('datFiles\\red_scenes2.csv', delimiter=',')
    red_probs = np.loadtxt('datFiles\\red_probs2.csv', delimiter=',')


    PRID = np.random.poisson(2, 30).copy()
    PRID[np.argwhere(PRID == 0)] = 1
    PRIS = np.random.poisson(4, 3).copy()
    PRIS[np.argwhere(PRIS == 0)] = 1
    modelM = microgrid_model(dfR, dfX, red_scenes,red_probs,withCon,PRID,PRIS)
    microgrid_solve(modelM)


