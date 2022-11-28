import pandas as pd
import cmath
import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.environ import value, NonNegativeReals, TransformationFactory
import numpy as np
from scenarioGeneration3 import initialize6, createScenariosN, windPow, initialize12, initialize33

import random

random.seed(10)

# 3-Bus

# dfR = pd.read_csv('datFiles/dat3R.txt', header=None, sep=' ')
# dfX = pd.read_csv('datFiles/dat3X.txt', header=None, sep=' ')
# pars = initialize3()

# 6 - Bus

# dfR = pd.read_csv('datFiles/datR6.csv', header=None)
# dfX = pd.read_csv('datFiles/datX6.csv', header=None)
# pars = initialize6()

# 12- Bus
# dfR = pd.read_csv('datFiles/datR12.txt', header=None, sep='\t')
# dfX = pd.read_csv('datFiles/datX12.txt', header=None, sep='\t')
# pars = initialize12()

# 33- Bus
dfR = pd.read_csv('datFiles/dat33R.csv', header=None, sep='\t')
dfX = pd.read_csv('datFiles/dat33X.csv', header=None, sep='\t')
pars = initialize33()

edges = []

dict_R = dict()
dict_X = dict()
for i in range(0, dfR.shape[0]):
    edges.append((i, i))
    for j in range(0, dfR.shape[1]):
        dict_R[i, j] = dfR.iloc[i, j]
        dict_X[i, j] = dfX.iloc[i, j]
        if dfR.iloc[i, j] != 0:
            edges.append((i, j))

del i, j

red_scenes, red_probs = createScenariosN(1, pars)

model = pyo.ConcreteModel()
model.N = pyo.Param(default=int(math.sqrt(len(dict_R))))
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)
model.J = pyo.RangeSet(0, value(model.N) - 1)
model.drgenSet = pyo.Set(initialize=pars['drSet'])
model.digenSet = pyo.Set(initialize=pars['diSet'])
model.renGen = pyo.Set(initialize=pars['renSet'])

model.R = pyo.Param(model.J, model.J, initialize=dict_R)
model.X = pyo.Param(model.J, model.J, initialize=dict_X)

model.S = pyo.Set(initialize=list(range(len(red_probs))))

nScenarios = len(red_scenes)

# renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
prenGen = dict()
qrenGen = dict()

PF = 0.328
for i in model.J:
    if i in model.renGen:
        for s in model.S:
            prenGen[s, i] = red_scenes[s, i]
            qrenGen[s, i] = red_scenes[s, i] * PF
del i

d1 = {}
for i in range(0, nScenarios):
    for j in range(0, len(red_scenes[0])):
        if j not in model.renGen:
            d1[i, j] = red_scenes[i, j]
        else:
            d1[i, j] = 0

d2 = {}
for k, v in d1.items():
    d2[k] = v * PF
del i, j, s, k

model.SPROBS = pyo.Param(model.S, initialize=red_probs)
model.p0 = pyo.Param(model.S * model.J, initialize=d1)
model.q0 = pyo.Param(model.S * model.J, initialize=d2)
model.PR = pyo.Param(model.S * model.renGen, initialize=prenGen)
model.QR = pyo.Param(model.S * model.renGen, initialize=qrenGen)
model.w0 = pyo.Param(initialize=1.00)
model.V0 = pyo.Param(initialize=1.01)
# model.SGmax = pyo.Param(model.drgenSet, {3: 1, 5: 1, 9: 1, 11: 1})
model.SGmax = pyo.Param(model.drgenSet, initialize=pars['SGmax'])

model.yReal = pyo.Var(model.J, model.J, model.S, initialize=1.0, within=pyo.Reals)
model.yIm = pyo.Var(model.J, model.J, model.S, initialize=1.0, within=pyo.Reals)
model.yMag = pyo.Var(model.J, model.J, model.S, initialize=1.0, within=pyo.Reals)
model.yThe = pyo.Var(model.J, model.J, model.S, initialize=1.0, within=pyo.Reals, bounds=(-1 * math.pi, 1 * math.pi))

model.ql = pyo.Var(model.S * model.J, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pl = pyo.Var(model.S * model.J, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pg = pyo.Var(model.S * model.J, within=pyo.NonNegativeReals, bounds=(0, 2))
model.qg = pyo.Var(model.S * model.J, within=pyo.NonNegativeReals, bounds=(0, 2))

model.v = pyo.Var(model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.95, 1.05))
model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=0.5, bounds=(-1 / 2 * math.pi, 1 / 2 * math.pi))
# model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=0.5, bounds=(-1 * math.pi, 1 * math.pi))

# model.mp = pyo.Var(model.drgenSet, domain=pyo.Reals, initialize=0.80, bounds=(-30, 30))
model.mp = pyo.Param(model.drgenSet, initialize={12: 0.8, 14: 1.9, 24: 4.5, 32: 2.8})
model.nq = pyo.Var(model.drgenSet, domain=pyo.Reals, initialize=5, bounds=(-4, 4))

# model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1.00, bounds=(0.99, 1.01))
model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1.00, bounds=(0.995, 1.005))


def obj_expression(m):
    return sum(model.SPROBS[s] * 0.5 * sum(
        ((model.yMag[i, j, s] * model.v[s, i] * model.v[s, j]) * pyo.cos(-m.d[s, i] + m.d[s, j] + m.yThe[i, j, s])) +
        ((model.yMag[i, j, s] * model.v[s, i] * model.v[s, j]) * pyo.cos(m.d[s, i] - m.d[s, j] + m.yThe[i, j, s])) for j
        in m.J if (i, j) in edges) - 0.5 *
               sum(((model.yMag[i, j, s] * model.v[s, i] * model.v[s, j]) * pyo.sin(
                   -m.d[s, i] + m.d[s, j] + m.yThe[i, j, s])) +
                   ((model.yMag[i, j, s] * model.v[s, i] * model.v[s, j]) * pyo.sin(
                       m.d[s, i] - m.d[s, j] + m.yThe[i, j, s])) for j
                   in m.J if (i, j) in edges)
               for s, i in m.S * m.J)


# def obj_expression(m):
#     return 1
# def obj_expression(m):
#     return sum(model.SPROBS[s] * pow((m.v[s, i] - 1.0), 2) for s, i in m.S * m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


# def ax_constraint_rule3(m, s, i):
#     if i in m.renGen:
#         return (m.pg[s, i] + m.PR[s, i] - m.pl[s, i]) - sum(
#             m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.cos(m.yThe[i, j, s] + m.d[s, j] - m.d[s, i]) for j in
#             m.J if (i, j) in edges) == 0
#     else:
#         return (m.pg[s, i] - m.pl[s, i]) - sum(
#             m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.cos(m.yThe[i, j, s] + m.d[s, j] - m.d[s, i]) for j in
#             m.J if (i, j) in edges) == 0

def ax_constraint_rule3(m, s, i):
    if i in m.renGen:
        return (m.pg[s, i] + m.PR[s, i] - m.pl[s, i]) - sum(
            m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.cos(-m.yThe[i, j, s] - m.d[s, j] + m.d[s, i]) for j in
            m.J if (i, j) in edges) == 0
    else:
        return (m.pg[s, i] - m.pl[s, i]) - sum(
            m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.cos(-m.yThe[i, j, s] - m.d[s, j] + m.d[s, i]) for j in
            m.J if (i, j) in edges) == 0


def ax_constraint_rule4(m, s, i):
    if i in m.renGen:
        return (m.qg[s, i] + m.QR[s, i] - m.ql[s, i]) - sum(
            m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.sin(-m.yThe[i, j, s] - m.d[s, j] + m.d[s, i]) for j in
            m.J if (i, j) in edges) == 0
    else:
        return (m.qg[s, i] - m.ql[s, i]) - sum(
            m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.sin(-m.yThe[i, j, s] - m.d[s, j] + m.d[s, i]) for j in
            m.J if (i, j) in edges) == 0


# def ax_constraint_rule4(m, s, i):
#     if i in m.renGen:
#         return (m.qg[s, i] + m.QR[s, i] - m.ql[s, i]) + sum(
#             m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.sin(m.yThe[i, j, s] + m.d[s, j] - m.d[s, i]) for j in
#             m.J if (i, j) in edges) == 0
#     else:
#         return (m.qg[s, i] - m.ql[s, i]) + sum(
#             m.v[s, i] * m.v[s, j] * m.yMag[i, j, s] * pyo.sin(m.yThe[i, j, s] + m.d[s, j] - m.d[s, i]) for j in
#             m.J if (i, j) in edges) == 0


def ax_constraint_rule5(m, s, i):
    if i in m.drgenSet:
        return m.pg[s, i] == 0.5 * (
                ((1 / model.mp[i]) * (model.w0 - model.w[s])) + ((1 / model.nq[i]) * (model.V0 - model.v[s, i])))
    elif i in m.digenSet:
        return m.pg[s, i] <= 0.1
    else:
        return m.pg[s, i] == 0


def ax_constraint_rule6(m, s, i):
    if i in m.drgenSet:
        return m.qg[s, i] == 0.5 * (
                ((1 / model.nq[i]) * (model.V0 - model.v[s, i])) - ((1 / model.mp[i]) * (model.w0 - model.w[s])))
    elif i in m.digenSet:
        return m.qg[s, i] <= 0.06
    else:
        return m.qg[s, i] == 0


# Frequency & voltage dependent load constraints
def ax_constraint_rule7(m, s, i):
    return m.pl[s, i] == m.p0[s, i] * pow(m.v[s, i] / m.V0, m.alpha) * (1 + m.KPF * (m.w[s] - m.w0))


def ax_constraint_rule8(m, s, i):
    return m.ql[s, i] == m.q0[s, i] * pow(m.v[s, i] / m.V0, m.beta) * (1 + m.KQF * (m.w[s] - m.w0))


def maxGenCons(m, s, i):
    if i in m.drgenSet:
        return pyo.sqrt(pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2)) <= model.SGmax[i]
    else:
        return pyo.Constraint.Skip


def admittanceReal(m, i, j, s):
    if i != j and m.R[i, j] != 0:
        return m.yReal[i, j, s] == -m.R[i, j] / (m.R[i, j] ** 2 + (m.X[i, j] * m.w[s]) ** 2)
    elif i != j and m.R[i, j] == 0:
        return m.yReal[i, j, s] == 0
    else:
        return pyo.Constraint.Skip


def admittanceIm(m, i, j, s):
    if i != j and m.R[i, j] != 0:
        return m.yIm[i, j, s] == (m.X[i, j] * m.w[s]) / (m.R[i, j] ** 2 + (m.X[i, j] * m.w[s]) ** 2)
    elif i != j and m.R[i, j] == 0:
        return m.yIm[i, j, s] == 0
    else:
        return pyo.Constraint.Skip


def admittanceDiagReal(m, i, j, s):
    if i == j:
        return m.yReal[i, j, s] == sum(-m.yReal[i, f, s] for f in [i for i in model.J] if f != i and (i, f) in edges)
    else:
        return pyo.Constraint.Skip


def admittanceDiagIm(m, i, j, s):
    if i == j:
        return m.yIm[i, j, s] == sum(-m.yIm[i, f, s] for f in [i for i in model.J] if f != i and (i, f) in edges)
    else:
        return pyo.Constraint.Skip


def admittanceMag(m, i, j, s):
    return m.yMag[i, j, s] == pyo.sqrt(m.yReal[i, j, s] ** 2 + m.yIm[i, j, s] ** 2)


def admittanceThe(m, i, j, s):
    if i == j:
        return m.yThe[i, j, s] == pyo.atan(m.yIm[i, j, s] / m.yReal[i, j, s])
    elif i != j and m.R[i, j] != 0:
        return m.yThe[i, j, s] == pyo.atan(m.yIm[i, j, s] / m.yReal[i, j, s]) + math.pi
    else:
        return m.yThe[i, j, s] == 0


model.cons3 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule8)
# model.cons20 = pyo.Constraint(model.S * model.J, rule=maxGenCons)
model.cons13 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceReal)
model.cons14 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceIm)
model.cons15 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceDiagReal)
model.cons16 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceDiagIm)
model.cons17 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceMag)
model.cons19 = pyo.Constraint(model.J * model.J * model.S, rule=admittanceThe)

model.name = "DroopControlledIMG"

TransformationFactory('contrib.aggregate_vars').apply_to(model)
TransformationFactory('contrib.constraints_to_var_bounds').apply_to(model)
# TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(model)
# TransformationFactory('contrib.detect_fixed_vars').apply_to(model)
# TransformationFactory('contrib.propagate_fixed_vars').apply_to(model)
TransformationFactory('contrib.init_vars_midpoint').apply_to(model)
# TransformationFactory('contrib.propagate_zero_sum').apply_to(model)

opt = pyo.SolverFactory("ipopt")
opt.options['acceptable_tol'] = 1e-2
# instance.pprint()
opt.options['max_iter'] = 100000

# log_infeasible_constraints(model, log_expression=True, log_variables=True)
# logging.basicConfig(filename='example2.log', level=logging.INFO)


results = opt.solve(model, tee=True)

# %%

# model.display()
# model.pprint()
# model
# for parmobject in model.component_objects(pyo.Param, active=True):
#     nametoprint = str(str(parmobject.name))
#     print("Parameter ", nametoprint)
#     for index in parmobject:
#         vtoprint = pyo.value(parmobject[index])
#         print("   ", index, vtoprint)
#
# for parmobject in model.component_objects(pyo.Var, active=True):
#     nametoprint = str(str(parmobject.name))
#     print("Variable ", nametoprint)
#     for index in parmobject:
#         vtoprint = pyo.value(parmobject[index])
#         print("   ", index, vtoprint)

np.concatenate((np.array([value(model.mp[i]) for i in model.mp]).round(6),
                np.array([value(model.nq[i]) for i in model.nq]).round(6)))

mpso = np.array([0.152746, -0.082997, 0.810816, -0.001912])
s1 = sum(abs(mpso))
nqso = np.array([0.775256, 1.088017, 0.097911, 0.930320])
s2 = sum(abs(nqso))

# mpso = np.array([0.001602,	1.914428,	-3.033656,	-4.543417])
# s1 = sum(mpso)
# nqso = np.array([0.106867,	0.053463,	0.890619,	0.555316])
# s2 = sum(nqso)

total = 0
total2 = 0

for i in model.drgenSet:
    total += abs(pyo.value(model.mp[i]))
    total2 += abs(pyo.value(model.nq[i]))

mpa = []
nqa = []
# sca = [1,1]
# for i in model.drgenSet:
#     mpa.append(abs(value(model.mp[i])) / (total / (s1*sca[0])))
#     nqa.append(abs(value(model.nq[i])) / (total2 / (s2*sca[1])))

sca = [1.5, 1]
for i in model.drgenSet:
    mpa.append(abs(value(model.mp[i])) / (total / sca[0]))
    nqa.append(abs(value(model.nq[i])) / (total2 / sca[1]))

mpa = list(np.array(mpa).round(3))
nqa = list(np.array(nqa).round(3))
mpa
nqa

for i in range(len(mpa)):
    if i in np.ravel(np.argwhere(mpso < 0)):
        mpa[i] = -mpa[i]

for i in range(len(mpa)):
    if i in np.ravel(np.argwhere(nqso < 0)):
        nqa[i] = -nqa[i]

mpa
nqa

#
# mpso = [1.589029, 0.034308, 1.138078, 0.919444, 0.050840, 1.144344]
# nqso = [0.256885, 0.784892, 2.360074, 1.574549, 2.131491, 0.250231]

# Key: Lower: Value: Upper: Fixed: Stale: Domain
# 3: -2:  0.08385836049914892: 2: False: False:  Reals
# 5: -2:   0.7497146270549877: 2: False: False:  Reals
# 9: -2: -0.01889586515019138: 2: False: False:  Reals
# 11: -2:   0.3259333401171334: 2: False: False:  Reals
#
# 3: -2: -0.01558202993382473: 2: False: False:  Reals
# 5: -2:  0.06282018664824221: 2: False: False:  Reals
# 9: -2:  -1.5812876344131355: 2: False: False:  Reals
# 11: -2:   0.7545275014515547: 2: False: False:  Reals


## Below are the results for 30 scenarios 6 bus system

# mp : Size=3, Index=drgenSet
#     Key : Lower : Value              : Upper : Fixed : Stale : Domain
#       0 :    -5 : 0.1891422461701308 :     5 : False : False :  Reals
#       1 :    -5 : 0.0758050561883269 :     5 : False : False :  Reals
#       5 :    -5 : 0.1025098663708134 :     5 : False : False :  Reals
# model.nq.pprint()
# nq : Size=3, Index=drgenSet
#     Key : Lower : Value                : Upper : Fixed : Stale : Domain
#       0 :    -5 :  -0.2612487064116036 :     5 : False : False :  Reals
#       1 :    -5 : -0.10818691672988243 :     5 : False : False :  Reals
#       5 :    -5 : -0.14453103747610807 :     5 : False : False :  Reals




# p : Size=4, Index=drgenSet, Domain=Any, Default=None, Mutable=False
#     Key : Value
#      12 :   0.8
#      14 :   1.9
#      24 :   4.5
#      32 :   2.8
# model.nq.pprint()
# nq : Size=4, Index=drgenSet
#     Key : Lower : Value               : Upper : Fixed : Stale : Domain
#      12 :   -30 : -12.071475878477672 :    30 : False : False :  Reals
#      14 :   -30 :  -19.97593128977721 :    30 : False : False :  Reals
#      24 :   -30 :  -7.580673255202304 :    30 : False : False :  Reals
#      32 :   -30 : -17.949231189444653 :    30 : False : False :  Reals



{0.8,1.9,4.5,2.8,-12.07,-19.97,-7.57,-17.94}