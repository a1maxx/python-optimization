import pandas as pd
import cmath
import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.environ import value, NonNegativeReals
import numpy as np

dfR = pd.read_csv('datFiles/dat33RM1.csv', header=None, sep='\t')
dfX = pd.read_csv('datFiles/dat33XM1.csv', header=None, sep='\t')
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

model = pyo.ConcreteModel()
model.N = pyo.Param(default=int(math.sqrt(len(dict_mag))))
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)

model.J = pyo.RangeSet(0, value(model.N) - 1)
model.drgenSet = pyo.Set(initialize={12, 14, 24, 32})
model.digenSet = pyo.Set(initialize={})
model.S = pyo.Set(initialize={0, 1, 2, 3, 4, 5})
model.renGen = pyo.Set(initialize={5, 9, 19, 28})


red_scenes = np.array([[1.36907560e-03, 1.54640271e-03, 1.56766085e-03, 1.50499852e-03,
        1.55982879e-03, 2.93229516e-02, 1.54711242e-03, 1.28637272e-03,
        1.61519029e-03, 0.00000000e+00, 1.56693575e-03, 1.70506424e-03,
        0.00000000e+00, 1.28900326e-03, 0.00000000e+00, 1.65271650e-03,
        1.62779322e-03, 1.36435136e-03, 1.32032944e-03, 7.25985380e-10,
        1.47263404e-03, 1.31912428e-03, 1.48273502e-03, 1.42207267e-03,
        0.00000000e+00, 1.36742916e-03, 1.47520962e-03, 1.42608546e-03,
        1.41272241e-02, 1.63529761e-03, 1.53800635e-03, 1.45586743e-03,
        0.00000000e+00],
       [1.39921704e-03, 1.46721944e-03, 1.31663129e-03, 1.46979191e-03,
        1.53278952e-03, 0.00000000e+00, 1.38032051e-03, 1.66945298e-03,
        1.51641760e-03, 0.00000000e+00, 1.58047121e-03, 1.51057545e-03,
        0.00000000e+00, 1.54407237e-03, 0.00000000e+00, 1.32640132e-03,
        1.52766109e-03, 1.46166691e-03, 1.41168136e-03, 6.39209250e-06,
        1.42243710e-03, 1.38051470e-03, 1.38945985e-03, 1.46045647e-03,
        0.00000000e+00, 1.36082889e-03, 1.36378970e-03, 1.42819112e-03,
        0.00000000e+00, 1.65549243e-03, 1.46511017e-03, 1.48456286e-03,
        0.00000000e+00],
       [1.61983354e-03, 1.54830266e-03, 1.36869751e-03, 1.48602966e-03,
        1.82483581e-03, 2.28499813e-04, 1.28330512e-03, 1.51884702e-03,
        1.63353467e-03, 0.00000000e+00, 1.73589625e-03, 1.59741433e-03,
        0.00000000e+00, 1.56317182e-03, 0.00000000e+00, 1.62610963e-03,
        1.65019190e-03, 1.50448843e-03, 1.59886549e-03, 6.02103206e-10,
        1.46962997e-03, 1.67192516e-03, 1.30891878e-03, 1.39723472e-03,
        0.00000000e+00, 1.39258399e-03, 1.60038697e-03, 1.55332831e-03,
        2.86834526e-02, 1.28589663e-03, 1.49715232e-03, 1.62272053e-03,
        0.00000000e+00],
       [1.51899344e-03, 1.50794776e-03, 1.65701861e-03, 1.43120087e-03,
        1.57652358e-03, 0.00000000e+00, 1.63410562e-03, 1.58899562e-03,
        1.51440683e-03, 8.22785901e-03, 1.58109415e-03, 1.63321051e-03,
        0.00000000e+00, 1.60680779e-03, 0.00000000e+00, 1.68957975e-03,
        1.44989626e-03, 1.38934498e-03, 1.45787376e-03, 1.92997407e-06,
        1.64715880e-03, 1.63590468e-03, 1.29068528e-03, 1.35934811e-03,
        0.00000000e+00, 1.40963064e-03, 1.55769680e-03, 1.45937317e-03,
        0.00000000e+00, 1.37872010e-03, 1.42011396e-03, 1.61847241e-03,
        0.00000000e+00],
       [1.45352653e-03, 1.45209667e-03, 1.36016255e-03, 1.46906591e-03,
        1.62070226e-03, 1.85998769e-02, 1.53436736e-03, 1.51886275e-03,
        1.60890966e-03, 1.84811571e-02, 1.33995886e-03, 1.43245596e-03,
        0.00000000e+00, 1.50214033e-03, 0.00000000e+00, 1.62681183e-03,
        1.47848938e-03, 1.28348512e-03, 1.37365496e-03, 1.12407886e-05,
        1.59404934e-03, 1.44139745e-03, 1.44036925e-03, 1.34375376e-03,
        0.00000000e+00, 1.48140658e-03, 1.38830824e-03, 1.57130501e-03,
        0.00000000e+00, 1.31715603e-03, 1.61739882e-03, 1.62395647e-03,
        0.00000000e+00],
       [1.56749384e-03, 1.67876452e-03, 1.47518838e-03, 1.39182731e-03,
        1.66411471e-03, 0.00000000e+00, 1.54426650e-03, 1.40795660e-03,
        1.40119108e-03, 0.00000000e+00, 1.28022889e-03, 1.60538204e-03,
        0.00000000e+00, 1.61283554e-03, 0.00000000e+00, 1.54931437e-03,
        1.46590150e-03, 1.49264517e-03, 1.45447718e-03, 8.35425652e-07,
        1.60770155e-03, 1.39931415e-03, 1.65124827e-03, 1.58622026e-03,
        0.00000000e+00, 1.58503249e-03, 1.49151124e-03, 1.56297890e-03,
        0.00000000e+00, 1.56991247e-03, 1.59374894e-03, 1.61569973e-03,
        0.00000000e+00]])


red_probs = np.array([0.40485334, 0.22507941, 0.15911502, 0.08346821, 0.06233838,
       0.06514564])


nScenarios = len(red_scenes)
a = np.zeros(nScenarios)
# renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
prenGen = dict()
qrenGen = dict()
#
# for i in model.J:
#     if i in model.drgenSet:
#         if i < len(red_scenes[0]):
#             red_scenes = np.insert(red_scenes, i, a, axis=1)
#         else:
#             red_scenes = np.append(red_scenes, np.zeros(shape=(nScenarios, 1)), axis=1)
PF = 0.328
for i in model.J:
    if i in model.renGen:
        for s in model.S:
            prenGen[s, i] = red_scenes[s, i]
            qrenGen[s, i] = red_scenes[s, i] * PF
        red_scenes[:, i] = np.zeros(nScenarios)

d1 = {}
for i in range(0, 6):
    for j in range(0, 33):
        d1[i, j] = red_scenes[i, j]


d2 = {}
for k, v in d1.items():
    d2[k] = v * PF

model.SPROBS = pyo.Param(model.S, initialize=red_probs)
model.p0 = pyo.Param(model.S * model.J, initialize=d1)
model.q0 = pyo.Param(model.S * model.J, initialize=d2)
d3, d4 = dict(), dict()

# for i in renGen:
#     for j in model.rengenSet:
#         d3[i, j] = renGen[i]
#         d4[i, j] = renGen[i] * PF

model.PR = pyo.Param(model.S * model.renGen, initialize=prenGen)
model.QR = pyo.Param(model.S * model.renGen, initialize=qrenGen)
model.w0 = pyo.Param(initialize=1.0)
model.V0 = pyo.Param(initialize=1.01)
model.SGmax = pyo.Param(model.drgenSet, initialize={12: 2.0, 14: 3.0, 24: 2.0, 32: 3.0})

model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

# model.ql = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.NonNegativeReals, bounds=(0, 0.3))
# model.pl = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.NonNegativeReals, bounds=(0, 0.3))
# model.pg = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.NonNegativeReals, bounds=(0, 0.3))
# model.qg = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.NonNegativeReals, bounds=(0, 0.3))

model.ql = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.Reals, bounds=(0, 0.3))
model.pl = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.Reals, bounds=(0, 0.3))
model.pg = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.Reals, bounds=(0, 0.3))
model.qg = pyo.Var(model.S * model.J, initialize=0.0015, within=pyo.Reals, bounds=(0, 0.3))

model.v = pyo.Var(model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.8, 1.2))
# model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))
model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=0.5, bounds=(-math.pi, math.pi ))

model.mp = pyo.Var(model.drgenSet, domain=pyo.Reals, initialize=-1, bounds=(-1.1, -0.04))
model.nq = pyo.Var(model.drgenSet, domain=pyo.Reals, initialize=-1, bounds=(-1.1, -0.04))

model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1, bounds=(0.980, 1.20))


def obj_expression(m):
    return sum(model.SPROBS[s] * sum((model.yMag[i, j] * model.v[s, i] * model.v[s, j]) *
                                     pyo.cos(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) +
                                     (-1 / 2) * (model.yMag[i, j] * model.v[s, i] * model.v[s, j]) * model.w[s] *
                                     pyo.sin(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) for j in m.J) for
               s, i in m.S * m.J)

# def obj_expression(m):
#     return sum(model.SPROBS[s] * pow((m.v[s, i] - 1.0), 2) for s, i in m.S * m.J)

#


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


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
    return m.pl[s, i] == m.p0[s, i] * pow(m.v[s, i] / m.V0, m.alpha) * (1 + m.KPF * (m.w[s] - m.w0))


def ax_constraint_rule8(m, s, i):
    return m.ql[s, i] == m.q0[s, i] * pow(m.v[s, i] / m.V0, m.beta) * (1 + m.KQF * (m.w[s] - m.w0))


def maxGenCons(m, s, i):
    if i in m.drgenSet:
        return pyo.sqrt(pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2)) <= model.SGmax[i]
    else:
        return pyo.Constraint.Skip


def dummyCons(m, i):
    return abs(m.mp[i]) >= 0.001


def dummyCons2(m, i):
    return abs(m.nq[i]) >= 0.001


model.cons3 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule8)
# model.cons20 = pyo.Constraint(model.S * model.J, rule=maxGenCons)
# model.cons25 = pyo.Constraint(model.drgenSet, rule=dummyCons)
# model.cons26 = pyo.Constraint(model.drgenSet, rule=dummyCons2)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
opt.options['acceptable_tol'] = 1e-4
# instance.pprint()
opt.options['max_iter'] = 10000

log_infeasible_constraints(model, log_expression=True, log_variables=True)
logging.basicConfig(filename='example2.log', level=logging.INFO)

results = opt.solve(model, tee=True)



# %%

model.display()
model.pprint()
model
for parmobject in model.component_objects(pyo.Param, active=True):
    nametoprint = str(str(parmobject.name))
    print("Parameter ", nametoprint)
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print("   ", index, vtoprint)

for parmobject in model.component_objects(pyo.Var, active=True):
    nametoprint = str(str(parmobject.name))
    print("Variable ", nametoprint)
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print("   ", index, vtoprint)



total = 0
total2 = 0

for i in model.drgenSet:
    total += pyo.value(model.mp[i])
    total2 += pyo.value(model.nq[i])

mpa = []
nqa = []

for i in model.drgenSet:
    mpa.append(value(model.mp[i]) / (total / 5))
    nqa.append(value(model.nq[i]) / (total2 / 7))

mpa
nqa

mps = np.array(mpa)
nqs = np.array(nqa)
pos = np.append(mps, nqs)
minuses = {3, 5, 7, 8, 10, 11}
pos2 = np.copy(pos)

for i in minuses:
    pos2[i] = -pos[i]

pos2


mpso = [1.589029, 0.034308, 1.138078, 0.919444, 0.050840, 1.144344]
sum(mpso)
nqso = [0.256885, 0.784892, 2.360074, 1.574549, 2.131491, 0.250231]
sum(nqso)



