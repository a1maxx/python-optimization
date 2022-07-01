import pandas as pd
import cmath
import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging

dfR = pd.read_csv('datFiles/datR6.csv', header=None)
dfX = pd.read_csv('datFiles/datX6.csv', header=None)
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
            dict_complex[i,j]= complex(0,0)

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
model.N = pyo.Param(default=6)
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)

model.J = pyo.RangeSet(0, model.N - 1)
model.drgenSet = pyo.Set(initialize={0, 4, 3})
model.digenSet = pyo.Set(initialize={1})


model.p0 = pyo.Param(model.J, initialize={0: 0, 1: 0.20, 2: 0.05, 3: 0.10, 4: 0.00, 5: 0.10})
model.q0 = pyo.Param(model.J, initialize={0: 0, 1: 0.12, 2: 0.48, 3: 0.04, 4: 0.00, 5: 0.06})
model.w0 = pyo.Param(initialize=1.0)
model.V0 = pyo.Param(initialize=1.01)
model.SGmax = pyo.Param(model.drgenSet, initialize=1.0)
model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

model.ql = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pl = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.qg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.5, 1.5))
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))

model.mp = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.03, bounds=(0, 1))
model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(0, 1))

model.w = pyo.Var(domain=pyo.NonNegativeReals, initialize=1)


def obj_expression(m):
    return sum(pow((m.v[i] - 1.0), 2) for i in m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def ax_constraint_rule3(m, i):
    return (m.pg[i] - m.pl[i]) - sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule4(m, i):
    return (m.qg[i] - m.ql[i]) + sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule5(m, i):
    if i in m.drgenSet:
        return m.pg[i] == (1 / m.mp[i]) * (m.w0 - m.w)
    elif i in m.digenSet:
        return m.pg[i] <= 0.1
    else:
        return m.pg[i] == 0


def ax_constraint_rule6(m, i):
    if i in m.drgenSet:
        return m.qg[i] == (1 / m.nq[i]) * (m.V0 - m.v[i])
    elif i in m.digenSet:
        return m.qg[i] <= 0.06
    else:
        return m.qg[i] == 0


# Frequency & voltage dependent load constraints
def ax_constraint_rule7(m, i):
    return m.pl[i] == m.p0[i] * pow(m.v[i] / m.V0, m.alpha) * (1 + m.KPF * (m.w - m.w0))


def ax_constraint_rule8(m, i):
    return m.ql[i] == m.q0[i] * pow(m.v[i] / m.V0, m.beta) * (1 + m.KQF * (m.w - m.w0))


def maxGenCons(m, i):
    if i in m.drgenSet:
        return pyo.sqrt(pow(m.pg[i], 2) + pow(m.qg[i], 2)) <= 1
    else:
        return pyo.Constraint.Skip


def maxwCons(m):
    return m.w <= 1.005


def minwCons(m):
    return m.w >= 0.995


model.cons3 = pyo.Constraint(model.J, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.J, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.J, rule=ax_constraint_rule8)
model.cons20 = pyo.Constraint(model.J, rule=maxGenCons)
model.cons21 = pyo.Constraint(rule=maxwCons)
model.cons22 = pyo.Constraint(rule=minwCons)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
opt.options['acceptable_tol'] = 1e-3
# instance.pprint()
opt.options['max_iter'] = 100000000

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

(pyo.value(model.V0) - pyo.value(model.v[1])) * (1 / pyo.value(model.nq[1]))
pyo.value((1 / model.nq[1]) * (model.V0 - model.v[1]))
pyo.value(model.qg[1])
