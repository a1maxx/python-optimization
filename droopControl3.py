import pandas as pd
import cmath
import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.environ import value, NonNegativeReals
import numpy as np

#            Demand 1  Demand 2  Demand 3  WindPower     Probs
# Scenarios
# 952.0      0.147230  0.187622  0.098033   0.027910  0.151260
# 682.0      0.186294  0.170627  0.197606   0.000000  0.268751
# 288.0      0.176485  0.076991  0.183314   0.056116  0.014465
# 685.0      0.171678  0.185415  0.147836   0.000000  0.364485
# 414.0      0.214218  0.174082  0.183267   0.007031  0.201038

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
model.drgenSet = pyo.Set(initialize={3, 5})
model.digenSet = pyo.Set(initialize={})
model.S = pyo.Set(initialize={0, 1, 2, 3, 4, 5})
model.renGen = pyo.Set(initialize={4})

# Original
# red_scenes = np.array([[0.16741245, 0.15497062, 0.16618976, 0.],
#                        [0.22163376, 0.36566567, 0.2472521, 0.],
#                        [0.38218843, 0.16201301, 0.1541336, 0.006174],
#                        [0.10972745, 0.14715076, 0.37857077, 0.00468016],
#                        [0.08370557, 0.36245649, 0.05029941, 0.],
#                        [0.1204056, 0.09385167, 0.03811169, 0.]])

red_scenes = np.array([[0.16741245, 0.15497062, 0.16618976, 0.],
                       [0.22163376, 0.36566567, 0.2472521, 0.],
                       [0.38218843, 0.16201301, 0.1541336, 0.006174],
                       [0.10972745, 0.14715076, 0.37857077, 0.00468016],
                       [0.08370557, 0.36245649, 0.05029941, 0.],
                       [0.1204056, 0.09385167, 0.03811169, 0.]])
red_probs = np.array([0.16288595972609665,
                      0.16891265078817608,
                      0.18907038769884246,
                      0.17893877297863156,
                      0.11370774649914571,
                      0.18648448230910744])

nScenarios = len(red_scenes)
a = np.zeros(nScenarios)
# renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
prenGen = dict()
qrenGen = dict()

for i in model.J:
    if i in model.drgenSet:
        red_scenes = np.insert(red_scenes, i, a, axis=1)

for i in model.J:
    if i in model.renGen:
        for s in model.S:
            prenGen[s, i] = red_scenes[s, i]
            qrenGen[s, i] = red_scenes[s, i] * 0.6
        red_scenes[:, i] = np.zeros(nScenarios)

# red_scenes = np.insert(red_scenes, 3, a, axis=1)
# red_scenes = np.insert(red_scenes, 5, a, axis=1)
# renGen = red_scenes[:, 4].copy()
# red_scenes[:, 4] = np.zeros(6)
d1 = {}
for i in range(0, 6):
    for j in range(0, 6):
        d1[i, j] = red_scenes[i, j]

# d1 = {(0, 0): 0.147, (0, 1): 0.187, (0, 2): 0.090, (0, 3): 0.00, (0, 4): 0.02, (0, 5): 0.0,
#       (1, 0): 0.186, (1, 1): 0.170, (1, 2): 0.197, (1, 3): 0.00, (1, 4): 0.00, (1, 5): 0.0,
#       (2, 0): 0.176, (2, 1): 0.076, (2, 2): 0.183, (2, 3): 0.00, (2, 4): 0.05, (2, 5): 0.0,
#       (3, 0): 0.171, (3, 1): 0.185, (3, 2): 0.147, (3, 3): 0.00, (3, 4): 0.00, (3, 5): 0.0,
#       (4, 0): 0.214, (4, 1): 0.174, (4, 2): 0.183, (4, 3): 0.00, (4, 4): 0.007, (4, 5): 0.0
#       }
PF = 0.6
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
model.SGmax = pyo.Param(model.drgenSet, initialize={3: 1, 5: 1.5})
model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

model.ql = pyo.Var(model.S * model.J, initialize=0, within=NonNegativeReals, bounds=(0, 2))
model.pl = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.qg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

model.v = pyo.Var(model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.5, 1.5))
model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))

model.mp = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.03, bounds=(0, 1))
model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(0, 1))

model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1)


def obj_expression(m):
    return sum(model.SPROBS[s] * sum((model.yMag[i, j] * model.v[s, i] * model.v[s, j]) *
                                     pyo.cos(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) +
                                     (-1 / 2) * (model.yMag[i, j] * model.v[s, i] * model.v[s, j]) *
                                     pyo.sin(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) for j in m.J) for
               s, i in m.S * m.J)


# def obj_expression(m):
#     return sum(model.SPROBS[s] * pow((m.v[s, i] - 1.0), 2) for s, i in m.S * m.J)


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


def maxwCons(m, s):
    return m.w[s] <= 1.005


def minwCons(m, s):
    return m.w[s] >= 0.995


def minvCons(m, s, i):
    return m.v[s, i] >= 0.95


def maxvCons(m, s, i):
    return m.v[s, i] <= 1.05


def dummyCons(m, i):
    return m.mp[i] >= 0.001


def dummyCons2(m, i):
    return m.nq[i] >= 0.01


model.cons3 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule8)
model.cons20 = pyo.Constraint(model.S * model.J, rule=maxGenCons)
model.cons21 = pyo.Constraint(model.S, rule=maxwCons)
model.cons22 = pyo.Constraint(model.S, rule=minwCons)
model.cons23 = pyo.Constraint(model.S, model.J, rule=minvCons)
model.cons24 = pyo.Constraint(model.S, model.J, rule=maxvCons)
model.cons25 = pyo.Constraint(model.drgenSet, rule=dummyCons)
model.cons26 = pyo.Constraint(model.drgenSet, rule=dummyCons2)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("scip")
# opt.options['acceptable_tol'] = 1e-3
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

total = 0
total2 = 0
for i in model.drgenSet:
    total += pyo.value(model.mp[i])
    total2 += pyo.value(model.nq[i])

mpa = []
nqa = []

for i in model.drgenSet:
    mpa.append(value(model.mp[i]) / total)
    nqa.append(value(model.nq[i]) / total2)

mpa
nqa

# pLoss = model.yMag[i,j] * model.v[s,i] *model.v[s,j]
# qLoss = -1/2 * (model.yThe[i,j] - model.d[s,i] * 2)

# %%

from pyomo.environ import ConcreteModel, minimize, SolverFactory, Set, \
    Param, Var, Constraint, Objective, \
    TransformationFactory
from pyomo.gdp import Disjunction
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

model = ConcreteModel()


edges = {(1, 2): {'band': 2},
         (1, 3): {'band': 2},
         (2, 4): {'band': 1},
         (3, 5): {'band': 2},
         (4, 6): {'band': 4},
         (5, 6): {'band': 2}
         }

bc = {(1, 2): {'uc': 2},
      (1, 3): {'uc': 2},
      (2, 4): {'uc': 1},
      (3, 5): {'uc': 2},
      (4, 6): {'uc': 4},
      (5, 6): {'uc': 2}
      }

qos = {1: {'lat': 10},
       2: {'lat': 5},
       3: {'lat': 4}
       }

demand = {(1, 2): {'q': 2, 'size': 100},
          (2, 4): {'q': 1, 'size': 20},
          (4, 6): {'q': 3, 'size': 50}
          }


model.D = pyo.Param(demand.keys(), initialize=demand)
model.E = Set(edges.keys(), initialize=edges)
model.dur = Param(model.TASKS, initialize=lambda model, j, m: TASKS[(j, m)]['dur'])


model.b = Var(edges,within=pyo.Binary)
model.x = Var(edges,within=pyo.NonNegativeReals)
model.tx = Var(demand)


model.obj1 = Objective()



def constraint1(m, i, j):
    return model.D[i, j]['size'] / model.x[i, j] == model.tx





from pyomo.environ import maximize, NonNegativeReals
import matplotlib.pyplot as plt


# max f1 = X1 <br>
# max f2 = 3 X1 + 4 X2 <br>
# st  X1 <= 20 <br>
#     X2 <= 40 <br>
#     5 X1 + 4 X2 <= 200 <br>

model = ConcreteModel()

model.X1 = Var(within=NonNegativeReals)
model.X2 = Var(within=NonNegativeReals)

model.C1 = Constraint(expr = model.X1 <= 20)
model.C2 = Constraint(expr = model.X2 <= 40)
model.C3 = Constraint(expr = 5 * model.X1 + 4 * model.X2 <= 200)

model.f1 = Var()
model.f2 = Var()
model.C_f1 = Constraint(expr= model.f1 == model.X1)
model.C_f2 = Constraint(expr= model.f2 == 3 * model.X1 + 4 * model.X2)
model.O_f1 = Objective(expr= model.f1  , sense=maximize)
model.O_f2 = Objective(expr= model.f2  , sense=maximize)

model.O_f2.deactivate()

solver = SolverFactory('cplex')
solver.solve(model)

print( '( X1 , X2 ) = ( ' + str(value(model.X1)) + ' , ' + str(value(model.X2)) + ' )')
print( 'f1 = ' + str(value(model.f1)) )
print( 'f2 = ' + str(value(model.f2)) )
f2_min = value(model.f2)


# ## max f2

model.O_f2.activate()
model.O_f1.deactivate()

solver = SolverFactory('cplex')
solver.solve(model)

print( '( X1 , X2 ) = ( ' + str(value(model.X1)) + ' , ' + str(value(model.X2)) + ' )')
print( 'f1 = ' + str(value(model.f1)) )
print( 'f2 = ' + str(value(model.f2)) )
f2_max = value(model.f2)


# ## apply normal $\epsilon$-Constraint

model.O_f1.activate()
model.O_f2.deactivate()

model.e = Param(initialize=0, mutable=True)

model.C_epsilon = Constraint(expr = model.f2 == model.e)

solver.solve(model);

print('Each iteration will keep f2 lower than some values between f2_min and f2_max, so ['       + str(f2_min) + ', ' + str(f2_max) + ']')

n = 4
step = int((f2_max - f2_min) / n)
steps = list(range(int(f2_min),int(f2_max),step)) + [f2_max]

x1_l = []
x2_l = []
for i in steps:
    model.e = i
    solver.solve(model)
    x1_l.append(value(model.X1))
    x2_l.append(value(model.X2))
plt.plot(x1_l,x2_l,'o-.')
plt.title('inefficient Pareto-front')
plt.grid(True)
plt.show()

# ## apply augmented $\epsilon$-Constraint

# max   f2 + delta*epsilon <br>
#  s.t. f2 - s = e

model.del_component(model.O_f1)
model.del_component(model.O_f2)
model.del_component(model.C_epsilon)

model.delta = Param(initialize=0.00001)

model.s = Var(within=NonNegativeReals)

model.O_f1 = Objective(expr = model.f1 + model.delta * model.s, sense=maximize)

model.C_e = Constraint(expr = model.f2 - model.s == model.e)

x1_l = []
x2_l = []
for i in range(160,190,6):
    model.e = i
    solver.solve(model)
    x1_l.append(value(model.X1))
    x2_l.append(value(model.X2))
plt.plot(x1_l, x2_l,'o-.')
plt.title('efficient Pareto-front')
plt.grid(True)
