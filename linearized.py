import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging

model = pyo.ConcreteModel()
model.N = pyo.Param(initialize=3)
model.BS = pyo.RangeSet(1, model.N)
model.NE = pyo.Set(model.BS, initialize={1: [2, 3], 2: [1, 3], 3: [1, 2]})


def edges_init(m):
    return [(i, n) for i in m.BS for n in m.NE[i]]


model.EDGES = pyo.Set(initialize=edges_init)
model.genSet = pyo.Set(initialize={1})
model.DP = pyo.Param(model.BS, initialize={1: 2, 2: 5, 3: 10})
model.DQ = pyo.Param(model.BS, initialize={1: 1.2, 2: 3, 3: 6})
model.R = pyo.Param(model.EDGES, initialize=0.00281)
model.X = pyo.Param(model.EDGES, initialize=0.0281)
model.VMAX = pyo.Param(initialize=200)
model.VMIN = pyo.Param(initialize=-100)
model.ALPHA = pyo.Param(initialize=math.pi / 8)
model.BETA = pyo.Param(initialize=3 * math.pi / 8)
model.GAMMA = pyo.Param(initialize=math.pi / 4)
model.IMAX = pyo.Param(initialize=300)

model.delta = pyo.Var(model.EDGES, within=pyo.Binary, initialize=0)
model.I = pyo.Var(model.EDGES, within=pyo.NonNegativeReals, initialize=0)
model.P = pyo.Var(model.EDGES, within=pyo.NonNegativeReals, initialize=0)
model.Q = pyo.Var(model.EDGES, within=pyo.NonNegativeReals, initialize=0)

model.pg = pyo.Var(model.BS, within=pyo.NonNegativeReals, initialize=0)
model.qg = pyo.Var(model.BS, within=pyo.NonNegativeReals, initialize=0)
model.v = pyo.Var(model.BS, within=pyo.NonNegativeReals, initialize=100)


def cons_1(m, i):
    return m.pg[i] + sum(m.P[j, i] for j in model.NE[i]) - m.DP[i] == sum(m.P[i, j] for j in m.NE[i])


model.cons1 = pyo.Constraint(model.BS, rule=cons_1)


def cons_2(m, i):
    return m.qg[i] + sum(m.Q[j, i] for j in model.NE[i]) - m.DQ[i] == sum(m.Q[i, j] for j in m.NE[i])


def cons_3(m, i, j):
    return m.v[j] == m.v[i] - 2 * (
            m.R[i, j] * m.P[i, j] + m.X[i, j] * m.Q[i, j]) + pow(m.R[i, j], 2) + pow(m.X[i, j], 2) * m.I[i, j]


def cons_4(m, i, j):
    return (pyo.cos(m.ALPHA) / pyo.sin(m.ALPHA) + 1 / pyo.sin(m.ALPHA)) * m.P[i, j] + m.Q[i, j] <= (
            pyo.cos(m.ALPHA) / pyo.sin(m.ALPHA) + 1 / pyo.sin(m.ALPHA)) * m.I[i, j] * m.VMAX


def cons_5(m, i, j):
    return m.Q[i, j] + (1 / pyo.cos(m.BETA) - pyo.cos(m.BETA) / pyo.sin(m.BETA)) * m.P[i, j] <= m.I[i, j] * m.VMAX


def cons_6(m, i, j):
    return pyo.cos(m.GAMMA) * m.P[i, j] + pyo.sin(m.GAMMA) * m.Q[i, j] >= m.I[i, j] * m.VMIN


def cons_7(m, i, j):
    return pyo.cos(m.GAMMA) * m.Q[i, j] + pyo.sin(m.GAMMA) * m.P[i, j] >= m.I[i, j] * m.VMIN


def cons_8(m, i, j):
    return m.I[i, j] <= m.IMAX


def cons_9(m, i, j):
    return m.delta[i, j] + m.delta[j, i] <= 1


def cons_10(m, i):
    if i not in m.genSet:
        return sum(m.delta[i, j] for j in model.NE[i]) == 1
    else:
        return pyo.Constraint.Skip


def cons_11(m, i, j):
    return m.P[i, j] + m.Q[i, j] <= 100000 * m.delta[i, j]


def cons_12(m, i):
    if i not in m.genSet:
        return m.pg[i] == 0
    else:
        return pyo.Constraint.Skip


model.cons2 = pyo.Constraint(model.BS, rule=cons_2)
model.cons3 = pyo.Constraint(model.EDGES, rule=cons_3)
model.cons4 = pyo.Constraint(model.EDGES, rule=cons_4)
model.cons5 = pyo.Constraint(model.EDGES, rule=cons_5)
model.cons6 = pyo.Constraint(model.EDGES, rule=cons_6)
model.cons7 = pyo.Constraint(model.EDGES, rule=cons_7)
model.cons8 = pyo.Constraint(model.EDGES, rule=cons_8)
model.cons9 = pyo.Constraint(model.EDGES, rule=cons_9)
model.cons10 = pyo.Constraint(model.BS, rule=cons_10)
model.cons11 = pyo.Constraint(model.EDGES, rule=cons_11)
model.cons12 = pyo.Constraint(model.BS, rule=cons_12)


def obj_expression(m):
    return sum(i * m.pg[i] for i in m.BS)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

model.name = "LinearizedPF"
opt = pyo.SolverFactory("gurobi")
results = opt.solve(model, tee=True)
log_infeasible_constraints(model, log_expression=True, log_variables=True)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

# %%

model.write("linearized.lp", io_options={'symbolic_solver_labels': True})

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
