import pandas as pd
import pyomo.environ as pyo
# noinspection PyUnresolvedReferences
from pyomo.environ import NonNegativeReals
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from importlib import reload

model = pyo.AbstractModel()
model.A = pyo.Set()
model.T = pyo.Set()
model.T0 = pyo.Set()
model.Gen = pyo.Set()
model.line = pyo.Set()
model.PD = pyo.Param(model.A, model.T, domain=NonNegativeReals)
model.QD = pyo.Param(model.A, model.T, domain=NonNegativeReals)

filename = "/Users/my_mac/PycharmProjects/python-optimization/secData2.xlsx"
data = pyo.DataPortal(model=model)
data.load(filename=filename, range="Atable", format='set', set='A')
data.load(filename=filename, range="Ttable", format='set', set='T')
data.load(filename=filename, range="T0Table", format='set', set='T0')
data.load(filename=filename, range="genset", format='set', set='Gen')
data.load(filename=filename, range="Pdemand", param='PD', format='array')
data.load(filename=filename, range="Qdemand", param='QD', format='array')
data.load(filename=filename, range="lineset", format='set', set='line')

df = pd.read_excel(filename, sheet_name="Sheet2")
d = dict()

for i in df.index:
    d[df.loc[i].iat[0], df.loc[i].iat[1]] = df.loc[i].iat[2]

for i in range(1, 7):
    target = 0
    for key, v in d.items():
        if i in key:
            target += -v
    d[i, i] = target

model.G = pyo.Param(model.line, initialize=d, default=0)

d2 = dict()
for i in df.index:
    d2[df.loc[i].iat[0], df.loc[i].iat[1]] = df.loc[i].iat[4]

for i in range(1, 7):
    target = 0
    for key, v in d2.items():
        if i in key:
            target += -v
    d2[i, i] = target

model.B = pyo.Param(model.line, initialize=d2, default=0)

model.VN = pyo.Param(initialize=1.0)

# Predicted active power output of RES at time t
model.PR = pyo.Param(model.A, model.T)
model.QR = pyo.Param(model.A, model.T)
data.load(filename=filename, range="PRTable", param='PR', format='array')
data.load(filename=filename, range="QRTable", param='QR', format='array')

model.SG = pyo.Param(model.Gen)
data.load(filename=filename, range='capgen', param="SG")

# model.SOC0 = pyo.Param(model.A)
# data.load(filename=filename, range='soc0', param='SOC0')

instance = model.create_instance(data)
# instance.pprint()

model.EPS = pyo.Param(initialize=0.05)

# Charging efficiency of energy storage
# model.PIC = pyo.Param(default=1)
# Discharging efficiency of energy storage
# model.PID = pyo.Param(default=1)

# model.EC = pyo.Param(model.A, default=10)
# data.load(filename=filename, range='ecset', param="EC")
model.PRI = pyo.Param(model.A)
data.load(filename=filename, range='priset', param="PRI")

# Decision variables
# Voltage magnitude of the node at time t
model.v = pyo.Var(model.A, model.T, within=pyo.NonNegativeReals)
# Phase difference between Vit and Vjt it can also be model.line instead of two model.A
model.theta = pyo.Var(model.line, model.T)
# Active/Reactive power flow at time t
model.p = pyo.Var(model.line, model.T)
model.q = pyo.Var(model.line, model.T)
# Active/Reactive power generation
model.pg = pyo.Var(model.A, model.T, within=pyo.NonNegativeReals)
model.qg = pyo.Var(model.A, model.T, within=pyo.NonNegativeReals)

# Active power output of the energy storage at time t
# model.pe = pyo.Var(model.A, model.T)
# Charging state of ES
# model.lamb = pyo.Var(model.A, model.T, within=pyo.Binary)
# Discharging state of ES
# model.phi = pyo.Var(model.A, model.T, within=pyo.Binary)
# Power deficiency of the distribution system operator
# model.soc = pyo.Var(model.A, model.T0)
# model.n1 = pyo.Var()

# Indicator of a boundary line of an MG
model.x = pyo.Var(model.line, within=pyo.Binary)
# Connection status of the load at time t
model.y = pyo.Var(model.A, model.T, within=pyo.Binary)


def obj_expression(m):
    return sum(sum(abs(m.v[k, t] - m.VN) + sum(m.x[k, j] for j in m.A if (k, j) in m.line) + m.PRI[k] * m.PD[k, t] * (
            1 - m.y[k, t])
                   for k in m.A) for t in m.T)


model.obj1 = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def constraint_rule32(m, k, t):
    return 1 - m.EPS <= m.v[k, t]


model.cons32 = pyo.Constraint(model.A, model.T, rule=constraint_rule32)


def constraint_rule32_2(m, k, t):
    return m.v[k, t] <= 1 + m.EPS


model.cons32_2 = pyo.Constraint(model.A, model.T, rule=constraint_rule32_2)


def constraint_rule33(m, k, t):
    return m.pg[k, t] - m.y[k, t] * m.PD[k, t] + m.PR[k, t] == sum(m.p[k, j, t] for j in m.A if (k, j) in m.line)


model.cons33 = pyo.Constraint(model.A, model.T, rule=constraint_rule33)


def constraint_rule34(m, k, t):
    return m.qg[k, t] - m.y[k, t] * m.QD[k, t] + m.QR[k, t] == sum(m.q[k, j, t] for j in m.A if (k, j) in m.line)


model.cons34 = pyo.Constraint(model.A, model.T, rule=constraint_rule34)


def constraint_rule35(m, k, j, t):
    return m.p[k, j, t] == m.x[k, j] * m.v[k, t] * m.v[j, t] * (
            m.G[k, j] * pyo.cos(m.theta[k, j, t]) + m.B[k, j] * pyo.sin(m.theta[k, j, t]))


model.cons35 = pyo.Constraint(model.line, model.T, rule=constraint_rule35)


def constraint_rule36(m, k, j, t):
    return m.q[k, j, t] == m.x[k, j] * m.v[k, t] * m.v[j, t] * (
            m.G[k, j] * pyo.sin(m.theta[k, j, t]) - m.B[k, j] * pyo.cos(m.theta[k, j, t]))


model.cons36 = pyo.Constraint(model.line, model.T, rule=constraint_rule36)


def constraint_rule37(m, k, t):
    return m.pg[k, t] ** 2 + m.qg[k, t] ** 2 <= m.SG[k]**2


model.cons37 = pyo.Constraint(model.Gen, model.T, rule=constraint_rule37)


def constraint_rule42(m, t):
    return sum(m.pg[k, t] + m.PR[k, t] for k in m.A) >= sum(m.PD[k, t] for k in m.A)


model.cons42 = pyo.Constraint(model.T, rule=constraint_rule42)


def constraint_rule42_2(m, k, t):
    if k not in m.Gen:
        return m.pg[k, t] == 0
    else:
        return pyo.Constraint.Skip


model.cons42_2 = pyo.Constraint(model.A, model.T, rule=constraint_rule42_2)


def constraint_rule42_3(m, k, t):
    if k not in m.Gen:
        return m.qg[k, t] == 0
    else:
        return pyo.Constraint.Skip


model.cons42_3 = pyo.Constraint(model.A, model.T, rule=constraint_rule42_3)


def constraint_rule43(m, t):
    return sum(m.qg[k, t] + m.QR[k, t] for k in m.A) >= sum(m.QD[k, t] for k in m.A)


model.cons43 = pyo.Constraint(model.T, rule=constraint_rule43)


instance = model.create_instance(data)

opt = pyo.SolverFactory("ipopt")
instance.name = "DroopControlledIMG"
opt.options['max_iter'] = 100000
opt.options['acceptable_tol'] = 1e-3
opt.options['tol'] = 1e-2
opt.options['ma27_pivtol'] = 1e-3
results = opt.solve(instance, tee=True)
