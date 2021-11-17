import pyomo.environ as pyo
# noinspection PyUnresolvedReferences
from pyomo.environ import NonNegativeReals
import pandas as pd

model = pyo.AbstractModel()
model.A = pyo.Set()
model.T = pyo.Set()
model.T0 = pyo.Set()
model.Gen = pyo.Set()
model.line = pyo.Set()
model.PD = pyo.Param(model.A, model.T)
model.QD = pyo.Param(model.A, model.T)

filename = "C:\\Users\\Administrator\\Desktop\\Datas\\secData.xlsx"
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

model.G = pyo.Param(model.line, initialize=d, default=0)
d.clear()

for i in df.index:
    d[df.loc[i].iat[0], df.loc[i].iat[1]] = df.loc[i].iat[4]

model.B = pyo.Param(model.line, initialize=d, default=0)

model.VN = pyo.Param(initialize=1)

# Predicted active power output of RES at time t
model.PR = pyo.Param(model.A, model.T)
model.QR = pyo.Param(model.A, model.T)
data.load(filename=filename, range="PRTable", param='PR', format='array')
data.load(filename=filename, range="QRTable", param='QR', format='array')

model.SG = pyo.Param(model.Gen)
data.load(filename=filename, range='capgen', param="SG")

model.SOC0 = pyo.Param(model.A)
data.load(filename=filename, range='soc0', param='SOC0')

instance = model.create_instance(data)
instance.pprint()

model.EPS = pyo.Param(initialize=0.05)

model.CG = pyo.Param(default=10)
# Redispatch cost of dispatchable unit $/kW
model.CRG = pyo.Param(default=10)
# Emission cost of dispatchable unit $/kW
model.CEM = pyo.Param(default=10)
# Emission factor of dispatchable unit kg/kWh
model.CS = pyo.Param(default=10)
# Price of selling electricity to consumers $/kW
model.CD = pyo.Param(default=10)
# Price of selling electricity to upstream grid
model.CS = pyo.Param(default=10)
# Price of buying electricity from upstream grid
model.CS = pyo.Param(default=10)
# Charging efficiency of energy storage
model.PIC = pyo.Param(default=1)
# Discharging efficiency of energy storage
model.PID = pyo.Param(default=1)

model.EC = pyo.Param(model.A, default=10)
data.load(filename=filename, range='ecset', param="EC")
model.PRI = pyo.Param(model.A)
data.load(filename=filename, range='priset', param="PRI")

# Decision variables
# Voltage magnitude of the node at time t
model.v = pyo.Var(model.A, model.T, within=NonNegativeReals)
# Phase difference between Vit and Vjt it can also be model.line instead of two model.A
model.theta = pyo.Var(model.A, model.A, model.T)
# Active/Reactive power flow at time t
model.p = pyo.Var(model.A, model.A, model.T)
model.q = pyo.Var(model.A, model.A, model.T)
# Active/Reactive power generation
model.pg = pyo.Var(model.A, model.T)
model.qg = pyo.Var(model.A, model.T)
# Active power output of the energy storage at time t
model.pe = pyo.Var(model.A, model.T)
# Charging state of ES
model.lamb = pyo.Var(model.A, model.T)
# Discharging state of ES
model.phi = pyo.Var(model.A, model.T)
# Power deficiency of the distribution system operator
model.soc = pyo.Var(model.A, model.T0)
model.n1 = pyo.Var()
# Power surplus of the distribution system operator
model.theta1 = pyo.Var()
# Redispatch cost f the MT at time t
model.crd = pyo.Var(model.A, model.T)
# Indicator of a boundary line of an MG
model.x = pyo.Var(model.A, model.A, within=pyo.Binary)
# Connection status of the load at time t
model.y = pyo.Var(model.A, model.T, within=pyo.Binary)


def obj_expression(m):
    return sum(sum(m.v[k, t] - m.VN + sum(m.x[k, j] for j in m.A) + m.PRI[k] * m.PD[k, t] * (1 - m.y[k, t])
                   for k in m.A) for t in m.T)


model.obj1 = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def constraint_rule32(m, k, t):
    return 1 - m.EPS <= m.v[k, t]


def constraint_rule32_2(m, k, t):
    return m.v[k, t] <= 1 + m.EPS


model.cons32 = pyo.Constraint(model.A, model.T, rule=constraint_rule32)
model.cons32_2 = pyo.Constraint(model.A, model.T, rule=constraint_rule32_2)


def constraint_rule33(m, k, t):
    return m.pg[k, t] - m.y[k, t] * m.PD[k, t] + m.PR[k, t] == sum(m.p[k, j, t] for j in m.A)


model.cons33 = pyo.Constraint(model.A, model.T, rule=constraint_rule33)


def constraint_rule34(m, k, t):
    return m.qg[k, t] - m.y[k, t] * m.QD[k, t] + m.QR[k, t] == sum(m.p[k, j, t] for j in m.A)


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
    return m.pg[k, t] ** 2 + m.qg[k, t] ** 2 <= m.SG[k]


model.cons37 = pyo.Constraint(model.Gen, model.T, rule=constraint_rule37)


def constraint_rule38(m, k, t):
    return m.pe[k, t] <= 200 * m.phi[k, t]


def constraint_rule38_2(m, k, t):
    return -m.pe[k, t] * m.lamb[k, t] <= m.pe[k, t]


model.cons38 = pyo.Constraint(model.A, model.T, rule=constraint_rule38)
model.cons38_2 = pyo.Constraint(model.A, model.T, rule=constraint_rule38_2)


def constraint_rule39(m, k, t):
    return m.lamb[k, t] + m.phi[k, t] <= 1


model.cons39 = pyo.Constraint(model.A, model.T, rule=constraint_rule39)


def constraint_rule40_2(m, k):
    return m.soc[k, 0] == m.SOC0[k]


model.cons40_2 = pyo.Constraint(model.A, rule=constraint_rule40_2)


# Time interval assumed to be 1
def constraint_rule40(m, k, t):
    if m.EC[k] == 0:
        return pyo.Constraint.Skip
    else:
        return m.soc[k, t] == m.soc[k, t - 1] - 1 / m.EC[k] * (
            m.phi[k, t] * m.pe[k, t] * m.PID ** -1 + m.lamb[k, t] * m.pe[k, t] * m.PIC)


model.cons40 = pyo.Constraint(model.A, model.T, rule=constraint_rule40)


# Maximum possible SOC assumed to be 100
def constraint_rule41(m, k, t):
    return 0 <= m.soc[k, t]


def constraint_rule41_2(m, k, t):
    return m.soc[k, t] <= 100


model.cons41 = pyo.Constraint(model.A, model.T, rule=constraint_rule41)

model.cons41_2 = pyo.Constraint(model.A, model.T, rule=constraint_rule41_2)


def constraint_rule42(m, t):
    return sum(m.pg[k, t] + m.pe[k, t] + m.PR[k, t] for k in m.A) >= sum(m.PD[k, t] for k in m.A)


model.cons42 = pyo.Constraint(model.T, rule=constraint_rule42)


def constraint_rule43(m, t):
    return sum(m.qg[k, t] + m.QR[k, t] for k in m.A) >= sum(m.QD[k, t] for k in m.A)


model.cons43 = pyo.Constraint(model.T, rule=constraint_rule43)

instance = model.create_instance(data)
instance.pprint()

# %%
# Sets
# Set of nodes in unfaulted area
model.I = pyo.RangeSet(1, 3)
model.N = 5
# Set of nodes in on-outage area
model.K = pyo.Set(1, model.N)
# Set of all nodes
model.A = pyo.RangeSet(1, model.N)
# Set of periods
model.T = pyo.RangeSet(1, 12)

# Parameters
# Line conductance between bus i and j
model.G = pyo.Param(model.A, model.A, initialize=0)
# Line susceptance between bus i an j
model.B = pyo.Param(model.A, model.A)
# Active/Reactive demand at node i
model.PD = pyo.Param(model.A, model.T)
model.QD = pyo.Param(model.A, model.T)
# Nominal Voltage
model.VN = pyo.Param(initialize=1)
# Power base for the system
model.Sbase = pyo.Param(default=10)
# Predicted active power output of RES at time t
model.PR = pyo.Param(model.A, model.T)
model.QR = pyo.Param(model.A, model.T)
# Capacity of the inverter
model.SG = pyo.Param(model.A)
# Maximum allowed voltage deviation
model.EPS = pyo.Param(default=10)
# Generation cost of dispatchable unit $/kW
model.CG = pyo.Param(default=10)
# Redispatch cost of dispatchable unit $/kW
model.CRG = pyo.Param(default=10)
# Emission cost of dispatchable unit $/kW
model.CEM = pyo.Param(default=10)
# Emission factor of dispatchable unit kg/kWh
model.CS = pyo.Param(default=10)
# Price of selling electricity to consumers $/kW
model.CD = pyo.Param(default=10)
# Price of selling electricity to upstream grid
model.CS = pyo.Param(default=10)
# Price of buying electricity from upstream grid
model.CS = pyo.Param(default=10)
# Charging efficiency of energy storage
model.PIC = pyo.Param(default=10)
# Discharging efficiency of energy storage
model.PID = pyo.Param(default=10)
# Capacity of energy storage
model.EC = pyo.Param(model.K, default=10)
# Priority index of the load
model.PRI = pyo.Param(model.A)
