import pyomo.environ as pyo

import cmath

model = pyo.AbstractModel()
model.N = pyo.Param(default=3)
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)


model.J = pyo.RangeSet(1, model.N)
model.B = pyo.RangeSet(1)

model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)

model.w0 = pyo.Param(initialize=1.0)
model.V0 = pyo.Param(initialize=1.01)
model.SGmax = pyo.Param(model.B, initialize=1.0)
model.R = pyo.Param(model.J, model.J)
model.X = pyo.Param(model.J, model.J)

model.ql = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.pl = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.pg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.qg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.y = pyo.Var(model.J, model.J)
model.t = pyo.Var(model.J, model.J)

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0)
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=0)

model.mp = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))
model.nq = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))

model.w = pyo.Var(domain=pyo.PositiveReals)

data = pyo.DataPortal()

# data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat", model=model)
# data.load(filename="/Users/my_mac/PycharmProjects/python-optimization/model.dat", model=model)
# model2 = model.create_instance(data)


def obj_expression(m):
    return sum(abs(m.v[i] - 1.0) for i in m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def admittance_rule(m, i, j):
    if i != j:
        m.y[i, j] == cmath.polar(-complex(m.R[i, j], m.X[i, j] * m.w) ** -1)[0]


def admittance_rule2(m, i, j):
    if i == j:
        total = complex(0, 0)
        for k in m.J:
            total += complex(m.R[i, k], m.X[i, k] * m.w)
    m.y[i, j] = cmath.polar(total ** -1)[0]


def admittance_rule3(m, i, j):
    if i != j:
        m.y[i, j] == cmath.polar(-complex(m.R[i, j], m.X[i, j] * m.w) ** -1)[1]


def admittance_rule4(m, i, j):
    if i == j:
        total = complex(0, 0)
        for k in m.J:
            total += complex(m.R[i, k], m.X[i, k] * m.w)
            print(total)
    m.y[i, j] = cmath.polar(total ** -1)[1]


# PQ buses
def ax_constraint_rule(m, i):
    # return the expression for the constraint for i
    return sum(m.v[i] * m.v[j] * m.y[i, j] * pyo.cos(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) - m.pl[i] == 0


def ax_constraint_rule2(m, i):
    # return the expression for the constraint for i
    return -sum(m.v[i] * m.v[j] * m.y[i, j] * pyo.sin(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) - m.ql[i] == 0


## Droop buses
def ax_constraint_rule3(m, i):
    # return the expression for the constraint for i
    return -((1 / m.mp[i]) * (m.w0 - m.w) - m.pl[i]) + sum(
        m.v[i] * m.v[j] * m.y[i, j] * pyo.cos(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i
    return -((1 / m.nq[i]) * (m.V0 - m.v[i]) - m.ql[i]) - sum(
        m.v[i] * m.v[j] * m.y[i, j] * pyo.sin(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule5(m, i):
    if i == 1:
        return m.pg[i] == (1 / m.mp[i]) * (m.w0 - m.w)


def ax_constraint_rule6(m, i):
    if i == 1:
        return m.qg[i] == (1 / m.nq[i]) * (m.V0 - m.v[i])

# Frequency dependent load and generation constraints
def ax_constraint_rule5(m, i):
    if i != 1:
        return m.pl[i] == m.p0[i]*pow(m.v[i], m.alpha) * (1+m.KPF*(m.w-m.w0))
    else:
        return m.pg[i] == (1/m.mp[i])*(m.w0-m.w)

## Voltage dependent load and generation constraints

def ax_constraint_rule6(m, i):
    if i != 1:
        return m.ql[i] == m.q0[i]*pow(m.v[i], m.beta) * (1+m.KQF*(m.w-m.w0))
    else:
        return m.qg[i] == (1/m.nq[i])*(m.V0-m.v[i])

def ax_constraint_rule7(m, i):
    return m.pg[i] <= 0.08

def ax_constraint_rule8(m, i):
    return m.qg[i] <= 0.06

def ax_constraint_rule9(m, i):
    return m.pl[i] == 0

def ax_constraint_rule10(m, i):
    return m.ql[i] == 0

def ax_constraint_rule11(m, i):
    return m.v[i] >= 0.5

def ax_constraint_rule12(m, i):
    return m.v[i] <= 1.5


model.cons1 = pyo.Constraint(model.J, rule=ax_constraint_rule)
model.cons2 = pyo.Constraint(model.J, rule=ax_constraint_rule2)

model.cons3 = pyo.Constraint(model.B, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.B, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.B, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.B, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.B, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.B, rule=ax_constraint_rule8)
model.cons9 = pyo.Constraint(model.B, rule=ax_constraint_rule9)
model.cons10 = pyo.Constraint(model.B, rule=ax_constraint_rule10)
# model.cons11 = pyo.Constraint(model.B, rule=ax_constraint_rule11)
# model.cons12 = pyo.Constraint(model.B, rule=ax_constraint_rule12)


model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
# instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
instance = model.create_instance(filename="/Users/my_mac/PycharmProjects/python-optimization/model2.dat")
opt.options['max_iter'] = 10000
opt.options['ma27_pivtol'] = 1e-5
results = opt.solve(instance, tee=True)
instance.display()

instance.pprint()

for parmobject in instance.component_objects(pyo.Param, active=True):
    nametoprint = str(str(parmobject.name))
    print("Parameter ", nametoprint)
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print("   ", index, vtoprint)

for parmobject in instance.component_objects(pyo.Var, active=True):
    nametoprint = str(str(parmobject.name))
    print("Variable ", nametoprint)
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print("   ", index, vtoprint)