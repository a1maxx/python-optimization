import pyomo.environ as pyo
import math
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

model.yReal = pyo.Var(model.J, model.J, initialize=1.0)
model.yIm = pyo.Var(model.J, model.J, initialize=1.0)
model.yMag = pyo.Var(model.J, model.J, initialize=1.0)
model.yThe = pyo.Var(model.J, model.J, initialize=1.0)

model.ql = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.pl = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.pg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)
model.qg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals)

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0)
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=0)

model.mp = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))
model.nq = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))

model.w = pyo.Var(domain=pyo.NonNegativeReals)


# data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat", model=model)
# data = pyo.DataPortal()
# data.load(filename="/Users/my_mac/PycharmProjects/python-optimization/model2.dat", model=model)
# model2 = model.create_instance(data)
# model2.pprint()

def obj_expression(m):
    return sum(pow((m.v[i] - 1.0), 2) for i in m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def ax_constraint_rule3(m, i):
    # return the expression for the constraint for i
    return (m.pg[i] - m.pl[i]) - sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i
    return (m.qg[i] - m.ql[i]) + sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule5(m, i):
    if i == 1:
        return m.pg[i] == (1 / m.mp[i]) * (m.w0 - m.w)
    else:
        return m.pg[i] == 0


def ax_constraint_rule6(m, i):
    if i == 1:
        return m.qg[i] == (1 / m.nq[i]) * (m.V0 - m.v[i])
    else:
        return m.qg[i] == 0


# Frequency & voltage dependent load constraints
def ax_constraint_rule7(m, i):
    return m.pl[i] == m.p0[i] * pow(m.v[i] / m.V0, m.alpha) * (1 + m.KPF * (m.w - m.w0))


def ax_constraint_rule8(m, i):
    return m.ql[i] == m.q0[i] * pow(m.v[i] / m.V0, m.beta) * (1 + m.KQF * (m.w - m.w0))


def ax_constraint_rule9(m, i):
    return m.d[i] <= math.pi/2


def ax_constraint_rule10(m, i):
    return m.d[i] >= -math.pi/2


def admittanceReal(m, i, j):
    if i != j:
        return m.yReal[i, j] == m.R[i, j] / (pow(m.R[i, j], 2) + pow((m.X[i, j] * m.w), 2))
    else:
        return pyo.Constraint.Skip


def admittanceIm(m, i, j):
    if i != j:
        return m.yIm[i, j] == -(m.X[i, j] * m.w) / (pow(m.R[i, j], 2) + pow((m.X[i, j] * m.w), 2))
    else:
        return pyo.Constraint.Skip


def admittanceDiagReal(m, i, j):
    if i == j:
        return m.yReal[i, j] == sum(m.yReal[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceDiagIm(m, i, j):
    if i == j:
        return m.yIm[i, j] == sum(m.yIm[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceMag(m, i, j):
    if i != j:
        return m.yMag[i, j] == - pyo.sqrt(pow(m.yReal[i, j], 2) + pow(m.yIm[i, j], 2))
    else:
        return pyo.Constraint.Skip


def admittanceDiagMag(m, i, j):
    if i == j:
        return m.yMag[i, j] == -sum(m.yMag[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceThe(m, i, j):
    return m.yThe[i, j] == pyo.atan(m.yIm[i, j] / m.yReal[i, j])


def maxGenCons(m, i):
    if i == 1:
        return pyo.sqrt(pow(m.pg[i], 2) + pow(m.qg[i], 2)) <= 1
    else:
        return pyo.Constraint.Skip


def maxwCons(m):
    return m.w <= 1.005


def minwCons(m):
    return m.w >= 0.995


model.cons3 = pyo.Constraint(model.B, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.B, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.J, rule=ax_constraint_rule8)
model.cons9 = pyo.Constraint(model.J, rule=ax_constraint_rule9)
model.cons10 = pyo.Constraint(model.J, rule=ax_constraint_rule10)

model.cons13 = pyo.Constraint(model.J, model.J, rule=admittanceReal)
model.cons14 = pyo.Constraint(model.J, model.J, rule=admittanceIm)
model.cons15 = pyo.Constraint(model.J, model.J, rule=admittanceDiagReal)
model.cons16 = pyo.Constraint(model.J, model.J, rule=admittanceDiagIm)
model.cons17 = pyo.Constraint(model.J, model.J, rule=admittanceMag)
model.cons18 = pyo.Constraint(model.J, model.J, rule=admittanceDiagMag)
model.cons19 = pyo.Constraint(model.J, model.J, rule=admittanceThe)
model.cons20 = pyo.Constraint(model.J, rule=maxGenCons)
model.cons21 = pyo.Constraint(rule=maxwCons)
model.cons22 = pyo.Constraint(rule=minwCons)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
# instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
instance = model.create_instance(filename="model2.dat")
instance.pprint()
# opt.options['max_iter'] = 50000
# opt.options['OF_fixed_variable_treatment'] = 'make_parameter'

# results = opt.solve(instance, tee=True)

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

(pyo.value(instance.V0) - pyo.value(instance.v[1])) * (1 / pyo.value(instance.nq[1]))

pyo.value((1 / instance.nq[1]) * (instance.V0 - instance.v[1]))
pyo.value(instance.qg[1])
