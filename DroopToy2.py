import pyomo.environ as pyo
import math

model = pyo.AbstractModel()
model.N = pyo.Param()
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)
model.genSet = pyo.Set()

model.J = pyo.RangeSet(1, model.N)
model.B = pyo.RangeSet(1)

model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)

model.w0 = pyo.Param(initialize=1.0)
model.V0 = pyo.Param(initialize=1.01)
model.SGmax = pyo.Param(model.B, initialize=1.0)
model.R = pyo.Param(model.J, model.J)
model.X = pyo.Param(model.J, model.J)

model.yReal = pyo.Var(model.J, model.J, initialize=1.0, bounds=(-100, 100))
model.yIm = pyo.Var(model.J, model.J, initialize=1.0, bounds=(-100, 100))
model.yMag = pyo.Var(model.J, model.J, initialize=1.0, bounds=(-100, 100))
model.yThe = pyo.Var(model.J, model.J, initialize=1.0, bounds=(-100, 100))

model.ql = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pl = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.qg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0, 2))
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=0, bounds=(-math.pi / 2, math.pi / 2))

model.mp = pyo.Var(model.genSet, domain=pyo.NonNegativeReals, initialize=0.1, bounds=(0, 1))
model.nq = pyo.Var(model.genSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(0, 1))

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
    # # return the expression for the constraint for i
    # return (m.pg[i] - m.pl[i]) - sum(
    #     m.v[i] * m.v[j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0

    # Approximated version - cosine
    return (m.pg[i] - m.pl[i]) - \
           sum(m.v[i] * m.v[j] * m.yMag[i, j] * (1 - pow((m.yThe[i, j] + m.d[j] - m.d[i]), 2) / 2) for j in m.J) == 0


def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i

    # return (m.qg[i] - m.ql[i]) + sum(
    #     m.v[i] * m.v[j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0

    # Approximated version - sine
    return (m.qg[i] - m.ql[i]) + sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * (16 * (m.yThe[i, j] + m.d[j] - m.d[i]) *
                                          (math.pi - (m.yThe[i, j] + m.d[j] - m.d[i]))) / (
                5 * pow(math.pi, 2) - 4 * (m.yThe[i, j] + m.d[j] - m.d[i]) *
                (math.pi - (m.yThe[i, j] + m.d[j] - m.d[i])))
        for j in m.J) == 0


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


def admittanceReal(m, i, j):
    if i != j:
        return m.yReal[i, j] == -(m.R[i, j] / (pow(m.R[i, j], 2) + pow((m.X[i, j] * m.w), 2)))
    else:
        return pyo.Constraint.Skip


def admittanceIm(m, i, j):
    if i != j:
        return m.yIm[i, j] == (m.X[i, j] * m.w) / (pow(m.R[i, j], 2) + pow((m.X[i, j] * m.w), 2))
    else:
        return pyo.Constraint.Skip


def admittanceDiagReal(m, i, j):
    if i == j:
        return m.yReal[i, j] == -sum(m.yReal[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceDiagIm(m, i, j):
    if i == j:
        return m.yIm[i, j] == -sum(m.yIm[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceMag(m, i, j):
    if i != j:
        return m.yMag[i, j] == -pyo.sqrt(pow(m.yReal[i, j], 2) + pow(m.yIm[i, j], 2))
    else:
        return pyo.Constraint.Skip


def admittanceDiagMag(m, i, j):
    if i == j:
        return m.yMag[i, j] == -sum(m.yMag[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceThe(m, i, j):
    # return m.yThe[i, j] == pyo.atan(m.yIm[i, j] / m.yReal[i, j])

    # Approximated version - arctan
    # return m.yThe[i, j] == (m.yIm[i, j] / m.yReal[i, j]) / (1 + 0.28125 * pow((m.yIm[i, j] / m.yReal[i, j]), 2))

    # Approximated version 2 - arctan
    return m.yThe[i, j] == math.pi / 4 * (m.yIm[i, j] / m.yReal[i, j]) - (m.yIm[i, j] / m.yReal[i, j]) * (
            abs(m.yIm[i, j] / m.yReal[i, j]) - 1) * (0.2447 + 0.0663 * abs(m.yIm[i, j] / m.yReal[i, j]))


def maxGenCons(m, i):
    if i == 1:
        return pyo.sqrt(pow(m.pg[i], 2) + pow(m.qg[i], 2)) <= 1
    else:
        return pyo.Constraint.Skip


def maxwCons(m):
    return m.w <= 1.005


def minwCons(m):
    return m.w >= 0.995


# def flow_cons3(m):
#     return m.v[1] * m.v[1] * m.yMag[1, 2] * (16 * (m.yThe[1, 2] + m.d[2] - m.d[1]) *
#                                              (math.pi - (m.yThe[1, 2] + m.d[2] - m.d[1]))) / (
#                    5 * pow(math.pi, 2) - 4 * (m.yThe[1, 2] + m.d[2] - m.d[1]) *
#                    (math.pi - (m.yThe[1, 2] + m.d[2] - m.d[1]))) <= 1e-6
#
#
# def flow_cons4(m):
#     return m.v[1] * m.v[2] * m.yMag[1, 2] * (1 - pow((m.yThe[1, 2] + m.d[2] - m.d[1]), 2) / 2) <= 1e-6

# def flow_cons1(m):
#     return abs(m.d[1] - m.d[2]) == 0
#
# def flow_cons2(m):
#     return abs(m.v[1] - m.v[2]) == 0

# model.flowCons = pyo.Constraint(rule=flow_cons1)
# model.flowCons2 = pyo.Constraint(rule=flow_cons2)
# model.flowCons3 = pyo.Constraint(rule=flow_cons3)
# model.flowCons4 = pyo.Constraint(rule=flow_cons4)


model.cons3 = pyo.Constraint(model.J, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.J, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.J, rule=ax_constraint_rule6)
model.cons7 = pyo.Constraint(model.J, rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.J, rule=ax_constraint_rule8)
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
# opt.options['acceptable_tol'] = 1e-3
instance = model.create_instance(filename="model2.dat")
# instance.pprint()
# opt.options['max_iter'] = 50000

results = opt.solve(instance, tee=True)

# %%

instance.display()
instance.pprint()
instance
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
