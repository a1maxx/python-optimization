import math

import pyomo.environ as pyo

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


def obj_expression(m):
    return sum(abs(m.v[i] - 1.0) for i in m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def admittance_rule(m, i, j):
    if i != j:
        return m.y[i, j] == -pyo.sqrt((m.R[i, j] / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)) ** 2 +
                                      ((-m.X[i, j] * m.w) / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)) ** 2)
    else:
        return pyo.Constraint.Skip


def admittance_rule2(m, i, j):
    if i == j:
        return m.y[i, j] == sum(-m.y[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittance_rule3(m, i, j):
    if i != j:
        return m.t[i, j] == pyo.atan(((-m.X[i, j] * m.w) / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)) / (
                m.R[i, j] / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)))
    else:
        return m.t[i, j] == pyo.atan(
            sum(((-m.X[i, f] * m.w) / (m.R[i, f] ** 2 + (m.X[i, f] * m.w) ** 2)) for f in [1, 2, 3] if f != i) /
            sum((m.R[i, f] / (m.R[i, f] ** 2 + (m.X[i, f] * m.w) ** 2)) for f in [1, 2, 3] if f != i))


model.cons13 = pyo.Constraint(model.J, model.J, rule=admittance_rule)
model.cons14 = pyo.Constraint(model.J, model.J, rule=admittance_rule2)
model.cons15 = pyo.Constraint(model.J, model.J, rule=admittance_rule3)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
# instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
instance = model.create_instance(filename="/Users/my_mac/PycharmProjects/python-optimization/model2.dat")
instance.pprint()

opt.options['max_iter'] = 50000
opt.options['ma27_pivtol'] = 1e-3
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
