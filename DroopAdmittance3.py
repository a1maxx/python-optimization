import math

import pyomo.environ as pyo
from pyomo.environ import value

model = pyo.AbstractModel()
model.N = pyo.Param(default=3)
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)
model.B = pyo.RangeSet(1)
model.J = pyo.RangeSet(1, model.N)
model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)
model.R = pyo.Param(model.J, model.J)
model.X = pyo.Param(model.J, model.J)

model.yReal = pyo.Var(model.J, model.J, initialize=1.0)
model.yIm = pyo.Var(model.J, model.J, initialize=1.0)
model.yMag = pyo.Var(model.J, model.J, initialize=1.0)
model.yThe = pyo.Var(model.J, model.J, initialize=1.0)

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0)
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=0)

model.mp = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))
model.nq = pyo.Var(model.B, domain=pyo.NonNegativeReals, initialize=0.5, bounds=(0, 1))

model.w = pyo.Param(initialize =1, domain=pyo.PositiveReals)


# def obj_expression(m):
#     return sum(abs(m.v[i] - 1.0) for i in m.J)


def obj_expression(m):
    return 1

def admittanceReal(m, i, j):
    if i != j:
        return m.yReal[i, j] == -m.R[i, j] / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)
    else:
        return pyo.Constraint.Skip


def admittanceIm(m, i, j):
    if i != j:
        return m.yIm[i, j] == (m.X[i, j] * m.w) / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)
    else:
        return pyo.Constraint.Skip


def admittanceDiagReal(m, i, j):
    if i == j:
        return m.yReal[i, j] == sum(-m.yReal[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceDiagIm(m, i, j):
    if i == j:
        return m.yIm[i, j] == sum(-m.yIm[i, f] for f in [1, 2, 3] if f != i)
    else:
        return pyo.Constraint.Skip


def admittanceMag(m, i, j):
        return m.yMag[i, j] == pyo.sqrt(m.yReal[i, j] ** 2 + m.yIm[i, j] ** 2)


def admittanceThe(m, i, j):
    if i == j:
        return m.yThe[i, j] == pyo.atan(m.yIm[i, j] / m.yReal[i, j])
    else:
        return m.yThe[i, j] == pyo.atan(m.yIm[i, j] / m.yReal[i, j]) + math.pi


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)
model.cons13 = pyo.Constraint(model.J, model.J, rule=admittanceReal)
model.cons14 = pyo.Constraint(model.J, model.J, rule=admittanceIm)
model.cons15 = pyo.Constraint(model.J, model.J, rule=admittanceDiagReal)
model.cons16 = pyo.Constraint(model.J, model.J, rule=admittanceDiagIm)
model.cons17 = pyo.Constraint(model.J, model.J, rule=admittanceMag)
model.cons19 = pyo.Constraint(model.J, model.J, rule=admittanceThe)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
# instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
instance = model.create_instance(filename="model2.dat")
instance.pprint()

# opt.options['max_iter'] = 50000
# opt.options['ma27_pivtol'] = 1e-1
results = opt.solve(instance, tee=True)

#
# value(instance.yReal[(1,2)])
# value(instance.yIm[1,2])
# a1 = pyo.atan(value(instance.yIm[(1,2)]) / value(instance.yReal[1,2])) + math.pi
#
# value(instance.yReal[(1,1)])
# value(instance.yIm[1,1])
# a2 = pyo.atan(value(instance.yIm[(1,1)]) /value(instance.yReal[1,1]))
# a2

(pyo.atan(15/-5) +math.pi) * 180/math.pi
(pyo.atan(-55/15) ) * 180/math.pi
# %%
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
