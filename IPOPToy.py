import math

import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.x1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1, 100))
model.x2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1, 100))

model.b = pyo.Param(initialize=10)


def obj_func(m):
    return m.x1 * m.x2


model.o = pyo.Objective(rule=obj_func, sense=pyo.minimize)


def constraint1(m):
    return m.x1 + m.x2 >= m.x1 * m.x2 * pyo.sqrt(m.x1/m.x2)


model.cons1 = pyo.Constraint(rule=constraint1)

opt = pyo.SolverFactory("scip")
# opt.options['max_iter'] = 50000
# opt.options['ma27_pivtol'] = 1e-10
results = opt.solve(model, tee=True)

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

x1 = pyo.value(model.x1)
x2 = pyo.value(model.x2)
obj = math.sin(x1 * x2)
obj
x1 + x2 >= x1 * x2
