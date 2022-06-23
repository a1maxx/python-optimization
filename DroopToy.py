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

model.ql = pyo.Var(model.J, initialize=0, within=pyo.Reals)
model.pl = pyo.Var(model.J, initialize=0, within=pyo.Reals)
model.pg = pyo.Var(model.J, initialize=0, within=pyo.Reals)
model.qg = pyo.Var(model.J, initialize=0, within=pyo.Reals)
model.y = pyo.Var(model.J, model.J)
model.t = pyo.Var(model.J, model.J)

model.v = pyo.Var(model.J, domain=pyo.Reals, initialize=1.0)
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=0)

model.mp = pyo.Var(model.B, domain=pyo.Reals, initialize=0.5, bounds=(0, 1))
model.nq = pyo.Var(model.B, domain=pyo.Reals, initialize=0.5, bounds=(0, 1))

model.w = pyo.Var(domain=pyo.PositiveReals)


# data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat", model=model)
# data = pyo.DataPortal()
# data.load(filename="/Users/my_mac/PycharmProjects/python-optimization/model2.dat", model=model)
# model2 = model.create_instance(data)
# model2.pprint()

def obj_expression(m):
    return sum(abs(m.v[i] - 1.0) for i in m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def ax_constraint_rule3(m, i):
    # return the expression for the constraint for i
    return (m.pg[i] - m.pl[i]) - sum(
        m.v[i] * m.v[j] * m.y[i, j] * pyo.cos(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i
    return (m.qg[i] - m.ql[i]) + sum(
        m.v[i] * m.v[j] * m.y[i, j] * pyo.sin(m.t[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0


def ax_constraint_rule5(m, i):
    if i == 1:
        return m.pg[i] == (1 / m.mp[i]) * (m.w0 - m.w)
    else:
        m.pg[i] == 0


def ax_constraint_rule6(m, i):
    if i == 1:
        return m.qg[i] == (1 / m.nq[i]) * (m.V0 - m.v[i])
    else:
        m.pg[i] == 0


# Frequency & voltage dependent load constraints
def ax_constraint_rule5(m, i):
    return m.pl[i] == m.p0[i] * pow(m.v[i] / m.V0, m.alpha) * (1 + m.KPF * (m.w - m.w0))

def ax_constraint_rule6(m, i):
    return m.ql[i] == m.q0[i] * pow(m.v[i] / m.V0, m.beta) * (1 + m.KQF * (m.w - m.w0))


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

# model.cons1 = pyo.Constraint(model.J, rule=ax_constraint_rule)
# model.cons2 = pyo.Constraint(model.J, rule=ax_constraint_rule2)

model.cons3 = pyo.Constraint(model.B, rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.B, rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.J, rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.J, rule=ax_constraint_rule6)


model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
# instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
instance = model.create_instance(filename="model2.dat")
# opt.options['max_iter'] = 50000
# opt.options['ma27_pivtol'] = 1e-2
results = opt.solve(instance, tee=True)

#%%
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
