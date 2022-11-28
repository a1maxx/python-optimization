import pyomo.environ as pyo
import math
from pyomo.util.infeasible import log_infeasible_constraints
import logging





model = pyo.AbstractModel()
model.N = pyo.Param()
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)

model.J = pyo.RangeSet(1, model.N)
model.B = pyo.RangeSet(1)
model.genSet = pyo.Set()

model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)

model.w0 = pyo.Param(initialize=1.0)
model.V0 = pyo.Param(initialize=1.01)
model.SGmax = pyo.Param(model.genSet, initialize=1.0)
model.R = pyo.Param(model.J, model.J)
model.X = pyo.Param(model.J, model.J)


def yReal_init(m, i, j):
    if i != j:
        return 0.1
    else:
        return 0.2


model.yReal = pyo.Var(model.J, model.J, initialize=yReal_init, bounds=(-100, 100))
model.yIm = pyo.Var(model.J, model.J, initialize=0.1, bounds=(-100, 100))

model.yMag = pyo.Var(model.J, model.J, initialize=1)
model.yThe = pyo.Var(model.J, model.J, initialize=1)

model.ql = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pl = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.pg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
model.qg = pyo.Var(model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

model.v = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.95, 1.05))
model.d = pyo.Var(model.J, domain=pyo.Reals, initialize=1.0)

model.mp = pyo.Var(model.genSet, domain=pyo.Reals, initialize=0.03,bounds=(-2,2))
model.nq = pyo.Var(model.genSet, domain=pyo.Reals, initialize=0.01, bounds=(-2,2))

model.w = pyo.Var(domain=pyo.NonNegativeReals, initialize=1,bounds=(0.995,1.005))

# data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat", model=model)
# data = pyo.DataPortal()
# data.load(filename="/Users/my_mac/PycharmProjects/python-optimization/model2.dat", model=model)
# model2 = model.create_instance(data)
# model2.pprint()

# def obj_expression(m):
#     return sum(pow((m.v[i] - 1.0), 2) for i in m.J)


def obj_expression(m):
    return 1/2  * sum((m.v[i]*m.v[j]*m.yMag[i,j]) * pyo.cos(m.d[i]+m.d[j]+m.yThe[i,j]) for i,j in m.J*m.J) + \
           (-1/2) * sum((m.v[i]*m.v[j]*m.yMag[i,j]) * pyo.sin(m.d[i]+m.d[j]+m.yThe[i,j]) for i,j in m.J*m.J)


model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)


def ax_constraint_rule3(m, i):
    # # return the expression for the constraint for i
    return (m.pg[i] - m.pl[i]) - sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0

    # Approximated version - cosine
    # return (m.pg[i] - m.pl[i]) - \
    #        sum(m.v[i] * m.v[j] * m.yMag[i, j] * (1 - pow((m.yThe[i, j] + m.d[j] - m.d[i]), 2) / 2) for j in
    #            m.J if (m.R[i, j] != 0 or i == j)) == 0


def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i

    return (m.qg[i] - m.ql[i]) + sum(
        m.v[i] * m.v[j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[j] - m.d[i]) for j in m.J) == 0

    # Approximated version - sine
    # return (m.qg[i] - m.ql[i]) + sum(
    #     m.v[i] * m.v[j] * m.yMag[i, j] * (16 * (m.yThe[i, j] + m.d[j] - m.d[i]) *
    #                                       (math.pi - (m.yThe[i, j] + m.d[j] - m.d[i]))) / (
    #             5 * pow(math.pi, 2) - 4 * (m.yThe[i, j] + m.d[j] - m.d[i]) *
    #             (math.pi - (m.yThe[i, j] + m.d[j] - m.d[i])))
    #     for j in m.J if (m.R[i, j] != 0 or i == j)) == 0


def ax_constraint_rule5(m, i):
    if i in m.genSet:
        return m.pg[i] == (1 / m.mp[i]) * (m.w0 - m.w)
    else:
        return m.pg[i] == 0


def ax_constraint_rule6(m, i):
    if i in m.genSet:
        return m.qg[i] == (1 / m.nq[i]) * (m.V0 - m.v[i])
    else:
        return m.qg[i] == 0


# Frequency & voltage dependent load constraints
def ax_constraint_rule7(m, i):
    return m.pl[i] == m.p0[i] * pow(m.v[i] / m.V0, m.alpha) * (1 + m.KPF * (m.w - m.w0))


def ax_constraint_rule8(m, i):
    return m.ql[i] == m.q0[i] * pow(m.v[i] / m.V0, m.beta) * (1 + m.KQF * (m.w - m.w0))


def admittanceReal(m, i, j):
    if i != j and m.R[i,j] != 0:
        return m.yReal[i, j] == -m.R[i, j] / (m.R[i, j] ** 2 + (m.X[i, j] * m.w) ** 2)
    else:
        return pyo.Constraint.Skip


def admittanceIm(m, i, j):
    if i != j and m.R[i,j] != 0:
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

def maxGenCons(m, i):
    if i in m.genSet:
        return pyo.sqrt(pow(m.pg[i], 2) + pow(m.qg[i], 2)) <= 1
    else:
        return pyo.Constraint.Skip

# def maxwCons(m):
#     return m.w <= 1.005
#
#
# def minwCons(m):
#     return m.w >= 0.997


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
# model.cons18 = pyo.Constraint(model.J, model.J, rule=admittanceDiagMag)
model.cons19 = pyo.Constraint(model.J, model.J, rule=admittanceThe)
model.cons20 = pyo.Constraint(model.J, rule=maxGenCons)
# model.cons21 = pyo.Constraint(rule=maxwCons)
# model.cons22 = pyo.Constraint(rule=minwCons)

model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
opt.options['acceptable_tol'] = 1e-2
opt.options['max_iter'] = 100000000
instance = model.create_instance(filename="model3.dat")
# instance.pprint()

log_infeasible_constraints(instance, log_expression=True, log_variables=True)
logging.basicConfig(filename='example2.log', level=logging.INFO)

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



# %%
import cmath,math
import numpy as np
a = complex(2,1)
b = complex(3,4)
c = complex(4,1)
d = complex(5,6)
ap = cmath.polar(a)
bp = cmath.polar(b)
cp = cmath.polar(c)
dp = cmath.polar(d)




def mmult2R(ap:tuple,bp:tuple):
    return  round((ap[0] * bp[0]) * math.cos(ap[1]+bp[1]),4)



def mmult3R(ap:tuple,bp:tuple,cp:tuple):
    return (ap[0] * bp[0] * cp[0]) * math.cos(ap[1]+bp[1]+cp[1])


def mmult3Rv2(ap:tuple,bp:tuple,cp:tuple):
    return 0.5 * (((ap[0]*bp[0]*cp[0]) * math.cos(-ap[1]+bp[1]+cp[1])) + ((ap[0]*bp[0]*cp[0]) * math.cos(ap[1]-bp[1]+cp[1])))

def mmult3Xv2(ap:tuple,bp:tuple,cp:tuple):
    return 0.5 * (((ap[0]*bp[0]*cp[0]) * math.sin(-ap[1]+bp[1]+cp[1])) + ((ap[0]*bp[0]*cp[0]) * math.sin(ap[1]-bp[1]+cp[1])))

def mmult3I(ap:tuple,bp:tuple,cp:tuple):
    return round((ap[0] * bp[0] * cp[0]) * math.sin(ap[1]+bp[1]+cp[1]),4)

def vanObj(a,b,c):
    return 0.5 * c * ((np.conj(a)*b) + (np.conj(b)*a))


