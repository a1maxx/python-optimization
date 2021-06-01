# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:06:04 2021

@author: Administrator
"""

# pyomo --solver=ipopt "C:\\Users\\Administrator\\Py Files\\abstractModel.py" "C:\\Users\\Administrator\\Py Files\\model.dat"

import pyomo.environ as pyo
# from pyomo.opt import SolverFactory


model = pyo.AbstractModel()
model.N = pyo.Param(default=6)
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)

model.I = pyo.RangeSet(2,model.N)
model.J = pyo.RangeSet(1,model.N)
model.B = pyo.RangeSet(4,model.N)

model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)
model.y = pyo.Param(model.J,model.J)
model.t = pyo.Param(model.J,model.J)
model.w0 = pyo.Param(initialize= 1.0)
model.V0 = pyo.Param(initialize= 1.01)

model.q = pyo.Var(model.J,initialize=0.2)
model.p = pyo.Var(model.J,initialize=0) 
model.v = pyo.Var(model.J,domain=pyo.NonNegativeReals,initialize=1.0)
model.d = pyo.Var(model.J,domain=pyo.Reals,initialize=0)
model.mp = pyo.Var(model.B,domain=pyo.Reals,initialize=0.5,bounds=(0,1))
model.nq = pyo.Var(model.B,domain=pyo.Reals,initialize=0.5,bounds=(0,1))
model.w = pyo.Var(domain=pyo.PositiveReals)

data = pyo.DataPortal()
data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat",model=model)
model2 = model.create_instance(data)


def obj_expression(m):
    return sum(abs(m.v[i]-1.0) for i in m.v)

model.o = pyo.Objective(rule=obj_expression,sense=1)

def ax_constraint_rule(m, i):
    # return the expression for the constraint for i
    return sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.cos(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) + m.p[i]   == 0

def ax_constraint_rule2(m, i):
    # return the expression for the constraint for i
    return   -sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.sin(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) + m.q[i] == 0

def ax_constraint_rule3(m, i):
    # return the expression for the constraint for i
    return -(1/m.mp[i])*(m.w0-m.w) + m.p[i] + sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.cos(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) == 0

def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i
    return -(1/m.nq[i])*(m.V0-m.v[i]) + m.q[i] - sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.sin(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) == 0

def ax_constraint_rule5(m, i):
    # return the expression for the constraint for i
    if(i<=3):
        return m.p[i] == m.p0[i]*pow(m.v[i],m.alpha) *(1+m.KPF*(m.w-m.w0))
    else:
        return m.p[i] == (1/m.mp[i])*(m.w0-m.w)
    
    
def ax_constraint_rule6(m, i):
    # return the expression for the constraint for i
    if(i<=3):
        return m.q[i] == m.q0[i]*pow(m.v[i],m.beta) *(1+m.KQF*(m.w-m.w0))
    else:
        return m.q[i] == (1/m.nq[i])*(m.V0-m.v[i])
    
    
model.cons1 = pyo.Constraint(model.I,rule=ax_constraint_rule)
model.cons2 = pyo.Constraint(model.I,rule=ax_constraint_rule2)
model.cons3 = pyo.Constraint(model.B,rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.B,rule=ax_constraint_rule4)
model.cons5 = pyo.Constraint(model.J,rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.J,rule=ax_constraint_rule6)


model.name = "Deneme"
opt = pyo.SolverFactory("ipopt")
instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
opt.options['max_iter'] = 10000
opt.options['ma27_pivtol'] = 1e-1
results = opt.solve(instance,tee=True)
instance.display()


# data = pyo.DataPortal()
# data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat",model=model)


for parmobject in instance.component_objects(pyo.Param, active=True):
    nametoprint = str(str(parmobject.name))
    print ("Parameter ", nametoprint)  
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print ("   ",index, vtoprint)  