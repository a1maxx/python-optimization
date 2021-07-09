# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:06:04 2021

@author: Administrator
"""

# pyomo --solver=ipopt "C:\\Users\\Administrator\\Py Files\\abstractModel.py" "C:\\Users\\Administrator\\Py Files\\model.dat"

import pyomo.environ as pyo
from pyomo.environ import NonNegativeReals

model = pyo.AbstractModel()
model.N = pyo.Param(default=6)
model.KPF = pyo.Param(initialize=1)
model.KQF = pyo.Param(initialize=-1)
model.alpha = pyo.Param(initialize=1)
model.beta = pyo.Param(initialize=1)

model.I = pyo.RangeSet(1,3)
model.J = pyo.RangeSet(1,model.N)
model.B = pyo.RangeSet(4,model.N)

model.p0 = pyo.Param(model.J)
model.q0 = pyo.Param(model.J)
model.y = pyo.Param(model.J,model.J)
model.t = pyo.Param(model.J,model.J)
model.w0 = pyo.Param(initialize= 1.0)
model.V0 = pyo.Param(initialize= 1.01)
# model.SGmax = pyo.Param(model.B,initialize=1.0)


model.ql = pyo.Var(model.J,initialize=0,within=NonNegativeReals)
model.pl = pyo.Var(model.J,initialize=0,within=NonNegativeReals) 
model.pg = pyo.Var(model.J,initialize=0,within=NonNegativeReals)
model.qg = pyo.Var(model.J,initialize=0,within=NonNegativeReals)

model.v = pyo.Var(model.J,domain=pyo.NonNegativeReals,initialize=1.0)
model.d = pyo.Var(model.J,domain=pyo.Reals,initialize=0)

model.mp = pyo.Var(model.B,domain=pyo.NonNegativeReals,initialize=0.5,bounds=(0,1))
model.nq = pyo.Var(model.B,domain=pyo.NonNegativeReals,initialize=0.5,bounds=(0,1))

model.w = pyo.Var(domain=pyo.PositiveReals)

data = pyo.DataPortal()
data.load(filename="C:\\Users\\Administrator\\Py Files\\model.dat",model=model)
model2 = model.create_instance(data)


def obj_expression(m):
    return sum(abs(m.v[i]-1.0) for i in m.v)

model.o = pyo.Objective(rule=obj_expression,sense=1)

# PQ buses
def ax_constraint_rule(m, i):
    # return the expression for the constraint for i
    return sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.cos(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) - m.pl[i]   == 0

def ax_constraint_rule2(m, i):
    # return the expression for the constraint for i
    return   -sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.sin(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) - m.ql[i] == 0


## Droop buses
def ax_constraint_rule3(m, i):
    # return the expression for the constraint for i
    return -((1/m.mp[i])*(m.w0-m.w) - m.pl[i]) + sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.cos(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) == 0

def ax_constraint_rule4(m, i):
    # return the expression for the constraint for i
    return -((1/m.nq[i])*(m.V0-m.v[i]) - m.ql[i]) - sum(m.v[i]*m.v[j]*m.y[i,j]*pyo.sin(m.t[i,j]+m.d[j]-m.d[i]) for j in m.J) == 0

def ax_constraint_rule5(m, i):
    if(i>3):
        return m.pg[i] == (1/m.mp[i])*(m.w0-m.w)

def ax_constraint_rule6(m, i):
    if(i>3):
        return m.qg[i] == (1/m.nq[i])*(m.V0-m.v[i])    

## Frequency dependent load and generation constraints
# def ax_constraint_rule5(m, i):
#     if(i<=3):
#         return m.pl[i] == m.p0[i]*pow(m.v[i],m.alpha) *(1+m.KPF*(m.w-m.w0))
#     else:
#         return m.pg[i] == (1/m.mp[i])*(m.w0-m.w)
    
# ## Voltage dependent load and generation constraints
# def ax_constraint_rule6(m, i):
#     if(i<=3):
#         return m.ql[i] == m.q0[i]*pow(m.v[i],m.beta) *(1+m.KQF*(m.w-m.w0))
#     else:
#         return m.qg[i] == (1/m.nq[i])*(m.V0-m.v[i])
    
def ax_constraint_rule7(m, i):    
        return m.pg[i] <= 0.08

def ax_constraint_rule8(m, i):    
        return m.qg[i] <= 0.06     
    
def ax_constraint_rule9(m, i):    
        return m.pl[i] == 0

def ax_constraint_rule10(m, i):    
        return m.ql[i] == 0  

model.cons1 = pyo.Constraint(model.I,rule=ax_constraint_rule)
model.cons2 = pyo.Constraint(model.I,rule=ax_constraint_rule2)

model.cons3 = pyo.Constraint(model.B,rule=ax_constraint_rule3)
model.cons4 = pyo.Constraint(model.B,rule=ax_constraint_rule4)
model.cons7 = pyo.Constraint(model.B,rule=ax_constraint_rule7)
model.cons8 = pyo.Constraint(model.B,rule=ax_constraint_rule8)
model.cons9 = pyo.Constraint(model.B,rule=ax_constraint_rule7)
model.cons10 = pyo.Constraint(model.B,rule=ax_constraint_rule8)

model.cons5 = pyo.Constraint(model.B,rule=ax_constraint_rule5)
model.cons6 = pyo.Constraint(model.B,rule=ax_constraint_rule6)


model.name = "DroopControlledIMG"
opt = pyo.SolverFactory("ipopt")
instance = model.create_instance(filename=("C:\\Users\\Administrator\\Py Files\\model.dat"))
opt.options['max_iter'] = 100000
opt.options['ma27_pivtol'] = 1e-5
results = opt.solve(instance,tee=True)
instance.display()

instance.pprint()

for parmobject in instance.component_objects(pyo.Param, active=True):
    nametoprint = str(str(parmobject.name))
    print ("Parameter ", nametoprint)  
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print ("   ",index, vtoprint)  
        
#%% Differential equation solver example
from pyomo.environ import *
from pyomo.dae import *
model = m = ConcreteModel()
m.t = ContinuousSet(bounds=(0, 1))
m.z = Var(m.t)
m.dzdt = DerivativeVar(m.z, wrt=m.t)
def _zdot(m, t):
    return m.dzdt[t] == m.z[t]**2 - 2*m.z[t] + 1
m.zdot = Constraint(m.t, rule=_zdot)

def _init_con(m):
    return m.z[0] == -3
m.init_con = Constraint(rule= _init_con)

# Discretize model using backward finite difference
discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m,nfe=10,scheme='BACKWARD')

solver= SolverFactory('ipopt')
results= solver.solve(model,tee=True)

if results.solver.termination_condition == TerminationCondition.optimal: 
    model.solutions.load_from(results)

for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(model, str(v))
    for index in varobject:
        print ("   ",index, varobject[index].value)


def ff(x):
    return x**2 - 2*x + 1

#%% Partial deifferential equation discretizer (PDE) example 

# Example 1 from http://www.mathworks.com/help/matlab/ref/pdepe.html

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation
from pyomo.dae.plugins.colloc import Collocation_Discretization_Transformation
import math

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0,2))
m.x = ContinuousSet(bounds=(0,1))
m.u = Var(m.x,m.t)

m.dudx = DerivativeVar(m.u,wrt=m.x)
m.dudx2 = DerivativeVar(m.u,wrt=(m.x,m.x))
m.dudt = DerivativeVar(m.u,wrt=m.t)

def _pde(m,i,j):
    if i == 0 or i == 1 or j == 0 :
        return Constraint.Skip
    return math.pi**2*m.dudt[i,j] == m.dudx2[i,j]
m.pde = Constraint(m.x,m.t,rule=_pde)

def _initcon(m,i):
    if i == 0 or i == 1:
        return Constraint.Skip
    return m.u[i,0] == sin(math.pi*i)
m.initcon = Constraint(m.x,rule=_initcon)

def _lowerbound(m,j):
    return m.u[0,j] == 0
m.lowerbound = Constraint(m.t,rule=_lowerbound)

def _upperbound(m,j):
    return math.pi*exp(-j)+m.dudx[1,j] == 0
m.upperbound = Constraint(m.t,rule=_upperbound)

m.obj = Objective(expr=1)

# Discretize using Finite Difference Method

discretizer = TransformationFactory('dae.finite_difference')
disc = discretizer.apply_to(m,nfe=25,wrt=m.x,scheme='BACKWARD')
disc = discretizer.apply_to(disc,nfe=20,wrt=m.t,scheme='BACKWARD',clonemodel=False)

# Discretize using Orthogonal Collocation
#discretize2 = Collocation_Discretization_Transformation()
#disc = discretize2.apply(disc,nfe=10,ncp=3,wrt=m.x,clonemodel=False)
#disc = discretize2.apply(disc,nfe=20,ncp=3,wrt=m.t,clonemodel=False)


solver='ipopt'
opt=SolverFactory(solver)

results = opt.solve(disc,tee=True)
disc.load(results)

#disc.u.pprint()

x = []
t = []
u = []

for i in sorted(disc.x):
    temp=[]
    tempx = []
    for j in sorted(disc.t):
        tempx.append(i)
        temp.append(value(disc.u[i,j]))
    x.append(tempx)
    t.append(sorted(disc.t))
    u.append(temp)


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_xlabel('Distance x')
ax.set_ylabel('Time t')
ax.set_title('Numerical Solution Using Backward Difference Method')
p = ax.plot_wireframe(x,t,u,rstride=1,cstride=1)
fig.show()



