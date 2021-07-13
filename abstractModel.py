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

def _back_diff(m,i):
    return m.dzdt[i] == (m.z[i]-m.z[i-1])/m.h
m.back_diff = Constraint
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
px=[]
py=[]
for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(model, str(v))
    for index in varobject:
        px.append(index)
        py.append(varobject[index].value)
        print ("   ",index, varobject[index].value)
        


def z(x):
    return (4*x-3)/(4*x+1)

import matplotlib.pyplot as plt
import numpy as np

type(px)
type(x)

x= np.linspace(0,1,10)
y = (4*x-3)/(4*x+1)

plt.scatter(px,py,color="blue")
plt.plot(x,y)
plt.show()

#%% Partial deifferential equation discretizer (PDE) example 

import math
from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation
from pyomo.dae.plugins.colloc import Collocation_Discretization_Transformation
from pyomo.opt import SolverFactory
 
m = ConcreteModel()
m.t = ContinuousSet(bounds=(0,2))
m.x = ContinuousSet(bounds=(0,1))
m.u = Var(m.x,m.t)

m.dudx = DerivativeVar(m.u,wrt=m.x)
m.dudx2 = DerivativeVar(m.u,wrt=(m.x,m.x))
m.dudt = DerivativeVar(m.u,wrt=m.t)

def _pde(m,i,j):
    if i==0 or i==1:
        return Constraint.Skip
    return math.pi**2*m.dudt[i,j] == m.dudx2[i,j]
m.pde = Constraint(m.x,m.t,rule=_pde)

def _initcon(m,i):
    if i==0 or i==1:
        return Constraint.Skip
    return m.u[i,0] == sin(math.pi*i)
m.initcon = Constraint(m.x,rule=_initcon)

def _lowerbound(m,j):
    return m.u[0,j] == 0
m.lowerbound = Constraint(m.t,rule=_lowerbound)

def _upperbound(m,j):
    return math.pi*exp(-j)+m.dudx[1,j]==0
m.upperbound = Constraint(m.t,rule=_upperbound)

discretize = Finite_Difference_Transformation()
disc = discretize.apply(m,nfe=25,wrt=m.x,scheme='BACKWARD')
disc = discretize.apply(disc,nfe=20,wrt=m.t,scheme='BACKWARD',clonemodel=False)
solver = 'ipopt'
opt = SolverFactory(solver)
results = opt.solve(disc,tee=True) 
disc.load(results)
#%%
from pyomo.environ import *
numpoints = 10
model = m = ConcreteModel()
m.points = RangeSet(0,numpoints)
m.h = Param(initialize=1.0/numpoints)
m.z = Var(m.points)
m.dzdt = Var(m.points)
m.obj = Objective(expr=1) # Dummy Objective
def _zdot(m, i):
    return m.dzdt[i] == m.z[i]**2 - 2*m.z[i] + 1
m.zdot = Constraint(m.points, rule=_zdot)
def _back_diff(m,i):
    if i == 0:
        return Constraint.Skip
    return m.dzdt[i] == (m.z[i]-m.z[i-1])/m.h
m.back_diff = Constraint(m.points, rule=_back_diff)

def _init_con(m):
    return m.z[0] == -3
m.init_con = Constraint(rule=_init_con)

solver= SolverFactory('ipopt')
results= solver.solve(model,tee=True)
results.write(num=1)
model.pprint()

for v in model.component_objects(Var, active=True):
    print ("Variable component object",v)
    print ("Type of component object: ", str(type(v))[1:-1]) # Stripping <> for nbconvert
    varobject = getattr(model, str(v))
    print ("Type of object accessed via getattr: ", str(type(varobject))[1:-1])
    for index in varobject:
        print ("   ", index, varobject[index].value)
#%%

from pyomo.environ import *
from pyomo.dae import *
model = m = ConcreteModel()
m.tf = Param(initialize = 1)
m.t = ContinuousSet(bounds=(0, m.tf))
m.u = Var(m.t, initialize=0)
m.x1 = Var(m.t)
m.x2 = Var(m.t)
m.x3 = Var(m.t)
m.dx1dt = DerivativeVar(m.x1, wrt=m.t)
m.dx2dt = DerivativeVar(m.x2, wrt=m.t)
m.dx3dt = DerivativeVar(m.x3, wrt=m.t)
m.obj = Objective(expr=m.x3[m.tf])
def _x1dot(m, t):
    return m.dx1dt[t] == m.x2[t]
m.x1dot = Constraint(m.t, rule=_x1dot)
def _x2dot(m, t):
    return m.dx2dt[t] == -m.x2[t] + m.u[t]
m.x2dot = Constraint(m.t, rule=_x2dot)
def _x3dot(m, t):
    return m.dx3dt[t] == m.x1[t]**2 + \
m.x2[t]**2 + 0.005*m.u[t]**2
m.x3dot = Constraint(m.t, rule=_x3dot)
def _con(m, t):
    return m.x2[t] - 8*(t-0.5)**2 + 0.5 <= 0
m.con = Constraint(m.t, rule=_con)
def _init(m):
    yield m.x1[0] == 0
    yield m.x2[0] == -1
    yield m.x3[0] == 0
m.init_conditions = ConstraintList(rule=_init)

TransformationFactory('dae.collocation').apply_to(
m, nfe=7, ncp=6, scheme='LAGRANGE-RADAU' )
# Solve algebraic model
results = SolverFactory('ipopt').solve(m,tee=True)

def plotter(subplot, x, *series, **kwds):
    plt.subplot(subplot)
    for i,y in enumerate(series):
        plt.plot(x, [value(y[t]) for t in x], 'brgcmk'[i%6]+kwds.get('points',''))
    plt.title(kwds.get('title',''))
    plt.legend(tuple(y.cname() for y in series))
    plt.xlabel(x.cname())
        
import matplotlib.pyplot as plt
plotter(121, m.t, m.x1, m.x2, title='Differential Variables')
plotter(122, m.t, m.u, title='Control Variable', points='o')
plt.show()
#%%

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator


def create_model():
    m = ConcreteModel()

    m.t = ContinuousSet(bounds=(0.0, 10.0))

    m.b = Param(initialize=0.25)
    m.c = Param(initialize=5.0)

    m.omega = Var(m.t)
    m.theta = Var(m.t)

    m.domegadt = DerivativeVar(m.omega, wrt=m.t)
    m.dthetadt = DerivativeVar(m.theta, wrt=m.t)

    # Setting the initial conditions
    m.omega[0] = 0.0
    m.theta[0] = 3.14 - 0.1

    def _diffeq1(m, t):
        return m.domegadt[t] == -m.b * m.omega[t] - m.c * sin(m.theta[t])
    m.diffeq1 = Constraint(m.t, rule=_diffeq1)

    def _diffeq2(m, t):
        return m.dthetadt[t] == m.omega[t]
    m.diffeq2 = Constraint(m.t, rule=_diffeq2)

    return m


def simulate_model(m):
    if False:
        # Simulate the model using casadi
        sim = Simulator(m, package='casadi')
        tsim, profiles = sim.simulate(numpoints=100, integrator='cvodes')
    else:
        # Simulate the model using scipy
        sim = Simulator(m, package='scipy')
        tsim, profiles = sim.simulate(numpoints=100, integrator='vode')

    # Discretize model using Orthogonal Collocation
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=8, ncp=5)

    # Initialize the discretized model using the simulator profiles
    sim.initialize_model()

    return sim, tsim, profiles


def plot_result(m, sim, tsim, profiles):
    import matplotlib.pyplot as plt

    time = list(m.t)
    omega = [value(m.omega[t]) for t in m.t]
    theta = [value(m.theta[t]) for t in m.t]

    varorder = sim.get_variable_order()

    for idx, v in enumerate(varorder):
        plt.plot(tsim, profiles[:, idx], label=v)
    plt.plot(time, omega, 'o', label='omega interp')
    plt.plot(time, theta, 'o', label='theta interp')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()

model = create_model()
sim, tsim, profiles = simulate_model(model)
plot_result(model, sim, tsim, profiles)
