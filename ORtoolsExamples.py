# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:34:58 2021

@author: Administrator
"""


from ortools.linear_solver import pywraplp

# %%
solver = pywraplp.Solver.CreateSolver('GLOP')

# Create the variables x and y.
x = solver.NumVar(0, 1, 'x')
y = solver.NumVar(0, 2, 'y')

print('Number of variables =', solver.NumVariables())

# %% Create a linear constraint, 0 <= x + y <= 2.
ct = solver.Constraint(0, 2, 'ct')
ct.SetCoefficient(x, 1)
ct.SetCoefficient(y, 1)

print('Number of constraints =', solver.NumConstraints())

#%% Create the objective function, 3 * x + y.
objective = solver.Objective()
objective.SetCoefficient(x, 3)
objective.SetCoefficient(y, 1)
objective.SetMaximization()
#%%
solver.Solve()
print('Solution:')
print('Objective value =', objective.Value())
print('x =', x.solution_value())
print('y =', y.solution_value())

#%%
from pyomo.environ import Objective
from pyomo.environ import ConcreteModel
from pyomo.environ import Var
from pyomo.environ import minimize
from pyomo.environ import ConstraintList
from pyomo.environ import SolverFactory
from pyomo.environ import value
# The mathematical operators from the pyomo package should be used instead of math package
from pyomo.environ import cos

model = ConcreteModel()
model.x = Var()
model.y = Var()
def rosenbrock(m):
    return (1.0-m.x)**2 + 100.0*(m.y - m.x**2)**2

model.obj = Objective(rule=rosenbrock, sense=minimize)
model.limits =  ConstraintList()
model.limits.add(model.x + model.y == 1)
model.limits.add(cos(model.x) == 1)

solver = SolverFactory('ipopt')
solver.solve(model, tee=True)

print('*** Solution *** :')
print('x:', value(model.x))
print('y:', value(model.y))

#%%
