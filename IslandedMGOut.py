import pyo as pyo
import pandas as pd
import pyomo.environ as pyo
# noinspection PyUnresolvedReferences
from pyomo.environ import NonNegativeReals
from pyomo.util.infeasible import log_infeasible_constraints
from IslandedMG import instance
import logging

for v in instance.component_objects(pyo.Var, active=True):
    print("Variable", v)
    for index in v:
        print("   ", index, pyo.value(v[index]))

print("Print values for all variables")
for v in instance.component_data_objects(pyo.Var):
    print(str(v), v.value)

for row in instance.A:
    for column in instance.T:
        print(pyo.value(instance.v[row, column]))

for i, j in instance.line:
    for k in instance.T:
        print(pyo.value(instance.theta[i, j, k]))

log_infeasible_constraints(instance, log_expression=True, log_variables=True)
logging.basicConfig(filename='example.log', level=logging.INFO)
