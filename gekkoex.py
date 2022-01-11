from gekko import GEKKO
import numpy as np


# Initialize Model
m = GEKKO()
x = m.Array(m.Var, 4, value=1, lb=1, ub=5)

x1, x2, x3, x4 = x  # rename variables
x2.value = 5
x3.value = 5  # change guess

m.Equation(np.prod(x) >= 25)  # prod>=25
m.Equation(m.sum([xi ** 2 for xi in x]) == 40)  # sum=40

m.Minimize(x1 * x4 * (x1 + x2 + x3) + x3)  # objective
m.solve()
print(x)
