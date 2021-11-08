#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Imports
#
import sys

from pyomo.core import *
import pyomo.environ
import pyomo.environ as pyo

model = AbstractModel()

model.A = Set()

model.B = Set()

model.C = model.A * model.B

model.D = Set(within=model.A * model.B)
#
# A multiple cross-product
#
model.E = Set(within=model.A * model.B * model.A)

#
# An indexed set
#
model.F = Set(model.A)
#
# An indexed set
#
model.G = Set(model.A, model.B)
#
# A simple set
#
model.H = Set()
#
# A simple set
#
model.I = Set()
#
# A two-dimensional set
#
model.J = Set(dimen=2)

model.Z = Param()

model.Y = Param(model.A)

model.X = Param(model.A)
model.W = Param(model.A)

model.U = Param(model.I, model.A)
model.T = Param(model.A, model.I)

model.S = Param(model.A)

model.R = Param(model.H, within=pyo.Reals)
model.Q = Param(model.H, within=pyo.Reals)

#
# An example of initializing parameters with a two-dimensional index set
#
model.P = Param(model.J, within=pyo.Reals)
model.PP = Param(model.J, within=pyo.Reals)
model.O = Param(model.J, within=pyo.Reals)


# model.pprint()
filenamee = "C:\\Users\\Administrator\\Desktop\\Datas\\excel.xls"
data = pyo.DataPortal(model=model)
data.load(filename=filenamee, range="Atable", format='set', set='A')
data.load(filename=filenamee, range="Btable", format='set', set='B')
data.load(filename=filenamee, range="Ctable", format='set', set='C')

data.load(filename=filenamee, range="Dtable", format='set_array', set='D')

data.load(filename=filenamee, range="Etable", format='set', set='E')
data.load(filename=filenamee, range="Itable", format='set', set='I')

data.load(filename=filenamee, range="Zparam", format='param', param='Z')
data.load(filename=filenamee, range="Ytable", index='A', param='Y')
data.load(filename=filenamee, range="XWtable", index='A', param=['X', 'W'])
data.load(filename=filenamee, range="Ttable", param='T', format='transposed_array')
data.load(filename=filenamee, range="Utable", param='U', format='array')
data.load(filename=filenamee, range="Stable", index='A', param='S')
data.load(filename=filenamee, range="RQtable", index='H', param=('R', 'Q'))
data.load(filename=filenamee, range="POtable", index='J', param=('P', 'O'))
data.load(filename=filenamee, range="PPtable", index=('A', 'B'), param="PP")

# try:
#     data.read()
# except pyomo.ApplicationError:
#     sys.exit(0)

instance = model.create_instance(data)
instance.pprint()
