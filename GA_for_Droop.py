from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

from pyomo.opt import SolverStatus, TerminationCondition
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize
import pandas as pd
import cmath
import pyomo.environ as pyo
import math

from pyomo.opt import SolverStatus
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.environ import value, NonNegativeReals
import numpy as np

def evFun(X):
    dfR = pd.read_csv('datFiles/dat30R.csv', header=None, sep='\t')
    dfX = pd.read_csv('datFiles/dat30X.csv', header=None, sep='\t')
    dict_complex = dict()
    dict_mag = dict()
    dict_the = dict()
    edges = []

    for i in range(0, dfR.shape[0]):
        for j in range(0, dfR.shape[1]):
            if i != j and dfR.loc[i, j] != 0:
                dict_complex[i, j] = - (1 / complex(dfR.loc[i, j], dfX.loc[i, j]))
                edges.append((i, j))
            else:
                dict_complex[i, j] = complex(0, 0)

    for i in range(0, dfR.shape[0]):
        total = complex(0, 0)
        for j in edges:
            if i == j[0]:
                total -= dict_complex[j]
        dict_complex[i, i] = total

    for i in range(0, dfR.shape[0]):
        for j in range(0, dfR.shape[1]):
            dict_mag[i, j], dict_the[i, j] = cmath.polar(dict_complex[i, j])

    del i, j

    model = pyo.ConcreteModel()
    model.N = pyo.Param(default=int(math.sqrt(len(dict_mag))))
    model.KPF = pyo.Param(initialize=1)
    model.KQF = pyo.Param(initialize=-1)
    model.alpha = pyo.Param(initialize=1)
    model.beta = pyo.Param(initialize=1)

    model.J = pyo.RangeSet(0, value(model.N) - 1)
    model.drgenSet = pyo.Set(initialize={0, 1, 12, 21, 22, 26})
    model.digenSet = pyo.Set(initialize={})
    model.S = pyo.Set(initialize={0, 1, 2, 3, 4, 5})
    model.renGen = pyo.Set(initialize={5, 9, 28})

    red_scenes = np.array([[0.0142871, 0.01661571, 0.0151843, 0.01447392, 0.01459851,
                            0.01648786, 0.01444571, 0.01712976, 0.01375529, 0.01401984,
                            0.0144069, 0.01544561, 0.01513461, 0.01652935, 0.01589923,
                            0.01483155, 0.01622019, 0.01617301, 0.01571031, 0.01440533,
                            0.01548189, 0.00398234, 0., 0.],
                           [0.01369282, 0.01419896, 0.01482991, 0.01365515, 0.01386188,
                            0.01542088, 0.01577201, 0.01607194, 0.01566899, 0.0141918,
                            0.01460279, 0.01411875, 0.01495357, 0.01329354, 0.01645609,
                            0.01401912, 0.01509148, 0.01489963, 0.01580174, 0.01446995,
                            0.01698059, 0., 0., 0.02341324],
                           [0.01440596, 0.01425452, 0.01633388, 0.01566497, 0.01515667,
                            0.01452574, 0.01456681, 0.01453036, 0.01389228, 0.01522612,
                            0.01502856, 0.01648278, 0.01469467, 0.01396081, 0.01346684,
                            0.01586603, 0.01351808, 0.01559508, 0.01458911, 0.01503125,
                            0.01530392, 0., 0.01678418, 0.],
                           [0.01450568, 0.01599863, 0.01466543, 0.01402471, 0.01516821,
                            0.01608881, 0.01540415, 0.01584648, 0.01702492, 0.01388498,
                            0.01538247, 0.01497677, 0.01649909, 0.01591028, 0.01480524,
                            0.01615183, 0.01472897, 0.01571282, 0.01387545, 0.01496012,
                            0.0153322, 0.01779982, 0., 0.],
                           [0.01473619, 0.01484334, 0.01539121, 0.01322294, 0.01372942,
                            0.01358307, 0.01411846, 0.01569493, 0.01411612, 0.01677775,
                            0.01464913, 0.01418254, 0.0154116, 0.01429859, 0.01565686,
                            0.01560324, 0.0158023, 0.01546488, 0.01512918, 0.01515104,
                            0.01636043, 0.01036646, 0.02721083, 0.],
                           [0.01398067, 0.0144375, 0.01403121, 0.01478536, 0.01533134,
                            0.01366032, 0.01532976, 0.01530912, 0.01720947, 0.01606686,
                            0.01409194, 0.01553819, 0.01448271, 0.0154707, 0.01486835,
                            0.01428264, 0.01534556, 0.01522338, 0.01507315, 0.01755332,
                            0.01547741, 0., 0., 0.00837387]])

    red_probs = np.array([0.3866797485561724,
                          0.14159303981175864,
                          0.17084273795134894,
                          0.11060358774509084,
                          0.07213292449978555,
                          0.11814796143584368])

    nScenarios = len(red_scenes)
    a = np.zeros(nScenarios)
    # renGen = np.zeros(shape=(nScenarios, len(model.rengenSet)))
    prenGen = dict()
    qrenGen = dict()

    for i in model.J:
        if i in model.drgenSet:
            if i < len(red_scenes[0]):
                red_scenes = np.insert(red_scenes, i, a, axis=1)
            else:
                red_scenes = np.append(red_scenes, np.zeros(shape=(nScenarios, 1)), axis=1)

    for i in model.J:
        if i in model.renGen:
            for s in model.S:
                prenGen[s, i] = red_scenes[s, i]
                qrenGen[s, i] = red_scenes[s, i] * 0.6
            red_scenes[:, i] = np.zeros(nScenarios)

    d1 = {}
    for i in range(0, 6):
        for j in range(0, 30):
            d1[i, j] = red_scenes[i, j]

    PF = 0.6
    d2 = {}
    for k, v in d1.items():
        d2[k] = v * PF

    model.SPROBS = pyo.Param(model.S, initialize=red_probs)
    model.p0 = pyo.Param(model.S * model.J, initialize=d1)
    model.q0 = pyo.Param(model.S * model.J, initialize=d2)
    d3, d4 = dict(), dict()

    # for i in renGen:
    #     for j in model.rengenSet:
    #         d3[i, j] = renGen[i]
    #         d4[i, j] = renGen[i] * PF

    model.PR = pyo.Param(model.S * model.renGen, initialize=prenGen)
    model.QR = pyo.Param(model.S * model.renGen, initialize=qrenGen)
    model.w0 = pyo.Param(initialize=1.0)
    model.V0 = pyo.Param(initialize=1.01)
    model.SGmax = pyo.Param(model.drgenSet, initialize={0: 0.05, 1: 0.06, 12: 0.05, 21: 0.05, 22: 0.06, 26: 0.05})

    model.yMag = pyo.Param(model.J, model.J, initialize=dict_mag)
    model.yThe = pyo.Param(model.J, model.J, initialize=dict_the)

    model.ql = pyo.Var(model.S * model.J, initialize=0, within=NonNegativeReals, bounds=(0, 2))
    model.pl = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.pg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))
    model.qg = pyo.Var(model.S * model.J, initialize=0, within=pyo.NonNegativeReals, bounds=(0, 2))

    model.v = pyo.Var(model.S * model.J, domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0.9, 1.1))
    model.d = pyo.Var(model.S * model.J, domain=pyo.Reals, initialize=1.0, bounds=(-math.pi / 2, math.pi / 2))

    # model.mp = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.03, bounds=(0, 1))
    # model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(0, 1))
    x = [X['mp1'],X['mp2'], X['mp3'],X['mp4'],X['mp5'],X['mp6']] 
    y = [X['nq1'],X['nq2'], X['nq3'],X['nq4'],X['nq5'],X['nq6']]

    model.mp = pyo.Param(model.drgenSet, initialize=dict(zip(model.drgenSet, x)))
    model.nq = pyo.Param(model.drgenSet, initialize=dict(zip(model.drgenSet, y)))
    # model.nq = pyo.Var(model.drgenSet, domain=pyo.NonNegativeReals, initialize=0.01, bounds=(0, 1))

    model.w = pyo.Var(model.S, domain=pyo.NonNegativeReals, initialize=1, bounds=(0.990, 1.00))

    # def obj_expression(m):
    #     return sum(model.SPROBS[s] * sum((model.yMag[i, j] * model.v[s, i] * model.v[s, j]) *
    #                                      pyo.cos(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) +
    #                                      (-1 / 2) * (model.yMag[i, j] * model.v[s, i] * model.v[s, j]) * model.w[s] *
    #                                      pyo.sin(model.yThe[i, j] + model.d[s, i] + model.d[s, j]) for j in m.J) for
    #                s, i in m.S * m.J)

    def obj_expression(m):
        return sum(model.SPROBS[s] * pow((m.v[s, i] - 1.0), 2) for s, i in m.S * m.J)

    model.o = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    def ax_constraint_rule3(m, s, i):
        if i in m.renGen:
            return (m.pg[s, i] + m.PR[s, i] - m.pl[s, i]) - sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0
        else:
            return (m.pg[s, i] - m.pl[s, i]) - sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.cos(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0

    def ax_constraint_rule4(m, s, i):
        if i in m.renGen:
            return (m.qg[s, i] + m.QR[s, i] - m.ql[s, i]) + sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0
        else:
            return (m.qg[s, i] - m.ql[s, i]) + sum(
                m.v[s, i] * m.v[s, j] * m.yMag[i, j] * pyo.sin(m.yThe[i, j] + m.d[s, j] - m.d[s, i]) for j in m.J) == 0

    def ax_constraint_rule5(m, s, i):
        if i in m.drgenSet:
            # return m.pg[s, i] == (1 / m.mp[i]) * (m.w0 - m.w[s])
            return m.pg[s, i] == 0.5 * (
                    ((1 / model.mp[i]) * (model.w0 - model.w[s])) + ((1 / model.nq[i]) * (model.V0 - model.v[s, i])))
        elif i in m.digenSet:
            return m.pg[s, i] <= 0.1
        else:
            return m.pg[s, i] == 0

    def ax_constraint_rule6(m, s, i):
        if i in m.drgenSet:
            # return m.qg[s, i] == (1 / m.nq[i]) * (m.V0 - m.v[s, i])
            return m.qg[s, i] == 0.5 * (
                    ((1 / model.nq[i]) * (model.V0 - model.v[s, i])) - ((1 / model.mp[i]) * (model.w0 - model.w[s])))
        elif i in m.digenSet:
            return m.qg[s, i] <= 0.06
        else:
            return m.qg[s, i] == 0

    # Frequency & voltage dependent load constraints
    def ax_constraint_rule7(m, s, i):
        return m.pl[s, i] == m.p0[s, i] * pow(m.v[s, i] / m.V0, m.alpha) * (1 + m.KPF * (m.w[s] - m.w0))

    def ax_constraint_rule8(m, s, i):
        return m.ql[s, i] == m.q0[s, i] * pow(m.v[s, i] / m.V0, m.beta) * (1 + m.KQF * (m.w[s] - m.w0))

    def maxGenCons(m, s, i):
        if i in m.drgenSet:
            return pyo.sqrt(pow(m.pg[s, i], 2) + pow(m.qg[s, i], 2)) <= model.SGmax[i]
        else:
            return pyo.Constraint.Skip

    model.cons3 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule3)
    model.cons4 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule4)
    model.cons5 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule5)
    model.cons6 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule6)
    model.cons7 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule7)
    model.cons8 = pyo.Constraint(model.S * model.J, rule=ax_constraint_rule8)
    model.cons20 = pyo.Constraint(model.S * model.J, rule=maxGenCons)

    model.name = "DroopControlledIMG"
    opt = pyo.SolverFactory("ipopt")
    # opt.options['acceptable_tol'] = 1e-3
    # instance.pprint()
    opt.options['max_iter'] = 100000000

    # log_infeasible_constraints(model, log_expression=True, log_variables=True)
    # logging.basicConfig(filename='example2.log', level=logging.INFO)

    results = opt.solve(model, tee=False)
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        return pyo.value(model.o)
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        return 1e10




class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "mp1": Real(bounds=(0, 1)),
            "mp2": Real(bounds=(0, 1)),
            "mp3": Real(bounds=(0, 1)),
            "mp4": Real(bounds=(0, 1)),
            "mp5": Real(bounds=(0, 1)),
            "mp6": Real(bounds=(0, 1)),
            "nq1": Real(bounds=(0, 1)),
            "nq2": Real(bounds=(0, 1)),
            "nq3": Real(bounds=(0, 1)),
            "nq4": Real(bounds=(0, 1)),
            "nq5": Real(bounds=(0, 1)),
            "nq6": Real(bounds=(0, 1))

        }
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        f = evFun(X)
        out["F"] = f

problem = MixedVariableProblem()

algorithm = MixedVariableGA(pop=100)

res = minimize(problem,
               algorithm,
               termination=('n_evals', 1000),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

X = {'mp1': 0.417022004702574, 'mp2': 0.019366957870297075, 'mp3': 0.32664490177209615, 'mp4': 0.07197427968948678, 'mp5': 0.9501761192470797, 'mp6': 0.00040202489135743313,
     'nq1': 0.8118586977205398, 'nq2': 0.7702387345549799, 'nq3': 0.9594343210617524, 'nq4': 0.187631672902434, 'nq5': 0.08748220955359087, 'nq6': 0.33870802548167966}