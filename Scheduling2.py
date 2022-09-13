from pyomo.environ import ConcreteModel, minimize, SolverFactory, Set, Param, Var, Constraint, Objective, \
    TransformationFactory
from pyomo.gdp import Disjunction
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def jobshop_model(TASKS):
    model = ConcreteModel()

    # tasks is a two dimensional set of (j,m) constructed from the dictionary keys
    model.TASKS = Set(initialize=TASKS.keys(), dimen=2)

    # the set of jobs is constructed from a python set
    model.JOBS = Set(initialize=list(set([j for (j, m) in model.TASKS])))

    # set of machines is constructed from a python set
    model.MACHINES = Set(initialize=list(set([m for (j, m) in model.TASKS])))

    # the order of tasks is constructed as a cross-product of tasks and filtering
    model.TASKORDER = Set(initialize=model.TASKS * model.TASKS, dimen=4,
                          filter=lambda model, j, m, k, n: (k, n) == TASKS[(j, m)]['prec'])

    # the set of disjunctions is cross-product of jobs, jobs, and machines
    model.DISJUNCTIONS = Set(initialize=model.JOBS * model.JOBS * model.MACHINES, dimen=3,
                             filter=lambda model, j, k, m: j < k and (j, m) in model.TASKS and (k, m) in model.TASKS)

    # load duration data into a model parameter for later access
    model.dur = Param(model.TASKS, initialize=lambda model, j, m: TASKS[(j, m)]['dur'])

    model.taskPri = Param(model.TASKS, initialize=lambda model, j, m: TASKS[(j, m)]["pri"])

    # establish an upper bound on makespan
    ub = sum([model.dur[j, m] for (j, m) in model.TASKS])

    # create decision variables
    model.makespan = Var(bounds=(0, ub))

    model.start = Var(model.TASKS, bounds=(0, ub))

    # model.objective1 = Objective(expr=model.makespan, sense=minimize)
    # model.objective1.deactivate()

    model.objective2 = Objective(expr=lambda model: sum((model.start[j, m] + model.dur[j, m]) * model.taskPri[j, m]
                                                        for (j, m) in model.TASKS), sense=minimize)
    # model.objective2.deactivate()

    model.finish = Constraint(model.TASKS, rule=lambda model, j, m:
    model.start[j, m] + model.dur[j, m] <= model.makespan)

    model.preceding = Constraint(model.TASKORDER, rule=lambda model, j, m, k, n:
    model.start[k, n] + model.dur[k, n] <= model.start[j, m])

    model.disjunctions = Disjunction(model.DISJUNCTIONS, rule=lambda model, j, k, m:
    [model.start[j, m] + model.dur[j, m] <= model.start[k, m],
     model.start[k, m] + model.dur[k, m] <= model.start[j, m]])
    #
    model.dummy = Constraint(rule=lambda model: model.start['Task_D', 'Server_1'] == 15)
    model.dummy2 = Constraint(rule=lambda model: model.start['Task_D2', 'Server_3'] == 4)

    TransformationFactory('gdp.hull').apply_to(model)
    return model


# %%

def jobshop_solve(model):
    SolverFactory('cbc').solve(model)
    results = [{'Job': j,
                'Machine': m,
                'Start': model.start[j, m](),
                'Duration': model.dur[j, m],
                'Finish': model.start[(j, m)]() + model.dur[j, m]}
               for j, m in model.TASKS]
    return results


def jobshop(TASKS):
    return jobshop_solve(jobshop_model(TASKS))


def getConsumption(schedule):
    dval = np.random.rand(int(max(schedule.Finish)))
    conLis = []
    for i in range(0, len(schedule)):
        conLis.append(sum(dval[range(int(schedule.loc[i].Start), int(schedule.loc[i].Finish))]))
    schedule['Consumption'] = conLis

    con_by_mac = schedule.groupby('Machine')['Consumption'].sum()

    return con_by_mac


def visualize(results):
    schedule = pd.DataFrame(results)
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()

    bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
    text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center', 'size': 'large'}
    # colors = mpl.cm.Dark2.colors

    colors = mpl.cm.tab10.colors
    mm = len(colors) - 1

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5 + (len(JOBS) + len(MACHINES)) / 4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), 'Start']
                xf = schedule.loc[(j, m), 'Finish']
                ax[0].plot([xs, xf], [jdx] * 2, c=colors[mdx % mm], **bar_style)
                ax[0].text((xs + xf) / 2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx] * 2, c=colors[jdx % mm], **bar_style)
                ax[1].text((xs + xf) / 2, mdx, j, **text_style)

    textpt = 18
    ax[0].set_title('Task Schedule', size=textpt, weight='bold')
    ax[0].set_ylabel('Task', size=textpt + 4)
    ax[0].tick_params(axis='x', labelsize=textpt)
    ax[0].tick_params(axis='y', labelsize=textpt)

    ax[1].set_title('ES Schedule', size=textpt, weight='bold')
    ax[1].set_ylabel('ES', size=textpt + 4)
    ax[1].tick_params(axis='x', labelsize=textpt)
    ax[1].tick_params(axis='y', labelsize=textpt)

    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top',
                     size='x-large')
        ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time', size=textpt + 4)
        ax[idx].grid(True)

    fig.tight_layout()

    return fig


# %%

if __name__ == "__main__":
    TASKS = {
        ('Task_1', 'Server_1'): {'dur': 25, 'prec': None, 'pri': 2},
        ('Task_1', 'Server_2'): {'dur': 10, 'prec': ('Task_1', 'Server_1'), 'pri': 2},
        ('Task_2', 'Server_1'): {'dur': 10, 'prec': None, 'pri': 3},
        ('Task_2', 'Server_2'): {'dur': 20, 'prec': ('Task_2', 'Server_1'), 'pri': 3},
        ('Task_2', 'Server_3'): {'dur': 12, 'prec': ('Task_2', 'Server_2'), 'pri': 3},
        ('Task_3', 'Server_2'): {'dur': 12, 'prec': ('Task_3', 'Server_3'), 'pri': 5},
        ('Task_3', 'Server_3'): {'dur': 28, 'prec': None, 'pri': 5},
        ('Task_3', 'Server_2'): {'dur': 12, 'prec': ('Task_3', 'Server_3'), 'pri': 5},
        ("Task_4", "Server_1"): {'dur': 12, 'prec': None, 'pri': 2},
        ("Task_4", "Server_2"): {'dur': 19, 'prec': ('Task_4', 'Server_1'), 'pri': 2},
        ("Task_4", "Server_3"): {'dur': 16, 'prec': ('Task_4', 'Server_2'), 'pri': 2},
        ("Task_5", "Server_2"): {'dur': 12, 'prec': None, 'pri': 2},
        ("Task_6", "Server_3"): {'dur': 10, 'prec': None, 'pri': 2},
        ("Task_6", "Server_3"): {'dur': 20, 'prec': None, 'pri': 2},
        ("Task_7", "Server_3"): {'dur': 15, 'prec': ('Task_3', 'Server_3'), 'pri': 2},
        ('Task_D', 'Server_1'): {'dur': 20, 'prec': None, 'pri': 0},
        ('Task_D2', 'Server_3'): {'dur': 12, 'prec': None, 'pri': 0}

    }

    results = jobshop(TASKS)
    schedule = pd.DataFrame(results)
    withCon = getConsumption(schedule)
    # mpl.rcParams['figure.dpi'] = 300
    # mpl.rcParams['savefig.dpi'] = 300
    fig = visualize(results)
    fig.show()
    # fig.savefig('graph6.svg')




