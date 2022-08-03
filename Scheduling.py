from pyomo.environ import ConcreteModel, minimize, SolverFactory, Set, Param, Var, Constraint, Objective, \
    TransformationFactory
from pyomo.gdp import Disjunction
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

TASKS = {
    ('Task_1', 'Curate'): {'dur': 45, 'prec': None, 'pri': 2},
    ('Task_1', 'Store'): {'dur': 10, 'prec': ('Task_1', 'Curate'), 'pri': 3},
    ('Task_2', 'Curate'): {'dur': 10, 'prec': None, 'pri': 4},
    ('Task_2', 'Store'): {'dur': 20, 'prec': ('Task_2', 'Curate'), 'pri': 5},
    ('Task_2', 'Process'): {'dur': 34, 'prec': ('Task_2', 'Store'), 'pri': 1},
    ('Task_3', 'Store'): {'dur': 12, 'prec': ('Task_3', 'Process'), 'pri': 2},
    ('Task_3', 'Process'): {'dur': 28, 'prec': None, 'pri': 1},
    ('Task_3', 'Store'): {'dur': 12, 'prec': ('Task_3', 'Process'), 'pri': 2},
    ('Task_D', 'Curate'): {'dur': 20, 'prec': None, 'pri': 1}
}
'''
dur(j,m) duration of task j computed at the edge device m
prec(j,m) a task (k,n) = prec_(j,m) must be completed before task (j,m)

'''


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

    model.objective1 = Objective(expr=model.makespan, sense=minimize)
    model.objective1.deactivate()

    model.objective2 = Objective(expr=lambda model: sum((model.start[j, m] + model.dur[j, m]) * model.taskPri[j, m]
                                                        for (j, m) in model.TASKS))

    # model.objective2.deactivate()
    model.consDum = Constraint(rule=lambda model: model.start['Task_D', "Curate"] == 10)

    model.finish = Constraint(model.TASKS, rule=lambda model, j, m:
    model.start[j, m] + model.dur[j, m] <= model.makespan)

    model.preceding = Constraint(model.TASKORDER, rule=lambda model, j, m, k, n:
    model.start[k, n] + model.dur[k, n] <= model.start[j, m])

    model.disjunctions = Disjunction(model.DISJUNCTIONS, rule=lambda model, j, k, m:
    [model.start[j, m] + model.dur[j, m] <= model.start[k, m],
     model.start[k, m] + model.dur[k, m] <= model.start[j, m]])

    TransformationFactory('gdp.hull').apply_to(model)
    return model


jobshop_model(TASKS)


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


results = jobshop(TASKS)
results

schedule = pd.DataFrame(results)


def getConsumption(schedule):
    dval = np.random.rand(max(schedule.Finish))

    range(int(schedule.loc[0].Start), int(schedule.loc[0].Finish))
    conLis = []

    for i in range(0, len(schedule)):
        conLis.append(sum(dval[range(int(schedule.loc[i].Start), int(schedule.loc[i].Finish))]))
    schedule['Consumption'] = conLis

    con_by_mac = schedule.groupby('Machine')['Consumption'].sum()

    return con_by_mac


print('\nSchedule by Job')
print(schedule.sort_values(by=['Job', 'Start']).set_index(['Job', 'Machine']))

print('\nSchedule by Machine')
print(schedule.sort_values(by=['Machine', 'Start']).set_index(['Machine', 'Job']))


# %%

def visualize(results):
    schedule = pd.DataFrame(results)
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()

    bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
    text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5 + (len(JOBS) + len(MACHINES)) / 4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), 'Start']
                xf = schedule.loc[(j, m), 'Finish']
                ax[0].plot([xs, xf], [jdx] * 2, c=colors[mdx % 7], **bar_style)
                ax[0].text((xs + xf) / 2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                ax[1].text((xs + xf) / 2, mdx, j, **text_style)

    ax[0].set_title('Task Schedule')
    ax[0].set_ylabel('Task')
    ax[1].set_title('ED Schedule')
    ax[1].set_ylabel('ED')

    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)

    fig.tight_layout()

    return fig


fig = visualize(results)
fig.show()

model = jobshop_model(TASKS)

# %%
data = """
2  44  3   5  5  58  4  97  0   9  7  84  8  77  9  96  1  58  6  89
4  15  7  31  1  87  8  57  0  77  3  85  2  81  5  39  9  73  6  21
9  82  6  22  4  10  3  70  1  49  0  40  8  34  2  48  7  80  5  71
1  91  2  17  7  62  5  75  8  47  4  11  3   7  6  72  9  35  0  55
6  71  1  90  3  75  0  64  2  94  8  15  4  12  7  67  9  20  5  50
7  70  5  93  8  77  2  29  4  58  6  93  3  68  1  57  9   7  0  52
6  87  1  63  4  26  5   6  2  82  3  27  7  56  8  48  9  36  0  95
0  36  5  15  8  41  9  78  3  76  6  84  4  30  7  76  2  36  1   8
5  88  2  81  3  13  6  82  4  54  7  13  8  29  9  40  1  78  0  75
9  88  4  54  6  64  7  32  0  52  2   6  8  54  5  82  3   6  1  26
"""

TASKS = {}
for job, line in enumerate(data.splitlines()[1:]):
    nums = line.split()
    prec = None
    for m, dur in zip(nums[::2], nums[1::2]):
        task = (f"J{job}", f"M{m}")
        TASKS[task] = {'dur': int(dur), 'prec': prec}
        prec = task

pd.DataFrame(TASKS).T
