import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
import scenario_reduction as scen_red


def windPow(ws):
    vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
    if ws < vin or ws > vout:
        return 0
    elif vin <= ws <= vrated:
        return (prated * (ws - vin)) / (vrated - vin)
    else:
        return prated


loadMean = 0.015
loadSD = loadMean * 0.1
n = 1000
timeInd = 2

bSet = set(range(0, 30))
drSet = {3, 5, 9, 11, 27, 29}
rSet = {4, 10, 28}
demandSet = bSet - (drSet | rSet)
dn = len(demandSet)

darr = np.arange(0, n)
vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75
shape, scale = 7.5, 3.5

l, p = dict(), dict()
for j in range(0, (dn + len(rSet)) * timeInd):
    if j % 24 < dn:
        a = norm(loadMean, loadSD).rvs(n)
        darr = np.column_stack((darr, a))
    elif j % 24 >= dn:
        p = np.array(weibull_min(shape).rvs(n) * scale)
        darr = np.column_stack((darr, p))

darr = np.delete(darr, 0, 1)
probs = []
for i in range(0, len(darr)):
    prob = 1
    for j in range(0, (dn + len(rSet)) * timeInd):
        if j % 24 < dn:
            if darr[i, j] > 0:
                prob *= norm.pdf(darr[i, j], loadMean, loadSD)
            elif darr[i, j] <= 0:
                prob *= norm.cdf(0, loadMean, loadSD)
        elif j % 24 >= dn:
            prob *= weibull_min.pdf(darr[i, j] / scale, shape)
        else:
            print("ERROR")
    probs.append(prob)

probabilities = probs / sum(probs)
scenarios = darr.T
S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm',
                               scen0=np.zeros(len(scenarios)))
# use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
S.fast_forward_sel(n_sc_red=6)

# get reduced scenarios
scenarios_reduced = S.scenarios_reduced

# get reduced probabilities
probabilities_reduced = S.probabilities_reduced

# probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

probabilities_reduced.tolist()
s_reduced = scenarios_reduced.T
len(s_reduced)
print(len(s_reduced[0]))


for i in range(0, len(s_reduced)):
    for j in range(0, len(s_reduced[0])):
        if j % 24 >= 21:
            s_reduced[i, j] = windPow(s_reduced[i, j])

s_res = np.reshape(s_reduced, (2, 6, 24))  ## Time - Scenario -  Bus
probabilities
# %%

# darr = np.column_stack((darr, probs / sum(probs)))

probabilities = probs / sum(probs)
scenarios = darr.T
S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm', r=2,
                               scen0=np.zeros(len(scenarios)))
# use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
S.fast_forward_sel(n_sc_red=6)

# get reduced scenarios
scenarios_reduced = S.scenarios_reduced

# get reduced probabilities
probabilities_reduced = S.probabilities_reduced

# probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

probabilities_reduced.tolist()
s_reduced = scenarios_reduced.T
len(s_reduced)
len(s_reduced[0])
np.argwhere(s_reduced[0] > 3)

for i in range(0, len(s_reduced)):
    for j in range(0, len(s_reduced[0])):
        if j % 24 >= 21:
            s_reduced[i, j] = windPow(s_reduced[i, j])

# indices = np.random.choice(np.arange(0, n), 5, p=(probs / sum(probs)).tolist())
# selection = darr[indices, :]
#
# names = []
# for i in range(1, 4):
#     s = "Demand {}".format(i)
#     names.append(s)
# names.insert(0, 'Scenarios')
# names.append('WindPower')
# names.append('Probs')
#
# df = pd.DataFrame(data=selection, columns=names)
# df.Probs = df.Probs / sum(df.Probs)
# df = df.set_index("Scenarios")

# %%
import numpy as np
import scenario_reduction as scen_red

scenarios = np.random.rand(10, 30)  # Create 30 random scenarios of length 10.
len(scenarios)
probabilities = np.random.rand(30)
probabilities = probabilities / np.sum(probabilities)  # Create random probabilities of each scenario and normalize

S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm', r=2, scen0=np.zeros(10))
S.fast_forward_sel(n_sc_red=5,
                   num_threads=4)  # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
scenarios_reduced = S.scenarios_reduced  # get reduced scenarios
len(scenarios_reduced[0])
len(scenarios_reduced)
probabilities_reduced = S.probabilities_reduced  # get reduced probabilities
