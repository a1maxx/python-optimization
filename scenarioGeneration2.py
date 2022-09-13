import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
import scenario_reduction as scen_red

loadMean = 0.015
loadSD = loadMean * 0.1
n = 1000

bSet = set(range(0, 30))
drSet = {3, 5, 9, 11, 27, 29}
rSet = {4, 10, 28}
demandSet = bSet - (drSet | rSet)
dn = len(demandSet)

darr = np.arange(0, n)

for i in range(1, dn + 1):
    a = norm(loadMean, loadSD).rvs(n)
    darr = np.column_stack((darr, a))

vin, vout, vrated, prated = 3.5, 20, 14.5, 0.75

shape, scale = 7.5, 3.5
l, p = dict(), dict()
for i in range(0, len(rSet)):
    l[i] = weibull_min(shape).rvs(n) * scale
p = np.zeros(shape=(n, len(rSet)))

for i in l:
    for j in range(0, n):
        if l[i][j] < vin or l[i][j] > vout:
            p[j, i] = 0
        elif vin <= l[i][j] <= vrated:
            p[j, i] = (prated * (l[i][j] - vin)) / (vrated - vin)
        else:
            p[j, i] = prated

darr[darr < 0] = 0

darr = np.column_stack((darr, p))

probs = []

for i in range(0, len(darr)):
    prob = 1
    for j in range(1, len(darr[1])):
        if j < dn + 1:
            if darr[i, j] > 0:
                prob *= norm.pdf(darr[i, j], loadMean, loadSD)
            elif darr[i, j] == 0:
                prob *= norm.cdf(0, loadMean, loadSD)
        elif j >= dn + 1:
            prob *= weibull_min.pdf(l[j-dn-1][i] / scale, shape)
        else:
            print("ERRoR")
    probs.append(prob)

#%%

# darr = np.column_stack((darr, probs / sum(probs)))



darrwoScen = np.delete(darr, 0, 1)
probabilities = probs / sum(probs)
scenarios = darrwoScen.T
S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='2norm', scen0=np.zeros(len(scenarios)))
# use fast forward selection algorithm to reduce to 5 scenarios with 4 threads
S.fast_forward_sel(n_sc_red=6, num_threads=4)
# get reduced scenarios
scenarios_reduced = S.scenarios_reduced
# get reduced probabilities
probabilities_reduced = S.probabilities_reduced

probabilities_reduced = probabilities_reduced / sum(probabilities_reduced)

probabilities_reduced.tolist()
scenarios_reduced.T

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
