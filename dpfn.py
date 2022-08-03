import math
from scipy.stats import norm


def stateFunction(s_tp, eps, t):
    if t == 0:
        return 0;
    else:
        return s_tp / 2 + 25 * s_tp / (1 + s_tp ^ 2) + 8 * math.cos(1.2 * t) + norm(0, 0.1).rvs(1)
