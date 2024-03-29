from numpy import random
from collections import Counter
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

a = 0
b = 7
y0 = 2

n = 50
X = []
Y = []
r = sts.uniform()
xi = r.rvs(size=n)

for i in range(n):
    x = xi[i] * (b - a) + a
    X.append(x)
    y = math.sqrt(pow(x, 3))
    Y.append(y)

Y.sort()

Fn = []
F = []
squared_deviation = []

for i in range(1, n + 1):
    Fn.append((i - 0.5) / n)
    F.append(pow(Y[i - 1], 2/3) / 8)
    squared_deviation.append((Fn[i - 1] - F[i - 1]) ** 2)

table = pd.DataFrame(data={"$y_i$": Y, "$F_n$": Fn, "$F$": F, "$\delta$": squared_deviation})
print(table)
print(1. / (12 * n) + sum(squared_deviation))

