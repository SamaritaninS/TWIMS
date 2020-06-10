from numpy import random
from collections import Counter
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as sts
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


def equal_probability(t, M, m):
    t.sort()
    A = np.zeros(M)
    B = np.zeros(M)
    A[0] = t[0]
    B[-1] = t[-1]
    for i in range(1, M):
        A[i] = (t[m * i] + t[m * i + 1]) / 2
        B[i - 1] = A[i]
    delta = []
    for i in range(len(A)):
        delta.append(B[i] - A[i])
    f_x = []
    x = []
    s = A[0]
    for i in delta:
        f_x.append(1. / (M * i))
        x.append(s)
        s += i
    return (x, f_x, A, B, delta)


a = 0
b = 7
y0 = 2

n = int(input())
X = []
Y = []
r = sts.uniform()
xi = r.rvs(size=n)

for i in range(n):
    x = xi[i] * (b - a) + a
    X.append(x)
    y = math.sqrt(pow(x, 3))
    Y.append(y)

print(Y)

if n <= 100:
    M = int(np.sqrt(n))
else:
    M = int(4 * np.log10(n))

m = n // M

x, f_x, A, B, delta = equal_probability(Y, M, m)

table = pd.DataFrame(data={"$x_i$": x, "$A_i$": A,
                           "$B_i$": B, "$delta_i$": delta,
                           "$v_i$": [m] * len(x), "$f_i$": f_x})
print(table)

x_theor = np.linspace(0, 7, n)
f_y = []

for xi in x_theor:
    f_y.append(pow(xi, -1/3) / 12)

plt.fill_between(x, y1=f_x, y2=[0] * len(f_x), color='b', step='post', alpha=0.5, label="Histogram")
plt.plot(x_theor, f_y, label='Theoretical distribution density', c='r')
plt.legend()
plt.xlabel("y")
plt.ylabel("f(Y)")
plt.show()

p_teor = []
pi = [m / n] * M
F_in_B = []
F_in_A = []
for i in range(len(A)):
    F_Ai = (pow(A[i], 2/3) / 8)
    F_Bi = (pow(B[i], 2/3) / 8)
    F_in_A.append(F_Ai)
    F_in_B.append(F_Bi)
    p_teor.append(F_Bi - F_Ai)

xi = []
for i in range(M):
    xi.append(n * (p_teor[i] - pi[i]) ** 2 / p_teor[i])

print("xi = ", sum(xi))

table2 = pd.DataFrame(data={"$F(A_i)$": F_in_A, "$F(B_i)$": F_in_B, "$p_i$": p_teor, "$p_i^*$": pi, "$\chi_i$": xi})
print(table2)
k = M - 1
print("k = ", k)
