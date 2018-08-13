import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))
from utilities import *

A = np.array([[-0.01, -0.06],
            [-0.01, -0.01]])


'''look for steady state between 3.75 and 4.25'''

N1_min = 3.5
N1_max = 4.5

N2_min = 3.5
N2_max = 4.5

C_min = 0
C_max = 10

q = 1.5
y = 1
Rmax = np.array([[7.],[10.]])
Km = np.array([[1.],[1.]])

params = (q,y, Rmax, Km)
def sdot(S, t, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    N = np.array(S[:num_species]).reshape(2,1)

    C = np.array(S[num_species:]).reshape(2,1)
    q, y, Rmax, Km = params
    R = monod(C, Rmax, Km)

    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    return dN


t = 0

for C1 in np.arange(C_min, C_max, 0.1):
    for C2 in np.arange(C_min, C_max, 0.1):
        for N1 in np.arange(N1_min, N1_max, 0.1):
            for N2 in np.arange(N2_min, N2_max, 0.1):
                S = np.array([N1, N2, C1, C2])
                dN = sdot(S, t, A, params, 2)

                if np.isclose(dN, np.array([0,0]), atol = 0.1).all():
                    print(N1, N2, C1, C2)
