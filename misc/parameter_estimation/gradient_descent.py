import numpy as np
from scipy.integrate import odeint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

A0 = np.identity(2)


actual_A = np.array([[-0.03, 0.035],
                      [-0.1, -0.02]])


def lotka_volterra(N, t, R, A):
    return N * (R + np.matmul(A,N))

def GD_step(N, A0, R, alpha, t):

    R += (np.random.rand(2,) - 0.5)
    # if population too low, raise growth rate
    for i, v in enumerate(N):
        if v < 0.2:
            R[i] += 5
        elif v > 10:
            R[i] -= 5


    pred_dN = lotka_volterra(N,0, R, A0)
    actual_dN = lotka_volterra(N,0, R, actual_A)


    # get the first part of the gradient descent step
    deltas = 2*(pred_dN - actual_dN)

    # make full gradient descent step matrix
    dA = np.array([N*N[0]*deltas[0],
                   N*N[1]*deltas[1])
    

    A0 -= alpha * dA

    N = odeint(lotka_volterra, N, [t, t+1], args=(R, actual_A,))[-1]

    return N, A0

def growth_rate(C, Rmax, Km):
    return (Rmax*C)/ (Km + C) # monod equation


N = np.array([1.0,2.0, 3.0])
alpha = 0.0000001
R = np.array([1.0,1.0, 1.0])
Rmax = np.array([25.0, 10.0])
Km = np.array([1.0,1.0])



for t in range(10000):
    R = growth_rate(N, Rmax, Km) # this keeps the growth rate from blowing up but is nonsense
    N, A0 = GD_step(N,A0, R,alpha, t)

print('A: ', A0)
'''
If the system is kept stable, with all species alive and not at steady state gradient descent can be used to estimate the LV
interaction matrix
'''
