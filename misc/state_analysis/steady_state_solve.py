import sympy as sp
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import newton_krylov
from sympy.utilities.lambdify import lambdify, implemented_function



def monod(C0, C, mu_max, Km, K0):
    return mu_max*C0*C/((K0+C0)*(Km+C))


def dN1(N1, N2, C1, C2, C0):
    return N1*(monod(C0, C1, mu1_max, Km, K0)) + a_11*N1 + a_12*N2 - q)

def dN2(N1, N2, C1, C2, C0):
    return N2*(monod(C0, C2, mu2_max, Km, K0)) + a_21*N1 + a_22*N2 - q)

def dC1(N1, N2, C1, C2, C0):
    return q*(C_1i - Ci) - 1/y_11 * monod(C0, C1, mu1_max, Km, K0) * N1


def dC2(N1, N2, C1, C2, C0):
    return q*(C_2i - Ci) - 1/y_22 * monod(C0, C2, mu2_max, Km, K0) * N2

def dC0(N1, N2, C1, C2, C0):
    return q*(C_0i - C0) - 1/y_01 * monod(C0, C1, mu1_max, Km, K0) * N1 - 1/y_02 * monod(C0, C2, mu2_max, Km, K0) * N2


def F(x):
    return [dN1(x[0], x[1], x[2], x[3], x[4]), dN2(x, x[1], x[2], x[3], x[4]), dC1(x, x[1], x[2], x[3], x[4]), dC2(x, x[1], x[2], x[3], x[4]), dC0(x, x[1], x[2], x[3], x[4])]

xin = [0,0,0,0,0]
