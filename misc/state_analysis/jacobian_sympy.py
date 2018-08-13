import sympy as sp
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import newton_krylov
from sympy.utilities.lambdify import lambdify, implemented_function


# define constants
(K_10, K_11, K_20, K_22, mu1_max, mu2_max, a_11, a_12, a_21, a_22, q, y_11, y_22, y_10, y_20) = sp.symbols(
    ['K_10', 'K_11', 'K_20', 'K_22', 'mu1_max', 'mu2_max', 'a_11', 'a_12', 'a_21', 'a_22',
        'q', 'y_11', 'y_22', 'y_10', 'y_20'])


# define state variables
N1 = sp.Symbol('N1')
N2 = sp.Symbol('N2')
C1 = sp.Symbol('C1')
C2 = sp.Symbol('C2')
C0 = sp.Symbol('C0')

x_i = [N1, N2, C1, C2, C0]


# define functions
mu1 = sp.Lambda((C0, C1), mu1_max*C0*C1/((K_10 + C0)*(K_11 + C1)) )
mu2 = sp.Lambda((C0, C2), mu2_max*C0*C2/((K_20 + C0)*(K_22 + C2)) )

#mu1 = sp.Lambda((C0, C1), mu1_max*C1/(K_11 + C1))
#mu2 = sp.Lambda((C0, C2), mu2_max*C2/(K_22 + C2))


func_dict = {
    'mu1': mu1,
    'mu2': mu2
}



# define differential equations
dN1 = sp.sympify('N1*(mu1(C0, C1) + a_11*N1 + a_12*N2 - q)', locals = func_dict)
dN2 = sp.sympify('N2*(mu2(C0, C2) + a_21*N1 + a_22*N2 - q)', locals = func_dict)
dC1 = sp.sympify('q*(C_1i - C1) - 1/y_11 * mu1(C0, C1) * N1', locals = func_dict)
dC2 = sp.sympify('q*(C_2i - C2) - 1/y_22 * mu2(C0, C1) * N2', locals = func_dict)
dC0 = sp.sympify('q*(C_0i - C0) - 1/y_01*mu1(C0, C1)*N1 - 1/y_02*mu2(C0, C2)*N2', locals = func_dict)



equs = [dN1, dN2, dC1, dC2, dC0]

f_i = sp.Matrix(equs)

# define the jacobian
J = f_i.jacobian(x_i)


keeping_alive_dict = {
    'K_10':1.,
    'K_11':2.,
    'K_20':1.,
    'K_22':1.,
    'mu1_max':5.,
    'mu2_max':15.,
    'a_11':-0.01,
    'a_12':-0.06,
    'a_21':-0.01,
    'a_22':-0.01,
    'q': 3,
    'y_11': 2.,
    'y_22': 1.,
    'y_01': 1.,
    'y_02': 1.,

}

finer_control_dict = {
    'K_10':1.,
    'K_11':1.,
    'K_20':1.,
    'K_22':1.,
    'mu1_max':7.,
    'mu2_max':10.,
    'a_11':-0.01,
    'a_12':-0.06,
    'a_21':-0.01,
    'a_22':-0.01,
    'q': 1.5,
    'y_11': 1.,
    'y_22': 1.,
    'y_01': 1.,
    'y_02': 1.,

}

SPOCK_params_dict = {
    'K_10':0.5, # Ks between 0.0001 and 0.5
    'K_11':0.5,
    'K_20':0.5,
    'K_22':0.5,
    'mu1_max':0.8,
    'mu2_max':1.,
    'a_11':-0.01,
    'a_12':-0.06,
    'a_21':-0.01,
    'a_22':-0.01,
    'q': 1.5,
    'y_11': 900., # ys between 1129.17 and 913.33
    'y_22': 900.,
    'y_01': 900.,
    'y_02': 900.,

}

noise_dict = {
    'N1': 1.05879964,
    'N2': 1.94953122,
    'C1': 0.30294612,
    'C2': 0.18064916,
    'C0': 99999999999.,
}


normal_dict = {
    'N1': 5.86086775,
    'N2': 2.98599383,
    'C1': 0.33023409,
    'C2': 0.18884416,
    'C0': 99999999999.,
}


recurrent_dict = {
    'N1': 5.27121812,
    'N2': 5.117429,
    'C1': 0.36180357,
    'C2': 0.19102725,
    'C0': 99999999999.,
}

FC_dict = {
    'N1': 4.37927702,
    'N2': 4.03026668,
    'C1': 0.34243859,
    'C2': 0.1882264 ,
    'C0': 99999999999.,
}



auxotoph_params_dict = {
    'K_10':0.00006845928, # Ks between 0.0001 and 0.5
    'K_11':0.00049,
    'K_20':0.00006845928,
    'K_22':0.00000102115,
    'mu1_max':2.,
    'mu2_max':2.,
    'a_11':-0.1,
    'a_12':-0.11,
    'a_21':-0.1,
    'a_22':-0.1,
    'q': 0.5,
    'y_11': 480000., # ys between 1129.17 and 913.33
    'y_22': 480000.,
    'y_01': 520000.,
    'y_02': 520000.,

}

simple_auxotoph_params_dict = {
    'K_10':0.00006845928, # Ks between 0.0001 and 0.5
    'K_11':0,
    'K_20':0.00006845928,
    'K_22':0.00000102115,
    'mu1_max':1.5,
    'mu2_max':3.,
    'a_11':0,
    'a_12':0,
    'a_21':0,
    'a_22':0,
    'q': 0.5,
    'y_11': 480000., # ys between 1129.17 and 913.33
    'y_22': 480000.,
    'y_01': 520000.,
    'y_02': 520000.,

}

#f = open('/Users/Neythen/masters_project/results/lookup_table_results/use_for_auxotroph_section/steady_state_sim')
f = open('/Users/Neythen/masters_project/results/lookup_table_results/simple_auxotroph/run_ss.sh.o29175')



auxotroph_SSs = []

# read in all steady states
for line in f:
    if line[0] == '[':
        line = line.replace("[", " ")
        line = line.replace("]", " ")
        line = line.split()
        line = [float(l) for l in line]
        N1 = line[0]
        N2 = line[1]
        C1 = line[2]
        C2 = line[3]

    if line[0] == ' ':

        line = line.replace("[", " ")
        line = line.replace("]", " ")
        line = line.split()
        line = [float(l) for l in line]
        C0 = line[0]

        auxotroph_dict = {
            'N1': N1,
            'N2': N2,
            'C1': C1,
            'C2': C2,
            'C0': C0,
        }

        auxotroph_SSs.append(auxotroph_dict)

# make jacobian for this system
stable = 0
unstable = 0
i = 0
for dict in auxotroph_SSs:

    J = f_i.jacobian(x_i)
    J = J.subs(dict)
    J = J.subs(auxotoph_params_dict)

    J = np.array(J).astype(np.float64)
    eig = np.linalg.eigvals(J)
    if all(e.real < 0 for e in eig):
        stable += 1
    else:
        unstable += 1
    if i%100 == 0:
        print(i)
        print(stable)
        print(unstable)
        print('-----------------')
    i += 1
print(stable)
print(unstable)


#print(sp.nsolve(equs, x_i, (0.,0.,0.,0.)))
