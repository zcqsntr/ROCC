import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import os
import yaml
import matplotlib
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))
from utilities import *


f = open('no_alg_with_C0.yaml')
param_dict = yaml.load(f)
f.close()

T_MAX = 1000

NUM_EPISODES, test_freq, explore_denom, step_denom, MIN_TEMP, MAX_TEMP, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE = param_dict['train_params']
NOISE, error = param_dict['noise_params']

matplotlib.rcParams.update({'font.size': 22})


# convert to numpy arrays
param_dict['Q_params'][1] = np.array(param_dict['Q_params'][1])

ode_params = param_dict['ode_params']
ode_params[2] = np.array(ode_params[2])
Q_params = param_dict['Q_params'][0:7]

tSol = np.linspace(0, T_MAX, T_MAX+1)

A, num_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params
count = 0
'''
for N1 in range(2,20,2):
    print(N1)
    for N2 in range(2,20,2):
        for action in range(100):
            initial_X = np.array([N1, N2])
            initial_C = np.array(param_dict['Q_params'][8])
            initial_C0 = param_dict['Q_params'][9]
            X = np.append(initial_X, initial_C)
            X = np.append(X, initial_C0)
            xSol = [X]
            Cin = action_to_state(action, 2, 10, [0, 1])
            for t in range(T_MAX):
                X = odeint(sdot2, X, [t, t+1], args=(Cin, A,ode_params, num_species))[-1]
                xSol.append(X)

                if (X[0] < 1/1000) or (X[1] < 1/1000):

                    break

                if t == T_MAX -1:
                    print('c')
                    count +=1



print('number: ', count)
'''




initial_X = np.array(param_dict['Q_params'][7])
initial_C = np.array(param_dict['Q_params'][8])
initial_C0 = param_dict['Q_params'][9]
X = np.append(initial_X, initial_C)
X = np.append(X, initial_C0)
xSol = np.array([X])
Cin = np.array([0.1,0.1,0.1])


time_diff = 3
for t in range(T_MAX):

    if t % time_diff == 0:
        Cin = np.random.randint(0,2, size = (1,3)) * 0.1
        print(Cin)

    sol = odeint(sdot2, X, [t + x *1 for x in range(time_diff)], args=(Cin, A,ode_params, num_species))[1:]

    X = sol[-1,:]

    xSol = np.append(xSol,sol, axis = 0)

    if (X[0] < 1/1000) or (X[1] < 1/1000):
        break

    if t == T_MAX -1:
        print('coexistanc')
print(sol)


plt.figure(figsize = (16.0,12.0))
xSol = np.array(xSol)
labels = ['N1', 'N2', 'C1', 'C2', 'C0']
for i in range(3):
    plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = labels[i])
plt.ylim([0, 1000])
plt.legend()
plt.show()
