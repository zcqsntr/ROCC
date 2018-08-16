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

# open parameter file
f = open('no_alg_no_C0.yaml')
param_dict = yaml.load(f)
f.close()

param_dict = convert_to_numpy(param_dict) # convert parameters to numpy arrays


# extract parameters
NUM_EPISODES, test_freq, explore_denom, step_denom, MIN_TEMP, MAX_TEMP, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE = param_dict['train_params']
NOISE, error = param_dict['noise_params']
matplotlib.rcParams.update({'font.size': 22})
ode_params = param_dict['ode_params']
Q_params = param_dict['Q_params'][0:7]
A, num_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params

tSol = np.linspace(0, T_MAX, T_MAX+1)


'''
n_coexistant = 0
for N1 in range(1, 10, 1):
    print(N1)
    for N2 in range(1, 10, 1):
        for action in range(100):
            initial_X = np.array([N1, N2])
            initial_C = np.array(param_dict['Q_params'][8])

            sol0 = np.append(initial_X, initial_C)

            Cin = action_to_state(action, 2, 10, [0, 10])

            xSol = odeint(sdot, sol0, tSol, args=(Cin, A,ode_params, num_species))

            if xSol[-1][0] > 1/1000 and xSol[-1][1] > 1/1000:
                n_coexistant += 1


print('n: ', n_coexistant)
'''

# set initial conditions
initial_X = np.array([5., 5.])
initial_C = np.array(param_dict['Q_params'][8])
X = np.append(initial_X, initial_C)
xSol = np.array([X])
Cin = np.array([10.,0.])

time_diff = 4
for t in range(T_MAX):
    Cin = np.random.randint(0,2, size = (1,2)) * 10 # chose a random C0

    # solve
    sol = odeint(sdot, X, [t + x *1 for x in range(time_diff)], args=(Cin, A,ode_params, num_species))[1:-1]
    X = sol[-1,:]
    xSol = np.append(xSol,sol, axis = 0)

    if (X[0] < 1/1000) or (X[1] < 1/1000):
        break

    if t == T_MAX -1:
        print('coexistanc')


# plot
plt.figure(figsize = (16.0,12.0))
labels = ['N1', 'N2', 'C1', 'C2', 'C0']

xSol = np.array(xSol)
for i in range(4):
    plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = labels[i])
plt.legend()
plt.show()
