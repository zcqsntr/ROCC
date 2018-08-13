import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from utilities import *

import yaml
f = open('no_alg_with_C0.yaml')
param_dict = yaml.load(f)
f.close()

T_MAX = 1000

NUM_EPISODES, test_freq, explore_denom, step_denom, MIN_TEMP, MAX_TEMP, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE = param_dict['train_params']
NOISE, error = param_dict['noise_params']


# convert to numpy arrays
param_dict['Q_params'][1] = np.array(param_dict['Q_params'][1])

ode_params = param_dict['ode_params']
ode_params[2] = np.array(ode_params[2])
ode_params[1] = np.array(ode_params[1])
Q_params = param_dict['Q_params'][0:7]

tSol = np.linspace(0, T_MAX, T_MAX+1)

A, num_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params
count = 0

log = open('log.txt', 'w')

log.write('started \n')
for i in range(10**6):


    initial_X = np.random.rand(1,2) * 20
    initial_C = Cin = np.random.rand(1,2)
    initial_C0 = C0in = 1.

    X = np.append(initial_X, initial_C)
    X = np.append(X, initial_C0)
    xSol = [X]


    for t in range(T_MAX):
        X = odeint(sdot, X, [t, t+1], args=(Cin, A,ode_params, num_species))[-1]
        xSol.append(X)


        if (X[0] < 1/1000) or (X[1] < 1/1000):
            break

        if t == T_MAX -1:
            xSol = np.array(xSol)

            '''
            plt.figure()
            for s in range(2):
                plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,s])
            plt.savefig('ss')
            '''
            if np.allclose(xSol[-300:, 0], xSol[-1:,0], atol = 1/1000) and np.allclose(xSol[-300:, 1], xSol[-1:,1], atol = 1/1000):
                count +=1
                log.write('initial X: ' + str(initial_X))
                log.write('\n intiial_C: ' + str(initial_C))
                log.write('\n final_X: ' + str(X))
                log.write('\n count: ' + str(count))
                log.write('\n')

                print('X: ', initial_X)
                print('C: ', initial_C)
                print(X)
                print('count: ' + str(count))
                '''
                plt.figure()
                for s in range(2):
                    plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,s])
                plt.savefig('ss' + str(t))

                '''
    if i%100 == 0:
        print(i)
        '''
        print(X)
        plt.figure()
        xSol = np.array(xSol)
        for s in range(2):
            plt.plot(np.linspace(0,T_MAX,len(xSol[:,0])), xSol[:,s])
        plt.savefig('ss' + str(i))
        '''

log.close()




print('Steady states found: ' + str(count))
