import numpy as np
from scipy.integrate import odeint
import os
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


from utilities import *
from agents import *

def GD_step(N, A0, R, alpha, t, v_t):

    pred_dN = lotka_volterra(N,0, R, A0)
    actual_dN = lotka_volterra(N,0, R, actual_A) # replace this with dN estimate from timesteps

    # get the first part of the gradient descent step
    deltas = 2*(pred_dN - actual_dN)

    # make full gradient descent step matrix
    dA = np.array([N*N[0]*deltas[0], N*N[1]*deltas[1]])

    '''
    # backtracking line search
    alpha = 1
    c = 0.01
    p = 0.1
    while (J(N, R, actual_A, A0 - alpha*deltas) >  J(N, R, actual_A, A0) - c*alpha*dA).any():
        alpha = p*alpha
    A0 -= alpha * dA


    # other line search
    y = 0.0005
    n = 0.0005

    v_t = y*v_t + n * delJ(N, R, actual_A, A0 - y*v_t)
    A0 = A0 - v_t
    '''
    alpha = 0.0000001

    A0 -= alpha*dA

    return A0

def sdot(S, t, C0, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    N = np.array(S[:num_species])
    C = np.array(S[num_species:])
    q, y, Rmax, Km = params
    R = monod(C, Rmax, Km)
    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(C0 - C) - (1/y)*R*N # sometimes dC.shape is (2,2)

    if dC.shape == (2,2):
        print(q,C0.shape,C0,C,y,R,N)
    return tuple(np.append(dN, dC))


# parameters
q = 3
y = 1
Rmax = np.array([5.0,10.0])
Km = np.array([1.0,1.0])
params = (q,y, Rmax, Km)

A0 = np.zeros((2,2))


actual_A = np.array([[-0.01, -0.06],
                      [-0.01, -0.01]])

alpha = 0.000001
num_species = 2

# growth rates can be varied as the integer values between 0.2 and -0.2 for both species
C0_bounds = [0.,10.]
# due to self competition the max pop is about ten anyway
x_bounds = [0,10]
# discretise into _ growth rate states per species
num_C0_states = 10
# descritise into _ states per species
num_x_states = 10
# create network and buffer
n_units = 20 # the output of GRU will be (time_steps, n_units)
h2_size = 200 # needs to be time_steps*n_untis
h3_size = 20
layer_sizes = [num_x_states**num_species, n_units, h2_size, h3_size,  num_C0_states**num_species]
buffer_size = 10
agent = RecurrentAgent(layer_sizes, buffer_size)

saver = tf.train.Saver()

X = np.array([2.0]*num_species)
C0 = np.array([1.0]*num_species)
C = np.array([1.0]*num_species)
R = monod(C,Rmax, Km)

xSol = [X]

buffer = Buffer(10, num_x_states**2)
state = state_to_one_hot(X, num_species, x_bounds, num_x_states)
buffer.add(state)


with tf.Session() as sess:
    saver.restore(sess,"/Users/Neythen/masters_project/project_code/A_estimation/saved/trained_network.ckpt")
    y = 0.05
    n = 0.05
    v_t = 0.00001 * delJ(X,R,actual_A, A0)
    for t in range(100000):
        # do GD step
        A0 = GD_step(X, A0, R, alpha, t, v_t)
        action, allQ = sess.run([agent.predict, agent.Qout], feed_dict= {agent.inputs:buffer.get_buffer()})
        action = np.random.randint(num_C0_states**2)


        # get new state and reward
        S = np.append(X, C)
        # turn action index into C0
        C0 = action_to_state(action, num_species, num_C0_states, C0_bounds) # take out this line to remove the effect of the algorithm

        sol = odeint(sdot, S, [t, t+1], args=(C0,actual_A,params, num_species))[-1]

        X1 = sol[:num_species]
        C1 = sol[num_species:]

        xSol.append(X1)
        state1 = state_to_one_hot(X1, num_species, x_bounds, num_x_states)
        buffer.add(state1)

        X = X1
        C = C1

        if t % 100 == 0:
            print('t: ', t)
            print(A0)
            print('N: ', X)
            print('C0: ', C0)
