import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint
from numpy import unravel_index
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))


from utilities import *
from agents import *
from plot_funcs import *


A = np.array([[-0.01, -0.06],
              [-0.01, -0.01]])
#A = np.array([[-0.01, -0.06],
#             [-0.01, -0.01]])

q = 1.5
y = 1

num_species = 2


C0_bounds = [0.,10.]
# due to self competition the max pop is about ten anyway
x_bounds = [0,20]
# discretise into _ growth rate states per species
num_C0_states = 2
# descritise into _ states per species
num_x_states = 10
Rmax = np.array([7.,10.])
Km = np.array([1.,1.])
gamma = 0.9
MIN_EXPLORE_RATE = 0
error = 0.1
step_size = 0.5
ode_params = (q,y, Rmax, Km)
Q_params = (A, num_species, num_x_states, x_bounds, num_C0_states, C0_bounds, step_size, y, error)

h1_size = 50
h2_size = 50
h3_size = 50
h4_size = 50
#h5_size = 200

layer_sizes = [num_x_states**num_species, h1_size, h2_size,  num_C0_states**num_species]
agent = NeuralAgent(layer_sizes)

X = np.array([4.]*num_species)
C0 = np.array([1.0]*num_species)
C = np.array([1.0]*num_species)


def simple_reward(X):
    '''Calculates reward based on simple condition on bacteria populations'''
    if all(x > 1 for x in X):
        return 1
    else:
        return -1
def recurrent_simple_reward(xSol):
    X = xSol[-1]

    if all(x > 1 for x in X):
        return 1
    else:
        return -1
saver = tf.train.Saver()
Q_actions = np.zeros((num_x_states**2))
xSol = [X]

T_MAX = 100

'''neural agent'''
'''
with tf.Session() as sess:
    saver.restore(sess, "/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_saved_network/trained_network.ckpt")


    for i in range(num_x_states**2):
        one_hot_state = np.zeros((1,num_x_states**2))
        one_hot_state[0,i] = 1
        allQ = np.array(sess.run([agent.Qout], feed_dict = {agent.inputs:one_hot_state})[0])

        Q_actions[i] = np.argmax(allQ)


'''
'''lookup table'''

LT_actions = np.zeros((num_x_states, num_x_states))
lookuptable = np.load("/Users/Neythen/masters_project/results/lookup_table_results/smaller_target_fixed/WORKING/Q_table.npy")
print(lookuptable.shape)
for i in range(num_x_states):
    for j in range(num_x_states):
        LT_actions[i,j] = np.argmax(lookuptable[i,j])
        if np.count_nonzero(lookuptable[i,j]) == 0:
            LT_actions[i,j] = - 1
        if LT_actions[i,j] == 2:
            LT_actions[i,j] = 1
        elif LT_actions[i,j] == 1:
            LT_actions[i,j] = 2




'''recurrent agent'''

'''
tf.reset_default_graph() #Clear the Tensorflow graph.
# create network and buffer
n_units = 20 # the output of GRU will be (time_steps, n_units)
h2_size = 200 # needs to be time_steps*n_untis
h3_size = 100
buffer_size = 10
layer_sizes = [num_x_states**num_species, n_units, h2_size, num_C0_states**num_species]
recurrent_agent = RecurrentAgent(layer_sizes, buffer_size)
saver = tf.train.Saver()
recurrent_Q_actions = np.zeros((num_x_states**2))





SS = np.array([2.2677554,9.3739891]) # keeping_species_alive
one_hot_SS = state_to_one_hot(SS, num_species, x_bounds, num_x_states)
with tf.Session() as sess1:
    saver.restore(sess1, "/Users/Neythen/masters_project/project_code/A_estimation/saved/trained_network.ckpt")
    for i in range(num_x_states**2):
        one_hot_state = np.zeros((1,num_x_states**2))
        one_hot_state[0,i] = 1
        buffer = Buffer(buffer_size, num_x_states**2) # create fresh buffer
        # add steady state into buffer
        for _ in range(buffer_size):
            buffer.add(one_hot_SS)
        buffer.add(one_hot_state)
        action = sess1.run([recurrent_agent.predict], feed_dict = {recurrent_agent.inputs:buffer.get_buffer()})
        recurrent_Q_actions[i] = action[0][0]


    for t in range(T_MAX):
        X,C,reward = recurrent_test_step(recurrent_agent, sess1, X,C,t,Q_params, ode_params,xSol)
        xSol.append(X)

        state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector

'''
#print(lookuptable)
#print(LT_actions)
#print(LT_actions[0,1])


print(np.rot90(LT_actions))




'''
Q_actions = Q_actions.reshape(num_x_states, num_x_states)

print(np.rot90((visited_states > 0).reshape(num_x_states, num_x_states)))
print(Q_actions)
print()
'''

#print(np.rot90(Q_actions))

visited_states = np.load('/Users/Neythen/masters_project/results/Q_learning_results/smaller_target/WORKING/repeat0/visited_states.npy')
Q_actions = np.load('/Users/Neythen/masters_project/results/Q_learning_results/smaller_target/WORKING/repeat0/state_action.npy')
Q_actions = Q_actions.reshape((num_x_states, num_x_states))
visited_states = visited_states.reshape((num_x_states, num_x_states))


for i in range(num_x_states):
    for j in range(num_x_states):
        if not visited_states[i,j]:
            Q_actions[i,j] = -1
        if Q_actions[i,j] == 2:
            Q_actions[i,j] = 1
        elif Q_actions[i,j] == 1:
            Q_actions[i,j] = 2
print(Q_actions.shape)
print(np.rot90(visited_states.reshape(num_x_states, num_x_states)))

print(np.rot90(Q_actions.reshape(num_x_states, num_x_states)))

#print(recurrent_Q_actions)
