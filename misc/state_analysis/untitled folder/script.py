'''updates NN after a certain number of time steps'''

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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT_DIR)
from utilities import *
from agents import Agent
from plotting import *


def simple_reward(X):
    '''Calculates reward based on simple condition on bacteria populations'''
    if all(x > 1 for x in X):
        return 1
    else:
        return -1

def Q_train_step(agent, sess, X, C, t, explore_rate, Q_params, ode_params):
    '''Carries out one instantaneous Q_learning training step
    Parameters:
        X: array storing the populations of each bacteria
        C: array contatining the concentrations of each rate limiting nutrient
        explore_rate: the current explore rate
    Returns:
        X1: new population vector after this timestep
        C1: new nutrient concentration vector after this timestep
        reward: reward recieved at current timestep
    '''
    A, num_species, num_x_states, x_bounds, num_C0_states, C0_bounds, step_size, y, error = Q_params

    state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector
    allQ = np.array(sess.run([agent.Qout], feed_dict= {agent.inputs:state})[0])

    # check network isnt outputting Nan
    assert all(not np.isnan(Q) for Q in allQ[0]) , 'Nan found in output, network probably unstable'

    action = epsilon_greedy(explore_rate, allQ)

    # get new state and reward
    S = np.append(X, C)

    C0 = action_to_state(action, num_species, num_C0_states, C0_bounds) # take out this line to remove the effect of the algorithm

    sol = odeint(sdot, S, [t, t+1], args=(C0,A,ode_params, num_species))[-1]

    X1 = sol[:num_species]
    C1 = sol[num_species:]

    assert len(C0) == num_species, 'C0 is the wrong length: ' + str(len(C0))
    assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
    assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C))

    # turn new state into one hot vector
    state1 = state_to_one_hot(X1, num_species, x_bounds, num_x_states) # flatten to a vector

    reward = simple_reward(X1)

    # get Q values for new state
    Q1 = sess.run(agent.Qout, feed_dict = {agent.inputs: state1})
    maxQ1 = np.max(Q1)
    targetQ = allQ
    targetQ[0, action] += step_size*(reward + y*maxQ1 - allQ[0,action])

    # train network based on target and predicted Q values
    sess.run([agent.updateModel], feed_dict = {agent.inputs: state, agent.nextQ:targetQ})

    return X1, C1, reward


def Q_test_step(agent, sess, X, C, t, Q_params, ode_params):
    A, num_species, num_x_states, x_bounds, num_C0_states, C0_bounds, step_size, y, error = Q_params
    state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector
    action = sess.run([agent.predict], feed_dict= {agent.inputs:state})

    # check network isnt outputting Nan
    assert not np.isnan(action), 'Nan found in output, network probably unstable'

    # get new state and reward
    S = np.append(X, C)
    # turn action index into C0


    C0 = action_to_state(action, num_species, num_C0_states, C0_bounds) # take out this line to remove the effect of the algorithm

    sol = odeint(sdot, S, [t, t+1], args=(C0,A,ode_params, num_species))[-1]
    X1 = sol[:num_species]
    C1 = sol[num_species:]

    # check X and R are of the right dimension
    assert len(C0) == num_species, 'C0 is the wrong length: ' + str(len(C0))
    assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
    assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C))

    reward = simple_reward(X1)
    return X1, C1, reward

def Q_learn():
    matplotlib.rcParams.update({'font.size': 22})
    # SIMULATION CONSTANTS
    NUM_EPISODES = 100
    test_freq = 5

    y = .99
    explore_rate = 1.
    T_MAX = 1000

    num_species = 2

    # growth rates can be varied as the integer values between 0.2 and -0.2 for both species
    C0_bounds = [0.,10.]
    # due to self competition the max pop is about ten anyway
    x_bounds = [0,10]
    # discretise into _ growth rate states per species
    num_C0_states = 10
    # descritise into _ states per species
    num_x_states = 10

    tf.reset_default_graph() #Clear the Tensorflow graph.

    h1_size = 100
    h2_size = 100
    #h3_size = 50
    #h4_size = 100
    #h5_size = 200

    layer_sizes = [num_x_states**num_species, h1_size, h2_size,  num_C0_states**num_species]
    agent = Agent(layer_sizes)
    saver = tf.train.Saver()

    step_size = 0.5

    init = tf.global_variables_initializer()
    A = np.array([[-0.01, -0.06],
                  [-0.01, -0.01]])
    #A = np.array([[-0.01, -0.06],
    #             [-0.01, -0.01]])

    q = 3
    y = 1

    Rmax = np.array([5.,15.])
    Km = np.array([2.,1.])
    gamma = 0.9
    MIN_EXPLORE_RATE = 0.0001
    error = 0.1
    ode_params = (q,y, Rmax, Km)
    Q_params = (A, num_species, num_x_states, x_bounds, num_C0_states, C0_bounds, step_size, y, error)

    NOISE = False

    with tf.Session() as sess:
        sess.run(init)

        test_rewards = []
        train_rewards = []
        test_ts = []
        train_ts = []
        t_total = 0
        train_reward = 0

        for episode in range(NUM_EPISODES):
            #reset
            X = np.array([2.]*num_species)
            C0 = np.array([1.0]*num_species)
            C = np.array([1.0]*num_species)

            xSol = [X]

            # convert state to one hot vector
            state = state_to_one_hot(X, num_species, x_bounds, num_x_states)

            running_reward = 0

            ep_history = np.array([[]])
            t = 0
            explore_rate = get_explore_rate(episode, MIN_EXPLORE_RATE, 10)
            while t < T_MAX:
                X, C, reward = Q_train_step(agent, sess, X, C, t, explore_rate, Q_params, ode_params)

                if NOISE:
                    X = add_noise(X, error)
                    C = add_noise(C, error)

                running_reward += reward

                xSol.append(X)

                if (not all(x>1/1000 for x in X)) or t == T_MAX - 1: # if done
                    break
                t += 1
                #Update our running tally of scores.

            if episode%test_freq == 0:
                train_ts.append(t_total/test_freq)
                print('Episode: ', episode)
                print('P: ' + str(X))
                print('C0: ' + str(C0))
                print('C: ' + str(C))
                print('R: ' + str(monod(C, Rmax, Km)))
                print('Explore rate: ' + str(explore_rate))
                print('Average time: ', t_total/test_freq)
                print('Average reward: ', train_reward/test_freq)

                train_rewards.append(train_reward/test_freq)
                train_reward = running_reward
                t_total = t

                # test_agent
                test_reward = 0

                X = np.array([2]*num_species)
                C0 = np.array([1.0]*num_species)
                C = np.array([1.0]*num_species)

                xSol = [X]
                test_t = 0
                while test_t < T_MAX:

                    X,C,reward = Q_test_step(agent, sess, X, C, t, Q_params, ode_params)
                    # add some random noise

                    if NOISE:
                        X = add_noise(X, error)
                        C = add_noise(C, error)

                    test_reward += reward
                    xSol.append(X)
                    state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector

                    if (not all(x > 1/1000 for x in X)): # if done
                        break
                    test_t += 1

                test_ts.append(test_t)
                test_rewards.append(test_reward)
                xSol = np.array(xSol)
                plt.figure(figsize = (22.0,12.0))
                plot_pops(xSol, '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/test/Qpops_test_' + str(int(episode/test_freq)) + '.png')
                np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/test/Qpops_test_' + str(int(episode/test_freq)) + '.npy', xSol)

                print('TEST:')
                print('Reward: ', test_reward)
                print('Time: ',test_t)
                print()

            else:
                train_reward += running_reward
                t_total += t

        xSol = np.array(xSol)
        save_path = saver.save(sess, "/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_saved_network/trained_network.ckpt")
        print('Trained Q learning model saved in: ', save_path)
        print('DONE')
        print('Episode: ',episode)
        print('Time: ',t_total/test_freq)
        print('Explore rate: ', explore_rate)
        print('C0: ', C0)
        print('Reward: ', running_reward)
        print()

        plt.figure(figsize = (16.0,12.0))
        plot_survival(train_ts,
                      '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/Q_train_survival.png',
                      NUM_EPISODES, T_MAX, 'Training')
        np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/Q_train_survival.npy', train_ts)

        plt.figure(figsize = (16.0,12.0))
        plot_survival(test_ts,
                      '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/Q_test_survival.png',
                      NUM_EPISODES, T_MAX, 'Testing')


        np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/Q_test_survival.npy', test_ts)

        plt.figure(figsize = (22.0,12.0))
        plot_pops(xSol, '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/Qpops.png')
        np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/QPops.npy', xSol)

        plt.figure(figsize = (16.0,12.0))

        plot_rewards(test_rewards,
                     '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/Qtest_rewards.png',
                     NUM_EPISODES, T_MAX, 'Testing')
        np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/Qtest_rewards.npy', test_rewards)

        train_rewards = train_rewards[1:-1]
        plt.figure(figsize = (16.0,12.0))
        plot_rewards(train_rewards,
                     '/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_graphs/Qtrain_rewards.png',
                     NUM_EPISODES,T_MAX, 'Training')
        np.save('/Users/Neythen/masters_project/results/Q_learning_results/WORKING/WORKING_data/Qtrain_rewards.npy', train_rewards)

        return agent, saver

Q_learn()
