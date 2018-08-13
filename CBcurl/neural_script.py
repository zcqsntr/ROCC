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

from utilities import *
from neural_agent import *
from plot_funcs import *

def neural_Q_learn(param_dict, save_path, debug = False, reward_func = False):
    matplotlib.rcParams.update({'font.size': 22})
    if debug: print('NEURAL')

    validate_param_dict(param_dict)
    param_dict = convert_to_numpy(param_dict)

    #extract parameters
    NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE, cutoff, hidden_layers, buffer_size = param_dict['train_params']
    NOISE, error = param_dict['noise_params']
    num_species, num_controlled_species, num_x_states, num_Cin_states = param_dict['Q_params'][1], param_dict['Q_params'][2],  param_dict['Q_params'][3],param_dict['Q_params'][5]
    ode_params = param_dict['ode_params']
    Q_params = param_dict['Q_params'][0:8]
    initial_X = param_dict['Q_params'][8]
    initial_C = param_dict['Q_params'][9]
    initial_C0 = param_dict['Q_params'][10]


    tf.reset_default_graph() #Clear the Tensorflow graph.

    #initialise agent, saver and tensorflow graph
    layer_sizes = [num_x_states**num_species] + hidden_layers + [num_Cin_states**num_controlled_species]
    agent = NeuralAgent(layer_sizes, buffer_size, True, reward_func)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    # make directories to store results
    os.makedirs(os.path.join(save_path,'WORKING_data','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'WORKING_graphs','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'WORKING_saved_network'), exist_ok = True)



    with tf.Session() as sess:
        sess.run(init)

        #initialise results tracking
        visited_states = np.zeros((1,num_x_states**num_species))
        rewards_avs, time_avs, reward_sds, time_sds = [], [], [], []
        episode_ts, episode_rewards = [], []

        # fill buffef with experiences based on random actions
        while len(agent.experience_buffer.buffer) < buffer_size:
            #reset
            X = initial_X
            C = initial_C
            C0 = initial_C0

            for t in range(T_MAX):
                X, C, C0 = agent.pre_train_step(sess, X, C, C0, t, Q_params, ode_params)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                    break

        nIters = 0 # keep track for updating target network
        for episode in range(1,NUM_EPISODES+1):

            # reset for this episode
            X = initial_X
            C = initial_C
            C0 = initial_C0
            xSol = np.array([X])
            running_reward = 0
            ep_history = np.array([[]])
            explore_rate = get_explore_rate(episode, MIN_EXPLORE_RATE, explore_denom)
            step_size = get_learning_rate(episode, MIN_STEP_SIZE, MAX_STEP_SIZE, step_denom)

            # run episode
            for t in range(T_MAX):
                nIters += 1 # for target Q update
                X, C, C0, xSol_next, reward, allQ, visited_states = agent.train_step(sess, X, C, C0, t, visited_states, explore_rate, step_size, Q_params, ode_params,nIters)

                if NOISE:
                    X = add_noise(X, error)

                running_reward += reward

                xSol = np.append(xSol, xSol_next, axis = 0)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                    break

            # track results
            if episode%test_freq == 0 and episode != 0:

                if debug:
                    print('Episode: ', episode)
                    print('Explore rate: ' + str(explore_rate))
                    print('Step size', step_size)
                    print('Average Time steps: ',np.mean(episode_ts))
                    print('Average Reward: ', np.mean(episode_rewards))
                    print()

                # add current results
                episode_rewards.append(running_reward)
                episode_ts.append(t)

                time_sds.append(np.std(episode_ts))
                time_avs.append(np.mean(episode_ts))

                rewards_avs.append(np.mean(episode_rewards))
                reward_sds.append(np.std(episode_rewards))

                # reset
                train_reward = running_reward
                episode_ts = []
                episode_rewards = []

                if debug:
                    # plot current population curves
                    plt.figure(figsize = (22.0,12.0))
                    plot_pops(xSol, os.path.join(save_path,'WORKING_graphs','train','Qpops_train_' + str(int(episode/test_freq)) + '.png'))
                    np.save(os.path.join(save_path,'WORKING_data','train','Qpops_train_' + str(int(episode/test_freq)) + '.npy'), xSol)
            else:
                episode_rewards.append(running_reward)
                episode_ts.append(t)

        xSol = np.array(xSol)
        network_save_path = saver.save(sess, os.path.join(save_path,'WORKING_saved_network','trained_network.ckpt'))

        # create and save state action plot
        visited_states = visited_states.reshape([num_x_states]*num_species)
        Q_actions = np.zeros((num_x_states**num_species))
        for i in range(num_x_states**num_species):
            if visited_states[i,j] == 0:
                Q_actions[i,j] = - 1
            else:
                one_hot_state = np.zeros((1,num_x_states**num_species))
                one_hot_state[0,i] = 1
                allQ = np.array(sess.run(agent.predQ, feed_dict = {agent.inputs:one_hot_state}))
                Q_actions[i] = np.argmax(allQ)
        np.save(os.path.join(save_path,'state_action.npy'), Q_actions)

        # plot results
        plt.figure(figsize = (16.0,12.0))
        plot_survival(time_avs,
                      os.path.join(save_path,'WORKING_graphs','Q_train_survival.png'),
                      NUM_EPISODES, T_MAX, 'Training')
        np.save(os.path.join(save_path,'WORKING_data','Q_train_survival.npy'), time_avs)

        plt.figure(figsize = (22.0,12.0))
        plot_pops(xSol, os.path.join(save_path,'WORKING_graphs','Qpops.png'))
        np.save(os.path.join(save_path,'WORKING_data','QPops.npy'), xSol)


        plt.figure(figsize = (16.0,12.0))
        plot_rewards(rewards_avs,
                     os.path.join(save_path,'WORKING_graphs','Qtrain_rewards.png'),
                     NUM_EPISODES,T_MAX, 'Training')

        # save results
        np.save(os.path.join(save_path,'WORKING_data','Qtrain_rewards.npy'), rewards_avs)
        np.save(os.path.join(save_path,'visited_states.npy'), visited_states)
        np.save(os.path.join(save_path,'reward_sds.npy'), reward_sds)
        np.save(os.path.join(save_path,'time_sds.npy'), time_sds)

        print(np.rot90(Q_actions))

        return agent, saver

if __name__ == '__main__': # for the server


    three_species_parameters = {
        'ode_params': [1., 0.5, [480000., 480000., 480000.], [520000., 520000., 520000.], [0.6, 0.6, 0.6], [0.00048776, 0.00000102115, 0.00000102115], [0.00006845928, 0.00006845928,  0.00006845928]],
        'Q_params': [[[0, -0.00005, -0.00005],[-0.00005, 0, -0.00005], [-0.00005,-0.00005,0]], 3,3, 10, [0.,1000.], 2, [0., 0.1], 0.9, [200., 200., 200.], [0.05, 0.05, 0.05], 1.],
        'train_params': [50000, 100, 970, 1000, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.1]
    }

    double_auxotroph_params = {
        'ode_params': [1., 0.5, [480000., 480000.], [520000., 520000.], [2., 2.], [0.00048776, 0.00000102115], [0.00006845928, 0.00006845928]],
        'Q_params': [[[-0.1, -0.11],[-0.1, -0.1]], 2,2, 10, [0.,20.], 2, [0., 0.1], 0.9, [10.,10.], [0.1,0.1], 1.],
        'train_params': [10000, 100, 970, 1000, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.1]
    }

    smaller_target_params = {
        'ode_params': [1., 0.5, [480000., 480000.], [520000., 520000.], [0.6, 0.6], [0.00048776, 0.00048776], [0.00006845928, 0.00006845928]],
        'Q_params': [[[-0.0001, -0.0001],[-0.0001, -0.0001]], 2,2, 10, [0.,1000.], 2, [0., 0.1], 0.9, [250.,550.], [0.05,0.05], 1.],
        'train_params': [10000, 100, 950, 1000, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.05]
    }
    single_auxotroph =  {
        'ode_params': [0.25, 0.5, [480000., 480000.], [520000., 520000.], [1.5, 3.], [0., 0.00000102115], [0.00006845928, 0.00006845928]],
        'Q_params': [[[0, 0],[0, 0]], 2,1, 10, [0.,120000.], 2, [0., 0.3], 0.9, [50000.,50000.], [1.,0.1], 0.25],
        'train_params': [10, 1, 95, 100, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.1]
    }

    #validate_param_dict(double_auxotroph_params)

    try:
        directory = sys.argv[1]
        repeat_n = sys.argv[2]
        save_path = '/home/zcqsntr/Scratch/neural/' + str(directory) + '/repeat' + str(repeat_n)
    except:
        #save_path = '/home/zcqsntr/Scratch/neural/WORKING/'
        save_path = os.path.join('/Users','Neythen','masters_project','results','Q_learning_results','WORKING')



    neural_Q_learn(single_auxotroph, save_path, debug = True)
