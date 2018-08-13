import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random
from scipy.integrate import odeint
from numpy import unravel_index
import os
import sys

from utilities import *
from plot_funcs import *
from lookuptable_agent import *


def lookuptable_Q_learn(param_dict, save_path, debug = False, reward_func = False):
    matplotlib.rcParams.update({'font.size': 22})

    if debug: print('LOOKUPTABLE')

    param_dict = convert_to_numpy(param_dict) # convert parameters to numpy arrays

    # extract parameters
    NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, MAX_STEP_SIZE, MIN_EXPLORE_RATE, cutoff, _, _  = param_dict['train_params']
    NOISE, error = param_dict['noise_params']
    num_species, num_controlled_species, num_x_states, num_Cin_states = param_dict['Q_params'][1], param_dict['Q_params'][2],  param_dict['Q_params'][3],param_dict['Q_params'][5]
    ode_params = param_dict['ode_params']
    Q_params = param_dict['Q_params'][0:8]
    initial_X = param_dict['Q_params'][8]
    initial_C = param_dict['Q_params'][9]
    initial_C0 = param_dict['Q_params'][10]

    #initialise results tracking
    visited_states = np.zeros((1,num_x_states**num_species))
    test_rewards, rewards_avs, test_ts, time_avs, reward_sds, time_sds = [], [], [], [], [], []
    episode_ts, episode_rewards = [], []

    agent = LookupTableAgent(num_x_states, num_Cin_states, num_species, num_controlled_species, reward_func)

    # make directories to store results
    os.makedirs(os.path.join(save_path, 'WORKING_data', 'train'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'WORKING_graphs', 'train'), exist_ok = True)
    os.makedirs(os.path.join(save_path ,'WORKING_saved_Q_table', 'train'), exist_ok = True)

    for episode in range(1,NUM_EPISODES + 1):
        # reset for this episode
        X = initial_X
        C = initial_C
        C0 = initial_C0
        xSol = np.array([X])
        running_reward = 0

        done = False

        explore_rate = get_explore_rate(episode, MIN_EXPLORE_RATE, explore_denom)
        step_size = get_learning_rate(episode, MIN_STEP_SIZE, MAX_STEP_SIZE, step_denom)

        #run episode
        for t in range(T_MAX):
            X, C, C0, xSol_next, reward = agent.train_step(X, C, C0, t, explore_rate, step_size, Q_params, ode_params)

            if NOISE:
                X = add_noise(X, error)

            xSol = np.append(xSol, xSol_next, axis = 0)
            running_reward += reward

            if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                break

        # track results
        if episode%test_freq == 0 and episode != 0:

            #print results
            if debug:
                print('Episode: ',episode)
                print('Average Time steps: ',np.mean(episode_ts))
                print('Explore rate: ', explore_rate)
                print('Step size: ', step_size)
                print('Average Reward: ', np.mean(episode_rewards))
                print('Non visited states: ', agent.Q_table.size - np.count_nonzero(agent.Q_table))
                print()

            # add current results
            episode_ts.append(t)
            episode_rewards.append(running_reward)

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
                plot_pops(xSol, os.path.join(save_path, 'WORKING_graphs', 'train','LTpops_train_' + str(episode/test_freq) + '.png'))
                np.save(os.path.join(save_path,'WORKING_data','train','train_' + str(int(episode/test_freq)) + '.npy'), xSol)

        else:
            episode_rewards.append(running_reward)
            episode_ts.append(t)


    xSol = np.array(xSol)

    # plot results
    plt.figure(figsize = (16.0,12.0))
    plot_survival(time_avs,
                  os.path.join(save_path,'WORKING_graphs','LT_train_survival.png'),
                  NUM_EPISODES, T_MAX, 'Training')
    np.save(os.path.join(save_path ,'WORKING_data','LT_train_survival.npy'), time_avs)


    plt.figure(figsize = (22.0,12.0))
    plot_pops(xSol, os.path.join(save_path,'WORKING_graphs','LTpops.png'))
    np.save(os.path.join(save_path,'WORKING_data','LTPops.npy'), xSol)

    plt.figure(figsize = (16.0,12.0))
    plot_rewards(rewards_avs,
                 os.path.join(save_path,'WORKING_graphs','LTtrain_rewards.png'),
                 NUM_EPISODES, T_MAX, 'Training')

    # save results
    np.save(os.path.join(save_path,'WORKING_data','LTtrain_rewards.npy'), rewards_avs)
    np.save(os.path.join(save_path,'Q_table.npy'), agent.Q_table)
    np.save(os.path.join(save_path,'reward_sds.npy'), reward_sds)
    np.save(os.path.join(save_path,'time_sds.npy'), time_sds)

    # create and save state action plot
    LT_actions = np.zeros([num_x_states] * num_species)
    lookuptable = agent.Q_table
    for i in range(num_x_states):
        for j in range(num_x_states):
            LT_actions[i,j] = np.argmax(lookuptable[i,j])
            if np.count_nonzero(lookuptable[i,j]) == 0:
                LT_actions[i,j] = - 1
    np.save(os.path.join(save_path,'state_action.npy'), LT_actions)

    print(np.rot90(LT_actions))

    return agent.Q_table



if __name__ == '__main__': # for the server

    three_speces = {
        'ode_params': [1., 0.5, [480000., 480000., 480000.], [520000., 520000., 520000.], [0.6, 0.6, 0.6], [0.00048776, 0.00000102115, 0.00000102115], [0.00006845928, 0.00006845928,  0.00006845928]],
        'Q_params': [[[0, -0.00005, -0.00005],[-0.00005, 0, -0.00005], [-0.00005,-0.00005,0]], 3,3, 10, [0.,1000.], 2, [0., 0.1], 0.9, [200., 200., 200.], [0.05, 0.05, 0.05], 1.],
        'train_params': [50000, 100, 970, 1000, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.1]
    }

    single_auxotroph =  {
        'ode_params': [0.25, 0.5, [480000., 480000.], [520000., 520000.], [1.5, 3.], [0., 0.00000102115], [0.00006845928, 0.00006845928]],
        'Q_params': [[[0, 0],[0, 0]], 2,1, 10, [0.,120000.], 2, [0., 0.3], 0.9, [50000.,50000.], [1.,0.1], 0.25],
        'train_params': [1000, 10, 95, 100, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.1]
    }
    smaller_target_params = {
        'ode_params': [1., 0.5, [480000., 480000.], [520000., 520000.], [0.6, 0.6], [0.00048776, 0.00048776], [0.00006845928, 0.00006845928]],
        'Q_params': [[[-0.0001, -0.0001],[-0.0001, -0.0001]], 2,2, 10, [0.,1000.], 2, [0., 0.1], 0.9, [250.,550.], [0.05,0.05], 1.],
        'train_params': [10000, 100, 950, 1000, 1000, 0.05, 0.5, 0., [50,50,50,50]],
        'noise_params': [False, 0.05]
    }

    # if repeat number and directory supplied run repeats, else just run once
    try:
        directory = sys.argv[1]
        repeat_n = sys.argv[2]
        save_path = '/home/zcqsntr/Scratch/lookup/' + str(directory) + '/repeat' + str(repeat_n)
    except:
        save_path = '/home/zcqsntr/Scratch/lookup/WORKING/'

    Q_table = lookup_table_Q_learn(single_auxotroph, save_path, debug = True)
