import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from plot_funcs import *
from utilities import *

matplotlib.rcParams.update({'font.size': 20})
LT_root_path = '/Users/Neythen/masters_project/results/lookup_table_results/single_aux_repeats'
N_root_path = '/Users/Neythen/masters_project/results/Q_learning_results/single_aux_repeats'

''' Plot reward and survival means over time '''
LT_reward_arrays = []
LT_survival_arrays = []
N_reward_arrays = []
N_survival_arrays = []

NUM_EPISODES = 10000
explore_rates = [get_explore_rate(e, 0, 950) for e in range(NUM_EPISODES)]
episodes = np.linspace(0, NUM_EPISODES, num = 100)
for i in range(1,4):
    LT_survival_path = LT_root_path + '/repeat' + str(i) + '/WORKING_data/LT_train_survival.npy'
    LT_reward_path = LT_root_path + '/repeat' + str(i) + '/WORKING_data/LTtrain_rewards.npy'
    N_survival_path = N_root_path + '/repeat' + str(i) + '/WORKING_data/Q_train_survival.npy'
    N_reward_path = N_root_path + '/repeat' + str(i) + '/WORKING_data/Qtrain_rewards.npy'
    LT_survival_array = np.load(LT_survival_path)

    LT_reward_array = np.load(LT_reward_path)
    N_survival_array = np.load(N_survival_path)[0:100]
    N_reward_array = np.load(N_reward_path)[0:100]
    LT_reward_arrays.append(LT_reward_array)
    LT_survival_arrays.append(LT_survival_array)
    N_reward_arrays.append(N_reward_array)
    N_survival_arrays.append(N_survival_array)

LT_survival_arrays = np.array(LT_survival_arrays)
LT_reward_arrays = np.array(LT_reward_arrays)
N_survival_arrays = np.array(N_survival_arrays)
N_reward_arrays = np.array(N_reward_arrays)

LT_survival_SDs = np.std(LT_survival_arrays, axis = 0)
LT_reward_SDs = np.std(LT_reward_arrays, axis = 0)
N_survival_SDs = np.std(N_survival_arrays, axis = 0)
N_reward_SDs = np.std(N_reward_arrays, axis = 0)

LT_survival_means = np.mean(LT_survival_arrays, axis = 0)
LT_reward_means = np.mean(LT_reward_arrays, axis = 0)
N_survival_means = np.mean(N_survival_arrays, axis = 0)
N_reward_means = np.mean(N_reward_arrays, axis = 0)


# plot survival
fig, ax1 = plt.subplots(figsize = (12, 8.0))
ax2 = ax1.twinx()


ax1.set_xlabel('Episode')
ax2.set_ylabel('Average Time Survived (hours)')
ax1.set_ylabel('Explore Rate')

ax2.set_xlim([0, NUM_EPISODES + 30])
ax1.set_xlim([0, NUM_EPISODES + 30])

ax2.set_ylim([0, 3030])

ax1.plot(np.linspace(0, NUM_EPISODES, len(explore_rates)), explore_rates, label = 'Explore Rate', color = 'c')
ax2.errorbar(episodes, LT_survival_means*3, LT_survival_SDs*3, color = 'g', label = 'Lookup Table Time')
ax2.errorbar(episodes, N_survival_means*3, N_survival_SDs*3, color = 'r', label = 'Neural Network Time')
ax2.legend(loc='center right', prop={'size': 15})
ax1.legend(loc='center right', bbox_to_anchor=(0,0,1, 0.8), prop={'size': 15})

plt.xlabel('Episode')
#ax1.set_ylabel('Explore Rate')
plt.savefig('survival')


# plot rewards
plt.figure()
fig, ax1 = plt.subplots(figsize = (12, 8.0))
ax2 = ax1.twinx()


ax1.set_xlabel('Episode')
ax2.set_ylabel('Average Reward')
ax1.set_ylabel('Explore Rate')

ax2.set_xlim([0, NUM_EPISODES+30])


ax1.set_xlim([0, NUM_EPISODES+30])

ax1.plot(np.linspace(0, NUM_EPISODES, len(explore_rates)), explore_rates, label = 'Explore Rate', color = 'c')

ax2.errorbar(episodes, LT_reward_means, LT_reward_SDs, color = 'g', label = 'Lookup Table Reward')
ax2.errorbar(episodes, N_reward_means, N_reward_SDs, color = 'r', label = 'Neural Network Reward')
ax2.legend(loc='upper center', prop={'size': 15})
ax1.legend(loc='upper center', bbox_to_anchor=(0,0,1, 0.8), prop={'size': 15})

plt.xlabel('Episode')
#ax1.set_ylabel('Explore Rate')
plt.savefig('rewards')





''' Plot all the population curves '''
#NN
T_MAX = 200
matplotlib.rcParams.update({'font.size': 25})
fig = plt.figure(figsize = (22,22))
ax = fig.add_subplot(111)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Population ($10^6 cellsL^{-1}$)', labelpad = 65)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

for i in range(1,4):
    ax = fig.add_subplot(5,2,i+1)

    path = N_root_path + '/repeat' + str(i) + '/WORKING_data/Qpops.npy'
    pops = np.load(path)[0:T_MAX, :]
    plot_pops(pops, save_path = False)
plt.savefig('N_repeats.png')


#LT
fig = plt.figure(figsize = (22,22))
ax = fig.add_subplot(111)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Population ($10^6 cellsL^{-1}$)', labelpad = 65)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
for i in range(1,4):
    ax = fig.add_subplot(5,2,i+1)

    path = LT_root_path + '/repeat' + str(i) + '/WORKING_data/LTpops.npy'
    pops = np.load(path)[0:T_MAX, :]
    plot_pops(pops, save_path = False)
plt.savefig('LT_repeats.png')


''' All the state-action plots '''
# lookuptable

num_x_states = 10
LT_state_actions_file = open('LT_state_actions', 'w+')
LT_state_actions = []
for i in range(1,4):
    LT_actions = np.zeros((num_x_states, num_x_states))
    path = LT_root_path + '/repeat' + str(i) + '/Q_table.npy'
    lookuptable = np.load(path)
    for i in range(num_x_states):
        for j in range(num_x_states):
            LT_actions[i,j] = np.argmax(lookuptable[i,j])
            '''
            if LT_actions[i,j] == 2:
                LT_actions[i,j] = 1
            elif LT_actions[i,j] == 1:
                LT_actions[i,j] = 2
            '''

            if np.count_nonzero(lookuptable[i,j]) == 0:

                LT_actions[i,j] = - 1
    LT_state_actions.append(np.rot90(LT_actions))
'''
for i in range(1,4):
    for j in range(1,4):
        if np.array_equal(LT_state_actions[i], LT_state_actions[j]):
            print(i,j)
print('-------------')
LT_state_actions = np.array(LT_state_actions)

# NN

N_state_actions = []
N_state_actions_file = open('N_state_actions', 'w+')
for i in range(1,4):
    N_visited_states = np.load(N_root_path + '/repeat' + str(i) + '/visited_states.npy')
    N_state_action = np.load(N_root_path + '/repeat'+ str(i) + '/state_action.npy')
    N_state_action = N_state_action.reshape(10,10)
    N_visited_states = N_visited_states.reshape(10,10)
    for i in range(num_x_states):
        for j in range(num_x_states):

            if N_state_action[i,j] == 2:
                N_state_action[i,j] = 1
            elif N_state_action[i,j] == 1:
                N_state_action[i,j] = 2


            if N_visited_states[i,j] == 0:
                N_state_action[i,j] = - 1



    N_state_actions.append(np.rot90(N_state_action))

for i in range(1,4):
    for j in range(1,4):
        if np.array_equal(N_state_actions[i], N_state_actions[j]):
            print(i,j)

conserved = N_state_actions[i]
for i in [3,4,5,7,8,9]:
    conserved[conserved == N_state_actions[i]] = 0

N_state_actions = np.array(N_state_actions)
'''
'''
for i in range(5):
    N_first = N_state_actions[2*i,:,:]
    N_second = N_state_actions[2*i+1, :,:]
    LT_first = LT_state_actions[2*i,:,:]
    LT_second = LT_state_actions[2*i+1, :,:]

    for j in range(10):
        N_state_actions_file.write(np.array2string(N_first[j]))
        N_state_actions_file.write('  ')
        N_state_actions_file.write(np.array2string(N_second[j]))
        N_state_actions_file.write('\n')

        LT_state_actions_file.write(np.array2string(LT_first[j]))
        LT_state_actions_file.write('  ')
        LT_state_actions_file.write(np.array2string(LT_second[j]))
        LT_state_actions_file.write('\n')


    N_state_actions_file.write('\n')
    LT_state_actions_file.write('\n')
'''
