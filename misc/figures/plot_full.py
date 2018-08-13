
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))


from plot_funcs import *
from utilities import *
'''plots for lookuptable and neural agent for spock system with reward given if N > 3'''

matplotlib.rcParams.update({'font.size': 20})

# load lookuptable data
LT_pops = np.load('/Users/Neythen/masters_project/results/lookup_table_results/single_aux_repeats/repeat0/WORKING_data/LTPops.npy')
LT_survival = np.load('/Users/Neythen/masters_project/results/lookup_table_results/smaller_target_fixed/WORKING/WORKING_data/LT_train_survival.npy')
LT_rewards = np.load('/Users/Neythen/masters_project/results/lookup_table_results/smaller_target_fixed/WORKING/WORKING_data/LTtrain_rewards.npy')


# load neural data
N_pops = np.load('/Users/Neythen/masters_project/results/lookup_table_results/single_aux_repeats/repeat1/WORKING_data/QPops.npy')
N_survival = np.load('/Users/Neythen/masters_project/results/Q_learning_results/smaller_target/WORKING/repeat0/WORKING_data/Q_train_survival.npy')
N_rewards = np.load('/Users/Neythen/masters_project/results/Q_learning_results/smaller_target/WORKING/repeat0/WORKING_data/Qtrain_rewards.npy')



NUM_EPISODES = 10000
explore_rates = [get_explore_rate(e, 0, 950) for e in range(NUM_EPISODES)]

pops = [LT_pops, N_pops]
rewards = [LT_rewards, N_rewards]
survival = [LT_survival, N_survival]

agents = ['Lookup Table', 'Neural Network']
colours = ['g', 'r']

T_MAX = 200
for p in [0,1]:
    plt.figure(figsize = (13, 8.0))
    plt.ylim([0, 135000])
    plt.plot(np.linspace(0,T_MAX,T_MAX), pops[p][0:T_MAX])

    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Population ($10^6$ cells/L)')

    plt.savefig('pops_' + agents[p])



time_labels = ['Lookup Table Time', 'Neural Network Time']

fig, ax1 = plt.subplots(figsize = (12, 8.0))
ax2 = ax1.twinx()

ax1.set_xlabel('Episode')

ax2.set_ylabel('Average Time Survived (hours)')

ax2.set_xlim([0, NUM_EPISODES])
ax2.set_ylim([0, 3010])



for s in [0,1]:
    ax2.plot(np.linspace(0, NUM_EPISODES, len(survival[s])), [survival[s][i] * 3 for i in range(len(survival[s]))], label = time_labels[s], color = colours[s])


ax1.plot(np.linspace(0, NUM_EPISODES, len(explore_rates)), explore_rates, label = 'Explore Rate', color = 'c')

ax2.legend(loc='center right', prop={'size': 15})
ax1.legend(loc='center right', bbox_to_anchor=(0,0,1, 0.8), prop={'size': 15})

plt.xlabel('Episode')
ax1.set_ylabel('Explore Rate')

plt.savefig('survival')


fig, ax1 = plt.subplots(figsize = (12, 8.0))
ax2 = ax1.twinx()

ax1.set_xlabel('Episode')
ax2.set_ylabel('Average Reward')

ax2.set_xlim([0, NUM_EPISODES])
reward_labels = ['Lookup Table Reward', 'Neural Network Reward']


for r in [0,1]:
    ax2.plot(np.linspace(0, NUM_EPISODES, len(rewards[r])), [rewards[r][i]  for i in range(len(rewards[r]))], label = reward_labels[r], color = colours[r])


ax1.set_ylabel('Explore Rate')
ax1.plot(np.linspace(0, NUM_EPISODES, len(explore_rates)), explore_rates, label = 'Explore Rate', color = 'c')

ax2.legend(loc='upper center', prop={'size': 15})
ax1.legend(loc='upper center', bbox_to_anchor=(0,0,1, 0.85), prop={'size': 15})
plt.savefig('rewards')
