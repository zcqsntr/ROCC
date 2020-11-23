import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)


from chemostat_envs import *
from fitted_Q_agents import *

from double_aux_rewards import *

param_path = os.path.join(C_DIR, 'parameter_files/smaller_target_good_ICs_mutation.yaml')

save_path = 'mutation_exp'

update_timesteps = 1
sampling_time = 3
delta_mode = False
tmax = 100
explore_rate = 0.

env = ChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)

agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species], cost_function = fig_6_reward_function_new_target)
agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
#agent.save_network_tensorflow(os.path.dirname(os.path.abspath(__file__)) + '/100eps/training_on_random/')
#agent.load_network_tensorflow('/Users/Neythen/Desktop/summer/fitted_Q_iteration/chemostat/100eps/training_on_random')

trajectory = agent.run_mutation_episode(env, explore_rate, tmax, train = True)
test_r = np.array([t[2] for t in trajectory])
test_a = np.array([t[1] for t in trajectory])
values = np.array(agent.values)
os.makedirs(save_path, exist_ok = True)
np.save(save_path + '/values.npy', values)
env.plot_trajectory([0,1])

plt.savefig(save_path + '/populations.png')
np.save(save_path + '/trajectory.npy', env.sSol)


plt.figure()
plt.plot(test_r)
np.save(save_path + '/rewards.npy', test_r)
plt.savefig(save_path + '/rewards.png')


plt.figure()
plt.plot(test_a)
np.save(save_path + '/actions.npy', test_a)
plt.savefig(save_path + '/actions.png')

plt.figure()
for i in range(4):
    plt.plot(values[:, i], label = 'action ' + str(i))
plt.legend()

plt.savefig(save_path + '/values.png')
