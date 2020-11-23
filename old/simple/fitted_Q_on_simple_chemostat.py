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

param_path = os.path.join(C_DIR, 'parameter_files', 'simple_example_params.yaml')

from argparse import ArgumentParser

def simple_cost_function(state, action, next_state):

    if 20000 < state[0] < 40000:
        cost = 0
        done = False
    elif state[0] > 100:
        cost = 0.1
        done = False
    else:
        done = True
        cost = 1
    return cost, done


def fig_6_reward_function(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10

    done = False


    if any(state < 10):
        reward = - 1
        done = True

    return reward, done

def continuous_cost_function(state, action, next_state):
    targ = 30000

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/targ**2)/10

    done = False


    if any(state < 10):
        reward = - 1
        done = True

    return reward, done


def entry():
    '''
    Entry point for command line application handle the parsing of arguments and runs the relevant agent
    '''
    # define arguments
    parser = ArgumentParser(description = 'Bacterial control app')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-r', '--repeat')
    arguments = parser.parse_args()

    # get number of repeats, if not supplied set to 1
    repeat = int(arguments.repeat)

    save_path = os.path.join(arguments.save_path, 'repeat' + str(repeat))
    print(save_path)
    run_test(save_path)

def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/simple_example_params.yaml')
    update_timesteps = 1
    sampling_time = 1/60
    delta_mode = False
    tmax = 1000
    n_episodes = 10
    times = []
    rewards = []
    env = SimpleChemostatEnv(param_path, sampling_time, update_timesteps, delta_mode)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species], cost_function = continuous_cost_function)
    
    for i in range(n_episodes):
        print('EPISODE: ', i)
        # training EPISODE
        #explore_rate = 0
        #explore_rate = agent.get_rate(i, 0, 1, 9)
        explore_rate = 1
        print(explore_rate)
        env.reset()
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        times.append(len(trajectory))
        rewards.append(train_r)

        '''
        env.reset()
        explore_rate = 0.2
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax)

        #hint_to_goal(agent.network, agent.optimiser)
        print('Train Time: ', len(trajectory))
        '''
        # testing EPISODE
        explore_rate = 0
        env.reset()
        #env.state = (np.random.uniform(-1, 1), 0, np.random.uniform(-0.5, 0.5), 0)
        trajectory, test_r = agent.run_episode(env, explore_rate, tmax, train = False)
        print('Test Time: ', len(trajectory))

        # get the number of timesteps in the goal
        times.append(len(trajectory))
        '''
        if test_r > 9:
            env.plot_trajectory([0,1])
            plt.show()
        '''
        print()

        rewards.append(test_r)

    rewards = np.array(rewards)
    os.makedirs(save_path, exist_ok = True)
    agent.save_results(save_path)
    agent.save_network(save_path)
    plt.figure()
    plt.plot(times)

    plt.xlabel('Timestep')
    plt.ylabel('Timesteps until terminal state')
    plt.savefig(save_path + '/times.png')

    env.plot_trajectory([0,1])
    plt.savefig(save_path + '/populations.png')
    np.save(save_path + '/trajectory.npy', env.sSol)

    plt.figure()
    plt.plot(rewards)
    plt.savefig(save_path + '/episode_rewards.png')


    # test trained policy with smaller time step


if __name__ == '__main__':
    entry()
