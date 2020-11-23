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

from matplotlib import pyplot as plt


def continuous_cost_function(state, action, next_state):
    targ = 30000

    SSE = sum((np.array(state)-targ)**2)

    reward = (1 - SSE/targ**1.7)/10

    done = False
    print(reward)

    if any(state < 10):
        reward = - 1
        done = True

    return reward, done

def test_trajectory():
    param_file = os.path.join(C_DIR, 'parameter_files', 'simple_example_params.yaml')
    print('reward: ', continuous_cost_function(np.array([30000]), None, None))
    print('reward: ', continuous_cost_function(np.array([31000]), None, None))

    print('reward: ', continuous_cost_function(np.array([40000]), None, None))


    update_timesteps = 1
    sampling_time = 1/60
    env = SimpleChemostatEnv(param_file, sampling_time, update_timesteps, False)
    rew = 0

    actions = []
    for i in range(1000):

        a = np.random.choice(range(2))

        #a = 3
        #print(a)
        '''
        a = 3
        if i == 400:
            a = 2

        if i == 500:
            a = 1
        '''

        state = env.get_state()

        '''
        if state > 30000:
            a = 0
        else:
            a = 1
        '''
        r, done = continuous_cost_function(state, None, None)
        print(r)
        rew += r
        env.step(a)
        if done:
            break

        actions.append(a)
    print(actions)

    env.plot_trajectory([0])
    plt.show()
    env.plot_trajectory([1])
    plt.show()

    print(rew)
if __name__ == '__main__':
    test_trajectory()
