import os
import sys
import yaml

# add to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *
from neural_script import neural_Q_learn

# open parameter file
f = open(os.path.join('parameter_files', 'double_auxotroph.yaml'))
parameters = yaml.load(f)
f.close()

validate_param_dict(parameters) # input validation
save_path = os.path.join('..', 'results', 'NN_smaller_target')

def smaller_target_reward(X):
    if 100 < X[0] < 400 and 400 < X[1] < 700:
        reward = 1
    else:
        reward = -1
    return reward

neural_Q_learn(parameters, save_path, debug = True, reward_func = smaller_target_reward)
