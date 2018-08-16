import os
import sys
import yaml

# add to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *
from neural_script import neural_Q_learn

# open parameter file
f = open(os.path.join('parameter_files', 'single_auxotroph.yaml'))
parameters = yaml.load(f)
f.close()

validate_param_dict(parameters) # input validation
save_path = os.path.join('..', 'results', 'NN_single_aux')

def single_aux_reward(X):
    if all(x > 10000 for x in X):
        reward = 1
    else:
        reward = -1
    return reward

neural_Q_learn(parameters, save_path, debug = True, reward_func = single_aux_reward)
