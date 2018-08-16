import os
import sys
import yaml

# add to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *
from lookuptable_script import lookuptable_Q_learn

# open parameter file
f = open(os.path.join('parameter_files', 'smaller_target.yaml'))
parameters = yaml.load(f)
f.close()

validate_param_dict(parameters) # input validation
save_path = os.path.join('..', 'results', 'LT_smaller_target')

def smaller_target_reward(X):
    if 100 < X[0] < 400 and 400 < X[1] < 700:
        reward = 1
    else:
        reward = - 1

    if any(x < 1/1000 for x in X):
        reward = - 10

    return reward

lookuptable_Q_learn(parameters, save_path, debug = True, reward_func = smaller_target_reward)
