import os
import sys
import yaml

# add to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *
from lookuptable_script import lookuptable_Q_learn


f = open(os.path.join('parameter_files', 'single_auxotroph.yaml'))
parameters = yaml.load(f)
f.close()

validate_param_dict(parameters) # input validation
save_path = os.path.join('..', 'results')
lookuptable_Q_learn(parameters, save_path, True)
