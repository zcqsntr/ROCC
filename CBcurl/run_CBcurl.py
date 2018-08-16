#!/usr/bin/env python

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import yaml


from utilities import *
from neural_script import neural_Q_learn
from lookuptable_script import lookuptable_Q_learn
from argparse import ArgumentParser

def entry():
    '''
    Entry point for command line application handle the parsing of arguments and runs the relevant agent
    '''
    # define arguments
    parser = ArgumentParser(description = 'Bacterial control app')
    parser.add_argument('agent')
    parser.add_argument('parameters')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-d', '--debug')
    parser.add_argument('-r', '--repeats')
    arguments = parser.parse_args()

    # open parameter file
    f = open(arguments.parameters)
    parameters = yaml.load(f)
    f.close()

    validate_param_dict(parameters) # input validation

    debug = arguments.debug is None

    # get number of repeats, if not supplied set to 1
    try:
        repeats = int(arguments.repeats)
    except:
        repeats = 1

    # run with selected agent, savepath and number of repeats
    if arguments.agent.upper() == 'N':
        for i in range(repeats):
            if arguments.save_path is not None: # if savepath supplied
                save_path = os.path.join(arguments.save_path, 'repeat' + str(i))
            else:
                save_path = os.path.join('..','results','neural_results','WORKING','repeat' + str(i))
            neural_Q_learn(parameters, save_path , debug)

    elif arguments.agent.upper() == 'L':
        for i in range(repeats):
            if arguments.save_path is not None: # if savepath supplied
                save_path = os.path.join(arguments.save_path, 'repeat' + str(i))
            else:
                save_path = os.path.join('..','results','lookuptable_results','WORKING','repeat' + str(i))
            lookuptable_Q_learn(parameters, save_path, debug)
    else:
        raise ValueError("Command line argument  'agent' needs to be L, N, or R")

if __name__ == '__main__':
    entry()
