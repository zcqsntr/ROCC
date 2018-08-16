## CBcurl: Control of Bacterial communities using reinforcement learning

### Installation
To use the package within python scropts, `CBcurl` must be in PYTHONPATH.
To use the package from the command line `CBcurl` must be in you $PATH variable

### Dependencies
Standard python dependencies are required: `numpy`, `scipy`, `matplotlib`.`yaml` is required to parse parameter files. `argparse` is required for the command line application. `pytest` is required to run the unit tests. If you would like to use the neural network functionality then `TensorFlow` is required, the lookuptable versions of the agents will work without 'TensorFlow'. Instructions for installing 'TensorFlow' can be found here:
 https://www.tensorflow.org/install/

### User Instructions
`CBcurl` can be used in two ways:
1) Code files can be imported into scripts, see examples

2) By running from the command line:
```console
$ CBcurl.py agent parameters [options]
```
positional arguments:

  - agent: L selects the lookuptable agent, N selects the neural agent

  - parameters: a .yaml file containing the parameters that define the system

optional arguments:

  - -s, --save_path: path to save results

  - -d, --debug: True or False, selects whether or not to print debug information

  - -r, --repeats: the number of time the training run should be repeated


Results will automatically be saved a directory structure will be created in the supplied save_path:

```
save_path
 ├── WORKING_data
 │   ├── train
 │   ├── train_survival.npy
 │   ├── pops.npy
 │   ├── train_rewards.npy
 │   ├── time_sds.npy
 │   └── reward_sds.npy
 ├── WORKING_graphs
 │   ├── train
 │   ├── train_survival.npy
 │   ├── pops.npy
 │   └── train_rewards.npy
 ├── WORKING_saved_network
 ├── Q_table.npy
 ├── state_action.npy
 └── visited_states.npy
```

WORKING_data contains the numpy arrays of gathered training data. train contains population curve of a training episode at intervals given by the test_freq parameter, these are only saved if DEBUG = True. train_survival.npy contains moving averages of the survival times of the episodes in the training run, time_sds.npy contains the standard deviations of these moving averages. train_rewards.npy contains moving averages of the rewards of the episodes in the training run, reward_sds.npy contains the standard deviations of these moving averages. pop.npy contains the population of the final episode after the agent has been trained.

WORKING_graphs contains .png graphs of the data in WORKING_data.

WORKING_saved_network contains the saved Q network, only saved by a neural agent

Q_table.npy contains the final Q_table, only saved by a lookuptable_agent

state_action.npy conatins the state action array

visited_states.npy contains an array of the number of times each state was visited, only saved by a neural agent



### Examples
The examples directory contains examples including the single auxotroph system, double auxotroph system and the smaller target system. These show how the functions can be imported into a script and used to carry out reinforcement learning on a system specified in a parameter file. These can be run without adding CBcurl to PYTHON_PATH. As they take a ling time to run, some of them have been run already and the results are in the results directory. These examples are run with python, for example:
```console
$ python LT_single_aux_example.py
```

### Testing
In the testing directory run

```console
$ py.test
```
to run all unit tests.


### misc
The misc directory contains code that is not directly involved in the user facing application. This includes the code used for the simulations to find steady states, the analysis of the eigenvalues of the Jacobian at these steady states and plotting code.
