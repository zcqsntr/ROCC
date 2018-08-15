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


Results will automatically be saved a directory structure will be created in the supplied save_path.

save_path\
 ├── WORKING_data\
 │   ├── train\
 │   ├── train_survival.npy\
 │   ├── pops.npy\
 │   ├── train_rewards.npy\
 │   ├── time_sds.npy\
 │   └── reward_sds.npy\
 ├── WORKING_graphs\
 │   ├── train\
 │   ├── train_survival.npy\
 │   ├── pops.npy\
 │   └── train_rewards.npy\
 ├── Q_table.npy\
 ├── state_action.npy\
 └── visited_states.npy


 .
├── build                   # Compiled files (alternatively `dist`)
├── docs                    # Documentation files (alternatively `doc`)
├── src                     # Source files (alternatively `lib` or `app`)
├── test                    # Automated tests (alternatively `spec` or `tests`)
├── tools                   # Tools and utilities
├── LICENSE
└── README.md


### Examples
The examples directory contains examples including the single auxotroph system, double auxotroph system and the smaller target system. These show how the functions can be imported into a script and used to carry out reinforcement learning on a system specified in a parameter file. These examples are run with python, for example:
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
