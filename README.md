

## CBcurl: Control of Bacterial communities using reinforcement learning

### Installation
To use the package, `CBcurl` must be in PYTHONPATH.

### Dependencies
Standard python dependencies are required: `numpy`, `scipy`, `matplotlib`.`yaml` is required to parse parameter files. `argparse` is required for the command line application. `pytest` is required to run the unit tests. If you would like to use the neural network functionality then `TensorFlow` is required, the lookuptable versions of the agents will work without 'TensorFlow'. Instructions for installing 'TensorFlow' can be found here:
 https://www.tensorflow.org/install/



### User Instructions
`CBcurl` can be used in two ways:
1) Code files can be imported into scripts, see examples
2) By running from the command line:
```console
$ python CBcurl.run_CBcurl.py agent parameters [options]
```
positional arguments:

  - agent: L selects the lookuptable agent, N selects the neural agent

  - parameters: a .yaml file containing the parameters that define the system

optional arguments:

  - -s, --save_path: path to save results

  - -d, --debug: True or False, selects whether or not to print debug information

  - -r, --repeats: the number of time the training run should be repeated

### Examples
The examples directory contains examples including the single auxotroph system, double auxotroph system and the smaller target system. These show how the functions can be imported into a script and used to carry out reinforcement learning on a system specified in a parameter file. These examples are run with python

### Testing
In the testing directory run

```console
:testing $ py.test
```
to run all unit tests.
