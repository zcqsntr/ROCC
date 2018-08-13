import numpy as np
import math
import random

from utilities import *
from scipy.integrate import odeint

class LookupTableAgent():
    '''
    Class that handles reinforcement learning using a lookuptable to store state-action value estimates
    '''
    def __init__(self, num_x_states, num_Cin_states, num_species, num_controlled_species, reward_func = False):
        '''
        Parameters:
            num_x_states: the number of discrete population states the agent can see for each species
            num_Cin_states: the number of different nutrient concentrations the agents can use for each auxotroph
            num_speceis: the number of different bacterial populations
        '''

        if reward_func:
            self.reward = reward_func
        else:
            self.reward = self.simple_reward

        # input validation
        if num_x_states < 1 or not isinstance(num_x_states, int):
            raise ValueError("num_x_States needs to be a positive integer")
        if num_Cin_states < 1 or not isinstance(num_Cin_states, int):
            raise ValueError("num_Cin_States needs to be a positive integer")
        if num_species < 1 or not isinstance(num_species, int):
            raise ValueError("num_species needs to be a positive integer")
        if num_controlled_species < 1 or not isinstance(num_controlled_species, int):
            raise ValueError("num_controlled_species needs to be a positive integer")
        if num_controlled_species > num_species:
            raise ValueError("num_controlled_species cannot be larger than num_species")

        #initilise Q_table
        self.Q_table = np.zeros(tuple([num_x_states]*num_species + [num_Cin_states]*num_controlled_species))


    def train_step(self, X, C, C0, t, explore_rate, learning_rate, Q_params, ode_params):
        '''
        Carries out one step of training, updates the agents value estimates and the state of the system
        Parameters:
            X: the populations of the bacteria
            C: the concentrations of each auxotrophic nutrient
            C0: the concentrations of the carbon source
            t: current time
            explore_rate: the chance of the agent taking a radnom action
            learning_rate: the agents learning_rate
            Q_params: learning parameters
            ode_params: parameters for the numerical solver odeint
        Returns:
            X1: populations at next timestep
            C1: concentrations of auxotrophc nutrient at each time step
            C01: concentration of the carbon source at next time point
            xSol: full next bit of the populations solutions, including the skipped frames
            reward
        '''
        #extract parameters
        A, num_species, num_controlled_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params

        #discritise current state
        state = np.array(state_to_bucket(X, x_bounds, num_x_states))

        # select an action
        action_indeces = self.select_action(state, explore_rate, num_Cin_states, num_controlled_species)
        action_index = np.ravel_multi_index(action_indeces, [num_Cin_states] * num_controlled_species) # turn into one hot index
        Cin = action_to_state(action_index, num_controlled_species, num_Cin_states, Cin_bounds) #turn index into C0

        if num_species - num_controlled_species == 1:
            Cin = np.append([1], Cin)

        #create state vector
        S = np.append(X, C)
        S = np.append(S, C0)

        # execute action

        time_diff = 4
        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin,A,ode_params, num_species))[1:]

        # extract next values from sol
        xSol = sol[:, 0:num_species]
        X1 = sol[-1, :num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]


        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))


        reward = self.reward(X1)

        #desctrise new state for next interation
        state1 = state_to_bucket(X1, x_bounds, num_x_states)

        # update Q table
        best_q = np.max(self.Q_table[tuple(state)])
        self.Q_table[tuple(state)][tuple(action_indeces)] += learning_rate*(reward+gamma*best_q - self.Q_table[tuple(state)][tuple(action_indeces)])

        return X1, C1, C01, xSol, reward


    def select_action(self, state, explore_rate, num_Cin_states, num_species):
        '''
        Chooses an action based on the epsilon greedy policy
        Parameters:
            state: current, descritised, state of the system
            explore_rate: the chance the agent will choose a random action
            num_Cin_states: the number of actions the agent can choose from
            num_species: the number of different bacterial populations
        Returns:
            action: the chosen action
        '''
        if random.random() < explore_rate:
            action = np.random.randint(0, num_Cin_states, size = num_species)
        else:
            action = np.unravel_index(np.argmax(self.Q_table[tuple(state)]), self.Q_table[tuple(state)].shape)
        return action

    def select_action_softmax(self, state, explore_rate, Cin_bounds):
        '''
        Chooses an action based on the a softmax policy
        Parameters:
            state: current, descritised, state of the system
            explore_rate: the chance the agent will choose a random action
            Cin_bounds: the bounds of the concentrations
        Returns:
            action: the chosen action
        '''
        Q_values = Q_table[tuple(state)]
        Q_values = Q_values.reshape(1, np.prod(np.array(Q_values.shape)))
        action = np.unravel_index(softmax_selection(explore_rate, Q_values),Q_table[state[0], state[1]].shape)
        return action

    def simple_reward(self,X):
        '''
        Simple reward funtion based on the populations of each bacterial species
        Parameters:
            X: array of all population levels
        Returns:
            reward: the reward recieved
        '''

        if all(x > 1.5 for x in X):
            reward = 1
        else:
            reward = - 1
        return reward
