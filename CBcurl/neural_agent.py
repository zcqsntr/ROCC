import numpy as np
import tensorflow as tf
import math
import random

from utilities import *
from scipy.integrate import odeint


class NeuralAgent():
    '''
    Class that handles reinforcement learning using a deep Q network to  store state-action value estimates
    '''
    def __init__(self, layer_sizes, buffer_size, target = False, reward_func = False):
        '''
        Parameters:
            layer_sizes: list holding the numebr of nodes in each layer, including input and output layers
            target: whether or not a seperate target network is used
        '''

        # set reward func if given
        if reward_func:
            self.reward = reward_func
        else:
            self.reward = self.simple_reward

        self.target = target
        #create primary network
        with tf.variable_scope('primary'):
            self.inputs, self.predQ = self.create_network(layer_sizes, 'primary')

        #create target network
        if target:
            with tf.variable_scope('target'):
                self.target_inputs, self.target_predQ = self.create_network(layer_sizes, 'target')

        # create placeholders and functions required for training
        self.TD_target= tf.placeholder(shape = [None, layer_sizes[-1]], dtype = tf.float32)
        loss = tf.reduce_sum(tf.square(tf.stop_gradient(self.TD_target) - self.predQ))
        trainer = tf.train.AdamOptimizer(learning_rate = 0.0001) # reduced learning rate makes it more stable
        self.updateModel = trainer.minimize(loss)
        self.predict = tf.argmax(self.predQ, 1)
        self.experience_buffer = ExperienceBuffer(buffer_size)




    def create_network(self,layer_sizes, type):
        '''
        Creates a neural network and returns the input and output layers
        Parameters:
            layer_sizes: list holding the numebr of nodes in each layer, including input and output layers
            type: specifies whether this is the primary or the target network
        Returns:
            input: the input layer
            output: the output layer
        '''
        initialiser = tf.contrib.layers.xavier_initializer()
        regulariser = tf.contrib.layers.l2_regularizer(scale = 0.01) # weight regularisation to prevent divergence

        inputs = tf.placeholder(shape = [None, layer_sizes[0]], dtype = tf.float32)

        # non linear hidden layers
        for l in range(len(layer_sizes)-2):
            if l > 0:
                previous = current
            else:
                previous = inputs
            bias = tf.get_variable(name = type + 'bias' + str(l),initializer = initialiser([layer_sizes[l+1]]), regularizer = regulariser)
            weights = tf.get_variable(name = type + 'weights' + str(l), initializer = initialiser([layer_sizes[l], layer_sizes[l+1]]),regularizer = regulariser)
            current = tf.nn.relu(tf.add(tf.matmul(previous, weights), bias)) # non linear hidden layers

        # linear output layer
        bias = tf.get_variable(name = type + 'bias-out',initializer = initialiser([layer_sizes[-1]]),regularizer = regulariser)
        weights = tf.get_variable(name = type +  'weights-out', initializer = initialiser([layer_sizes[-2], layer_sizes[-1]]),regularizer = regulariser)
        output = tf.add(tf.matmul(current, weights), bias)

        return inputs, output


    def train_step(self, sess, X, C, C0, t, visited_states, explore_rate, step_size, Q_params, ode_params,n):
        '''Carries out one instantaneous Q_learning training step
        Parameters:
            X: array storing the populations of each bacteria
            C: array contatining the concentrations of each rate limiting nutrient
            C0: the concentration of the common carbon source at this time point
            t: the current time
            visited_states: array to keep track of which states have been visited
            explore_rate: the current explore rate
            step_size: current Q_learning step size
            Q_params: learning parameters
            ode_params: parameters for the numerical solver odeint
        Returns:
            X1: populations at next timestep
            C1: concentrations of auxotrophc nutrient at each time step
            C01: concentration of the carbon source at next time point
            xSol: full next bit of the populations solutions, including the skipped frames
            reward
        '''

        A, num_species, num_controlled_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params # extract parameters

        state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a on hot vector
        allQ = np.array(sess.run(self.predQ, feed_dict= {self.inputs:state})) # get Q values for this state
        visited_states += state # count states visited

        # check network isnt outputting Nan
        assert all(not np.isnan(Q) for Q in allQ[0]) , 'Nan found in output, network probably unstable'

        action = epsilon_greedy(explore_rate, allQ) # get predicted best action

        # create state vector
        S = np.append(X, C)
        S = np.append(S, C0)

        # convert chosen action index to a concentration
        Cin = action_to_state(action, num_controlled_species, num_Cin_states, Cin_bounds) # take out this line to remove the effect of the algorithm

        if num_species - num_controlled_species == 1: # hacky way to check for single zuxotroph system
            Cin = np.append([1], Cin)

        time_diff = 4 # frame skipping
        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin,A,ode_params, num_species))[1:]

        # extract information from solution
        xSol = sol[:, 0:num_species]
        X1 = sol[-1, :num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]

        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))

        # turn new state into one hot vector
        state1 = state_to_one_hot(X1, num_species, x_bounds, num_x_states) # flatten to a vector
        reward = self.reward(X1)

        # get Q values for new state
        Q1 = sess.run(self.predQ, feed_dict = {self.inputs: state1})

        #build temporal difference target
        maxQ1 = np.max(Q1)
        targetQ = allQ
        targetQ[0, action] += step_size*(reward + gamma*maxQ1 - allQ[0,action])

        # train network based on target and predicted Q values
        sess.run([self.updateModel], feed_dict = {self.inputs: state, self.TD_target:targetQ})

        return X1, C1, C01, xSol, reward, allQ, visited_states

    def pre_train_step(self, sess, X, C, C0, t, Q_params, ode_params):
        '''
        Carries out one random training step to gather experience for the experience buffer when using DQN
        Parameters:
            X: array storing the populations of each bacteria
            C: array contatining the concentrations of each rate limiting nutrient
            C0: the concentration of the common carbon source at this time point
            t: the current time
            Q_params: learning parameters
            ode_params: parameters for the numerical solver odeint
        Returns:
            X1: populations at next timestep
            C1: concentrations of auxotrophc nutrient at each time step
            C01: concentration of the carbon source at next time point
        '''

        A, num_species, num_controlled_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params # extract parameters

        state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector
        action = np.random.randint(num_Cin_states**num_controlled_species) # choose random action

        # get new state and reward
        S = np.append(X, C)
        S = np.append(S, C0)

        # convert chosen action index to a concentration
        Cin = action_to_state(action, num_controlled_species, num_Cin_states, Cin_bounds) # take out this line to remove the effect of the algorithm

        if num_species - num_controlled_species == 1: # hacky way to check for auxotroph system
            Cin = np.append([1], Cin)

        # get next time step
        time_diff = 4  # frame skipping
        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin,A,ode_params, num_species))[1:]

        # extract information from sol
        xSol = sol[:, 0:2]
        X1 = sol[-1, :num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]


        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))

        # turn new state into one hot vector
        state1 = state_to_one_hot(X1, num_species, x_bounds, num_x_states) # flatten to a vector
        reward = self.reward(X1)

        self.experience_buffer.add([state, action, reward, state1])

        return X1, C1, C01

    def train_step_target(self, sess, X, C, C0, t, visited_states, explore_rate, step_size, Q_params, ode_params, nIters):
        '''Carries out one instantaneous Q_learning training step using the target Q network
        Parameters:
            X: array storing the populations of each bacteria
            C: array contatining the concentrations of each rate limiting nutrient
            C0: the concentration of the common carbon source at this time point
            t: the current time
            visited_states: array to keep track of which states have been visited
            explore_rate: the current explore rate
            step_size: current Q_learning step size
            Q_params: learning parameters
            ode_params: parameters for the numerical solver odeint
        Returns:
            X1: populations at next timestep
            C1: concentrations of auxotrophc nutrient at each time step
            C0: concentration of the carbon source at next time point
            xSol: full next bit of the populations solutions, including the skipped frames
            reward
            nIters: the number of training iterations since last update of the target network
        '''

        A, num_species, num_controlled_species, num_x_states, x_bounds, num_Cin_states, Cin_bounds, gamma = Q_params

        state = state_to_one_hot(X, num_species, x_bounds, num_x_states) # flatten to a vector

        if np.random.rand(1) < explore_rate:
            action = np.random.randint(num_Cin_states**num_controlled_species)
        else:
            action = np.array(sess.run(self.predict, feed_dict= {self.inputs:state}))
        visited_states += state

        # get new state and reward
        S = np.append(X, C)
        S = np.append(S, C0)

        # convert chosen action index to a concentration
        Cin = action_to_state(action, num_controlled_species, num_Cin_states, Cin_bounds) # take out this line to remove the effect of the algorithm

        if num_species - num_controlled_species == 1: # hacky way to check for auxotroph system
            Cin = np.append([1], Cin)

        # get next time step
        time_diff = 4 # frame skipping
        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin,A,ode_params, num_species))[1:]

        # extract information from sol
        xSol = sol[:, 0:num_species]
        X1 = sol[-1, :num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]

        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))

        # turn new state into one hot vector
        state1 = state_to_one_hot(X1, num_species, x_bounds, num_x_states) # flatten to a vector

        reward = self.reward(X1)

        self.experience_buffer.add([state, action, reward, state1])

        # sample from buffer
        max_time = 1
        batch_size = 10
        experience_sample = self.experience_buffer.sample(batch_size, max_time)

        # extract from buffer
        states, actions, rewards, state1s = [],[],[],[]
        for experience_trace in experience_sample:
            for experience in experience_trace:
                states.append(create_one_hot(num_x_states**num_species, experience[0])[0])
                actions.append(experience[1])
                rewards.append(experience[2])
                state1s.append(create_one_hot(num_x_states**num_species, experience[1])[0])

        states, actions, rewards, state1s = np.array(states), np.array(actions), np.array(rewards), np.array(state1s)


        Qs = sess.run(self.predQ,feed_dict = {self.inputs: states})
        assert not np.all(np.isnan(Qs)), 'Nan found in output, network probably unstable'

        # get Q1 values for experience buffer
        Q1s = sess.run(self.target_predQ, feed_dict = {self.target_inputs: state1s})

        maxQ1s = np.max(Q1s, axis = 1)

        # build targets for all Qs in experience buffer
        for i in range(Qs.shape[0]):
            Qs[i, actions[i]] += step_size*(rewards[i] + gamma*maxQ1s[i] - Qs[i,actions[i]])

        # train network based on target and predicted Q values
        sess.run([self.updateModel], feed_dict = {self.inputs: states, self.TD_target:Qs})

        #update target network
        target_update_freq = 100
        tau = 1. # proportion of primary network used in update
        if nIters % target_update_freq == 0:
            # update target network to the primary networks weights
            primary_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'primary')
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target')

            sess.run([var_target.assign(var_target * (1-tau) + var_primary*tau) for var_target, var_primary in zip(target_vars, primary_vars)])


        return X1, C1, C01, xSol, reward, Qs, visited_states

    def simple_reward(self,X):
        '''
        Simple reward funtion based on the populations of each bacterial species
        Parameters:
            X: array of all population levels
        Returns:
            reward: the reward recieved
        '''
        '''
        if (100 < X[0] < 400) and (400 < X[1] < 700):
            reward =  1
        else:
            reward = - 1
        if X[0] < 1/100 or X[1] < 1/100:
            reward = -5
        '''

        if all(x > 1.5 for x in X):
            reward = 1
        else:
            reward = - 1

        if any(x < 1/1000 for x in X):
            reward = - 10
        return reward
