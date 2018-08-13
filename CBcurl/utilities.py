import numpy as np
import math


"""
General functions and classes used by all agents.
"""


def sdot(S, t, Cin, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    '''
    Calculates and returns derivatives for the numerical solver odeint
    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        dsol: array of the derivatives for all state variables
    '''
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])


    C0in, q, y, y3, Rmax, Km, Km3 = params

    R = monod(C, C0, Rmax, Km, Km3)

    Cin = Cin[:num_species]

    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC)
    dsol = np.append(dsol, dC0)
    return tuple(dsol)


# calculates r as a function of the concentration of the rate limiting nutrient
def monod(C, C0, Rmax, Km, Km0):
    '''
    Calculates the growth rate based on the monod equation
    Parameters:
        C: the concetrations of the auxotrophic nutrients for each bacterial population
        C0: concentration of the common carbon source
        Rmax: array of the maximum growth rates for each bacteria
        Km: array of the saturation constants for each auxotrophic nutrient
        Km0: array of the saturation constant for the common carbon source for each bacterial species
    '''

    C = np.array(C)
    Rmax = np.array(Rmax)
    Km = np.array(Km)
    C0 = np.array(C0)
    Km0 = np.array(Km0)

    growth_rate = ((Rmax*C)/ (Km + C)) * (C0/ (Km0 + C0))

    return growth_rate



def get_explore_rate(episode, MIN_EXPLORE_RATE, denominator): # increase denominator to explore for longer
    '''
    Calculates the logarithmically decreasing explore rate
    Parameters:
        episode: the current episode
        MIN_EXPLORE_RATE: the minimum possible explore_rate
        denominator: controls the rate of decay of the explore rate
    Returns:
        explore_rate: the chance the agent will take a random action
    '''
    if not 0 <= MIN_EXPLORE_RATE <= 1:
        raise ValueError("MIN_EXPLORE_RATE needs to be bewteen 0 and 1")

    if not 0 < denominator:
        raise ValueError("denominator needs to be above 0")

    explore_rate = max(MIN_EXPLORE_RATE, min(1.0,1.0 - math.log10((episode+1)/(denominator))))
    return explore_rate

def get_learning_rate(episode, MIN_LEARNING_RATE,  MAX_LEARNING_RATE, denominator):
    '''
    Calculates the logarithmically decreasing explore rate
    Parameters:
        episode: the current episode
        MIN_LEARNING_RATE: the minimum possible step size
        MAX_LEARNING_RATE: maximum step size
        denominator: controls the rate of decay of the step size
    Returns:
        step_size: the Q-learning step size
    '''

    if not 0 <= MIN_LEARNING_RATE <= 1:
        raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

    if not 0 <= MAX_LEARNING_RATE <= 1:
        raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

    if not 0 < denominator:
        raise ValueError("denominator needs to be above 0")

    step_size = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, 1.0 - math.log10((episode+1)/denominator)))
    return step_size

def state_to_bucket(state, x_bounds, num_x_states):
    '''
    Takes a state vactor and descritises it, returns the descritised bucket each value is in
    Parameters:
        state: current state of the system
        x__bounds: list of the upper and lower limit of the states the agent can distinguish
        num_x_states: number of discrete state the agent can see
    Returns:
        bucket_indice: the indexes of the bucket each state
    '''

    if num_x_states < 0 or not isinstance(num_x_states, int):
        raise ValueError("num_x_states needs to be a positive integer")

    if not all(s > -0.001 for s in state):
        raise ValueError("state needs to be positive")
    if not all(x >= 0 for x in x_bounds):
        raise ValueError("x_bounds needs to be >= 0")

    bucket_indice = []

    for x in state:
        if x <= x_bounds[0]:
            bucket_index = 0
        elif x >= x_bounds[1]:
            bucket_index = num_x_states - 1
        else:
            bound_width  = x_bounds[1] - x_bounds[0]
            offset = (num_x_states - 1) * x_bounds[0] / bound_width
            scaling = (num_x_states-1)/bound_width
            bucket_index = int(round(scaling*x - offset))

        bucket_indice.append(bucket_index)

    return np.array(bucket_indice)


def action_to_state(action, num_species, num_Cin_states, Cin_bounds):
    '''
    Takes a discrete action index and returns the corresponding continuous state vector
    Paremeters:
        action: the descrete action
        num_species: the number of bacterial populations
        num_Cin_states: the number of action states the agent can choose from for each species
        Cin_bounds: list of the upper and lower bounds of the Cin states that can be chosen
    Returns:
        state: the continuous Cin concentrations correspoding to the chosen action
    '''
    buckets = np.unravel_index(action, [num_Cin_states]*num_species)
    state = []
    for r in buckets:
        state.append(Cin_bounds[0] + r*(Cin_bounds[1]-Cin_bounds[0])/(num_Cin_states-1))
    state = np.array(state).reshape(num_species,)
    return state

def state_to_one_hot(state, num_species, x_bounds, num_x_states):
    '''
    Converts a continuous state vector to one hot vector
    Parameters:
        state: continuous state
        num_species: number of bacterial populations
        x_bounds: list of the lower and upper population bounds that the agent can see
        num_x_states: the number of population states the agent can distinguish
    Returns:
        one_hot_state: the converted state
    '''
    buckets = tuple(np.array(state_to_bucket(state, x_bounds, num_x_states))) #change to tuple as numpys indexing with arrays is not the behaviour we want
    one_hot_state = np.zeros(tuple([num_x_states for _ in range(num_species)]))

    one_hot_state[buckets] = 1 # set one

    one_hot_state = one_hot_state.reshape(1,num_x_states**num_species) # flatten to a vector

    return one_hot_state

def epsilon_greedy(explore_rate, Q_values):
    '''
    Chooses an action based on the epsilon greedy policy
    Parameters:
        explore_rate: the chance the agent will choose a random action
        Q_values: the Q_values for each action
    Returns:
        action: the chosen action
    '''
    if not 0 <= explore_rate <= 1:
        raise ValueError("Invalid explore rate (" + str(explore_rate)+ "), must be between zero and 1")
    if np.random.rand(1) < explore_rate:
        action = np.random.randint(Q_values.shape[1])
    else:
        action = np.argmax(Q_values)
    return action

'''
def get_temperature(episode, MIN_TEMP, MAX_TEMP, denominator): # increase denominator to explore for longer

    Calculates the logarithmically decreasing temperature for softmax
    Parameters:
        episode: the current episode
        MIN_TEMP: the minimum possible step size
        MAX_TEMP: maximum step size
        denominator: controls the rate of decay of the step size
    Returns:
        step_size: the Q-learning step size

    return max(MIN_TEMP, min(MAX_TEMP,MAX_TEMP*(1.0 - math.log10((episode+1)/(denominator)))))
'''

'''
def softmax_selection(temperature, Q_values):

    Chooses an action based on the softmax policy
    Parameters:
        temperature: controls the width of the distribution
        Q_values: the Q_values for each action
    Returns:
        action: the chosen action

    if np.random.rand(1) < explore_rate:
        action = np.random.randint(Q_values.shape[1])
    else:

        np.pdist = np.exp((Q_values-np.max(Q_values))/temperature) / np.sum(np.exp((Q_values-np.max(Q_values))/temperature))
        action = np.random.choice(allQ.shape[1], p = pdist.reshape(pdist.shape[1]))

    return action, pdist
'''

def add_noise(X, error):
    '''
    Adds normally distributed noise to a state vector
    Parameters:
        X: state vector
        error: the maximum amount of error added
    Returns:
        noisey_X: the state with added noise
    '''

    if error < 0:
        raise ValueError("Error needs to be positive")
    noisey_X  = X + np.random.normal() * error *X + np.random.normal()*error
    return noisey_X

def convert_to_numpy(param_dict):
    '''
    Takes a parameter dictionary and converts the required parameters into numpy arrays
    Parameters:
        param_dict: the parameter dictionary
    Returns:
        param_dict: the converted parameter dictionary
    '''
    param_dict['ode_params'][1], param_dict['ode_params'][2], param_dict['ode_params'][3] = np.array(param_dict['ode_params'][1]), np.array(param_dict['ode_params'][2]), np.array(param_dict['ode_params'][3])

    param_dict['Q_params'][0] = np.array(param_dict['Q_params'][0])
    param_dict['Q_params'][8] = np.array(param_dict['Q_params'][8])
    param_dict['Q_params'][9] = np.array(param_dict['Q_params'][9])
    return param_dict

def validate_param_dict(param_dict):
    '''
    Performs input validation on the parameter dictionary supplied by the user.
    Parameters:
        param_dict: the parameter dictionary
    '''
    ode_params = param_dict['ode_params']

    if ode_params[0] <= 0:
        raise ValueError("C0in needs to be positive")
    if ode_params[1] <= 0:
        raise ValueError("q needs to be positive")
    if not all(y > 0 for y in ode_params[2]) or not all(y3 > 0 for y3 in ode_params[3]):
        raise ValueError("all bacterial yield constants need to be positive")
    if not all(Rmax > 0 for Rmax in ode_params[4]):
        raise ValueError("all maximum growth rates need to be positive")
    if not all(Km >= 0 for Km in ode_params[5]) or not all(Km3 >= 0 for Km3 in ode_params[6]):
        raise ValueError("all saturation constants need to be positive")

    Q_params = param_dict['Q_params']
    num_species = Q_params[1]
    if num_species < 0 or not isinstance(num_species, int):
        raise ValueError("num_species needs to be a positive integer")
    A = Q_params[0]
    if len(A) != num_species or not all(len(row) == num_species for row in A):
        raise ValueError("A needs to be a square matrix with shape (num_species, num_species)")

    if Q_params[2] > num_species or Q_params[2] < 0 or not isinstance(Q_params[2], int):
        raise ValueError("num_controlled_species needs to be a positive integer <= to num_species")
    if Q_params[3] < 0 or not isinstance(Q_params[3], int):
        raise ValueError("num_x_states needs to be a positive integer")


    if len(Q_params[4]) != 2 or Q_params[4][0] < 0 or Q_params[4][0] >= Q_params[4][1]:
        raise ValueError("x_bounds needs to be a list with two values in ascending order")
    if Q_params[5] < 0 or not isinstance(num_species, int):
        raise ValueError("num_C0_states needs to be a positive integer")
    if len(Q_params[6]) != 2 or Q_params[6][0] < 0 or Q_params[6][0] >= Q_params[6][1]:
        raise ValueError("C0_bounds needs to be a list with two values in ascending order")
    if not 0 < Q_params[7] < 1:
        raise ValueError("discount factor needs to be between zero and one")

    if not all(x > 0 for x in Q_params[8]):
        raise ValueError("all initial populations need to be positive")
    if not all(c > 0 for c in Q_params[9]):
        raise ValueError("all initial concentrations need to be positive")
    if Q_params[10] < 0:
        raise ValueError("initial C0 needs to be positive")


    train_params = param_dict['train_params']
    if train_params[0] <= 0 or not isinstance(train_params[0], int):
        raise ValueError("num_episodes needs to be a positive integer")
    if train_params[1] <= 0 or not isinstance(train_params[1], int) or train_params[1] > train_params[0]:
        raise ValueError("test_freq needs to be a positive integer and smaller than num_episodes")
    if train_params[2] <= 0:
        raise ValueError("explore_denom needs to be a positive number")
    if train_params[3] <= 0:
        raise ValueError("train_denom needs to be a positive number")
    if train_params[4] <= 0 or not isinstance(train_params[4], int):
        raise ValueError("T_MAX needs to be a positive integer")
    if not 0 <= train_params[5] <= 1:
        raise ValueError("MIN_STEP_SIZE needs to be between zero and one")
    if not 0 <= train_params[6] <= 1:
        raise ValueError("MAX_STEP_SIZE needs to be between zero and one")
    if not 0 <= train_params[7] <= 1:
        raise ValueError("MIN_EXPLORE_RATE needs to be between zero and one")
    if not all(isinstance(l, int) and l > 0 for l in train_params[9]):
        raise ValueError("layers sizes needs to be a list of positive integers")

    noise_params = param_dict['noise_params']

    if not isinstance(noise_params[0], bool):
        raise TypeError("noise needs to be a boolean")
    if not noise_params[1] >= 0:
        raise ValueError("'error needs to be greater than zero'")





class ExperienceBuffer():
    '''
    Class to handle the management of the QDN storage buffer, stores experience in the form [state, action, reward, next_state]
    '''
    def __init__(self, buffer_size = 1000):
        '''
        Parameters:
            buffer_size: number of experiences that can be stored
        '''

        if buffer_size <= 0 or not isinstance(buffer_size, int):
            raise ValueError("Buffer size must be a positive integer")
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        '''
        Adds a peice of experience to the buffer and removes the oldest experince if the buffer is full
        Parameters:
            experience: the new experience to be added, in the format [state, action, reward, state1]
        '''

        # seems to be working

        if len(experience) != 4:
            raise ValueError("Experience must be length 4, of the for [state, action, reward, state1]")
        if len(self.buffer) == self.buffer_size:
            self.buffer = self.buffer[1:, :]

        # convert from 1 hot to indices for storage into buffer
        experience[0] = np.argwhere(np.array(experience[0]))[0][1]
        experience[3] = np.argwhere(np.array(experience[3]))[0][1]
        experience = np.array(experience).reshape(1,4)
        if self.buffer == []:
            self.buffer = experience
        else:
            self.buffer = np.append(self.buffer, experience, axis = 0)



    def sample(self, batch_size, max_time):
        '''
        Randomly samples the experience buffer
        Parameters:
            batch_size: the number of experience traces to sample
            max_time: the length of each experience trace (larger than one used for recurrent network)
        Returns:
            sample: the sampled experience
        '''
        # seems to be working
        if len(self.buffer) < max_time:
            raise ValueError("Attempted sample is longer than current buffer size (" + str(max_time) + ">" + str(len(self.buffer))+ ")")

        starts = np.random.randint(0, len(self.buffer) - max_time+1, size = (batch_size)) # start of experience traces

        sample = []
        for start in starts:
            sample.append(self.buffer[start:start+max_time]) # use += to prevent extra dimension

        return np.array(sample)


def create_one_hot(size, index):
    zeros = np.zeros(size)
    zeros[index] = 1
    return zeros.reshape(1, size)
