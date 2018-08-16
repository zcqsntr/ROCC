import pytest
import numpy as np
from numpy.testing import assert_array_equal
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'CBcurl'))

from utilities import *

"""test state to bucket"""
one_dim = [([0], [0, 10], 10, [0]), ([11], [0, 10], 10, [9]) , ([5.5], [0, 10], 10, [5])] # first last and middle
two_dim = [([0, 0], [0, 10], 10, [0, 0]), ([11, 11], [0, 10], 10, [9, 9]) , ([0, 5.5], [0, 10], 10, [0,5]), ([5.5, 0], [0, 10], 10, [5,0]), ([5.5, 5.5], [0, 10], 10, [5,5])]
three_dim = [([0, 0, 0], [0, 10], 10, [0, 0, 0]), ([11, 11, 11], [0, 10], 10, [9, 9, 9]) , ([5.5, 5.5, 5.5], [0, 10], 10, [5, 5, 5])]
state_to_bucket_data = one_dim + two_dim + three_dim
@pytest.mark.parametrize("state, x_bounds, num_x_states, expected", state_to_bucket_data)
def test_state_to_bucket(state, x_bounds, num_x_states, expected):
    calculated = state_to_bucket(state, x_bounds, num_x_states)
    assert np.allclose(calculated, expected)


"""test action to state"""
one_dim = [(0, 1, 2, [0,1], [0]), (1, 1, 2, [0,1], [1])]
two_dim = [(0, 2, 2, [0,1], [0,0]), (1, 2, 2, [0,1], [0, 1]), (2, 2, 2, [0,1], [1, 0]), (3, 2, 2, [0,1], [1, 1])]
three_dim = [(0, 3, 2, [0,1], [0,0, 0]), (1, 3, 2, [0,1], [0,0, 1]), (2, 3, 2, [0,1], [0, 1, 0]), (3, 3, 2, [0,1], [0, 1, 1])\
           , (4, 3, 2, [0,1], [1, 0, 0]), (5, 3, 2, [0,1], [1, 0, 1]), (6, 3, 2, [0,1], [1, 1, 0]),  (7, 3, 2, [0,1], [1, 1, 1])]
action_to_state_data = one_dim + two_dim + three_dim
@pytest.mark.parametrize("action, num_species, num_Cin_states, Cin_bounds, expected", action_to_state_data)
def test_action_to_state(action, num_species, num_Cin_states, Cin_bounds, expected):
    calculated = action_to_state(action, num_species, num_Cin_states, Cin_bounds)
    assert np.allclose(calculated, expected)


"""test state_to_one_hot"""
one_dim = [([0], 1, [0,10], 10, create_one_hot(10,0)), ([11], 1, [0,10], 10, create_one_hot(10,9)), ([5.5], 1, [0,10], 10, create_one_hot(10,5))]
two_dim = [([0, 0], 2, [0, 10], 10, create_one_hot(100,0)), ([11, 11], 2, [0, 10], 10, create_one_hot(100,99)), ([5.5, 5.5], 2,  [0, 10], 10, create_one_hot(100,55))]
three_dim = [([0, 0, 0], 3,  [0, 10], 10, create_one_hot(1000,0)), ([11, 11, 11], 3,  [0, 10], 10, create_one_hot(1000,999)), ([5.5, 5.5, 5.5], 3,  [0, 10], 10, create_one_hot(1000,555))]

action_to_state_data = one_dim + two_dim + three_dim

@pytest.mark.parametrize("state, num_species, x_bounds, num_x_states, expected", action_to_state_data)
def test_state_to_one_hot(state, num_species, x_bounds, num_x_states, expected):
    calculated = state_to_one_hot(state, num_species, x_bounds, num_x_states)
    assert np.allclose(calculated, expected, atol = 0.01)


"""test epsilon greedy"""

# explore_rate = 0
zero = [(0, create_one_hot(4, i%4), i%4) for i in range(100)]

# explore_rate = 1
one = np.array([epsilon_greedy(1,create_one_hot(4, i%4)) for i in range(1000)])

# explore_rate = 0.5
half = np.array([epsilon_greedy(0.5,create_one_hot(4, 1)) for i in range(1000)])

epsilon_greedy_data = zero
@pytest.mark.parametrize("explore_rate, Q_values, expected", epsilon_greedy_data)
def test_epsilon_greedy_zero(explore_rate, Q_values, expected):
    calculated = epsilon_greedy(explore_rate, Q_values)
    assert np.array_equal(calculated, expected)

def test_epsilon_greedy():
    _, one_counts = np.unique(one, return_counts = True)
    assert all([np.allclose(count, [250], atol = 50) for count in one_counts])

    _, half_counts = np.unique(half, return_counts = True)
    print(half_counts)
    assert np.allclose(half_counts[1], [500], atol = 200)

"""test experience_buffer"""

# test adding to buffer and max size
def test_buffer_500():
    buffer = ExperienceBuffer()
    for i in range(500):
        experience = [np.array([[0,0,1]]), 1, 1, np.array([[1, 0, 0]])] # need to make new experience each time
        buffer.add(experience)
    assert len(buffer.buffer) == 500

def test_buffer_1000():
    buffer = ExperienceBuffer()
    for i in range(1100):
        experience = [np.array([[0,0,1]]), 1, 1, np.array([[1, 0, 0]])]
        buffer.add(experience)
    assert len(buffer.buffer) == 1000

def test_buffer_sampling():
    buffer = ExperienceBuffer()
    for i in range(1000):
        experience = [np.array([[0,0,1]]), 1, 1, np.array([[1, 0, 0]])]
        buffer.add(experience)
    # test sample
    sample1 = buffer.sample(1,1)
    sample2 = buffer.sample(5, 1)
    assert len(sample1) == 1
    assert len(sample2) == 5

# test error throwing
def test_buffer_error():
    buffer = ExperienceBuffer()
    for i in range(3):
        experience = [np.array([[0,0,1]]), 1, 1, np.array([[1, 0, 0]])]
        buffer.add(experience)

    with pytest.raises(ValueError):
        sample = buffer.sample(1, 5)


"""test get explore rate"""
# test max value is one and min value MIN_EXPORE_RATE
explore_rate_data = [(0, 0, 1, 1), (999999, 0, 1, 0)]

@pytest.mark.parametrize("t, MIN_EXPLORE_RATE, divisor, expected_output", explore_rate_data)
def test_get_explore_rate(t, MIN_EXPLORE_RATE, divisor, expected_output):
    assert math.isclose(get_explore_rate(t, MIN_EXPLORE_RATE, divisor),expected_output, rel_tol = 0.0001)


"""test get learning rate """
# test max value is MAX_LEARNING_RATE and min value is MIN_LEARNING_RATE
learning_rate_data = [(0, 0.05, 0.5, 1, 0.5), (999999, 0.05, 0.5, 1 , 0.05)]

@pytest.mark.parametrize("t, MIN_LEARNING_RATE, MAX_LEARNING_RATE, divisor, expected_output", learning_rate_data)
def test_get_learning_rate(t, MIN_LEARNING_RATE, MAX_LEARNING_RATE, divisor, expected_output):
    assert math.isclose(get_learning_rate(t, MIN_LEARNING_RATE, MAX_LEARNING_RATE, divisor),expected_output, rel_tol = 0.0001)
