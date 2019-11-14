from random import random, randint
import numpy as np


eps = 0.1   # Exploration rate


def eps_greedy(Q_values, cur_row, cur_col, sat_state, nactions):
    # Observe unifrom random variable between 0 and 1
    rand = random()
    if rand < eps:
        action = randint(0, nactions - 1)
    else:
        action = np.argmax(Q_values[cur_row][cur_col][sat_state])
    return action
