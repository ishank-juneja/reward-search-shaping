import numpy as np
from balanced_diet_gridworld import World
from policy import eps_greedy
from scipy.stats import bernoulli
import sys


# Define constants
grid_shape = (6, 6)
# Start location
start = (0, 5)
# Fat position in grid
fat_loc = (5, 0)
# Protein location in grid
pro_loc = (0, 5)
# Init the grid world object, manages state transitions internally
environment = World(grid_shape, start, fat_loc, pro_loc)
# Algorithm parameters
alpha = 0.5 # Learning rate
# Number of time steps in continuing task
max_steps = 150000
# discount factor, < 1 avoids blow up
gamma = 0.99
# Action space: Up, Down, Right, Left, Eat
nactions = 5
# Decay probability of satiatedness
prob_fat = 0.1
prob_pro = 0.1
# Random seed for above randomness
rs = 1
# Reward array, follows same coding order as
# the Q[s][a] matrix
# State coding is 0 --> Un sat in both, 1 --> sat in fat only,
# 2 --> sat in pro only, 3 --> sat in both so 4 possible states
rew_array = np.array([-0.1, -0.05, -0.05, 1])



if __name__ == '__main__':
	# Declare array to hold Q_values Q(s, a), init with zeros
	# State coding is 0 --> Un sat in both, 1 --> sat in fat only,
	# 2 --> sat in pro only, 3 --> sat in both so 4 possible states
	Q_values = np.zeros((grid_shape[0], grid_shape[1], 4, nactions))
	# Initialise state as seen by agent
	row = start[0]
	col = start[1]
	sat_state = 0
	# Intialise cummulative fitness
	cum_fit = 0
	# Pre-generate sequence of bernoulli random numbers
	X1 = bernoulli(prob_fat)
	rands_fat = X1.rvs(max_steps, random_state=rs+1)
	# Use a different random seed for un-correlatedness
	X2 = bernoulli(prob_pro)
	rands_pro = X2.rvs(max_steps, random_state=rs)
	for t in range(max_steps):
		# Get epsilon-greedy action wrt current state
		action = eps_greedy(Q_values, row, col, sat_state, nactions)
		# Take action, get next state and incremental fitness that will be gained on moving to that state
		fitness, (next_row, next_col, next_sat_state) = \
			environment.update_state(action, bool(rands_fat[t]), bool(rands_pro[t]))
		# Get reward associated with making this transition (to next state)
		rew = rew_array[next_sat_state]
		# rew = fitness
		#print(fitness)
		#print(row, col, sat_state)
		#print(next_row, next_col, next_sat_state)
		#print(row, col)
		#if (row, col) == pro_loc:
			#print("Here")
		if next_sat_state == 3:
			print("Success")
		# Make Q-learning(0) update to Q values
		Q_values[row][col][sat_state][action] += \
			alpha * (rew + gamma * np.max(Q_values[next_row][next_col][next_sat_state]) - Q_values[row][col][sat_state][action])
		(row, col, sat_state) = (next_row, next_col, next_sat_state)
		cum_fit += fitness

	print(cum_fit)
