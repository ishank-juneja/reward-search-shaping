import numpy as np
from balanced_diet_gridworld import World
from policy import eps_greedy
from scipy.stats import bernoulli
from reward_scheme import get_discretization
from constants import *


# Init the grid world object, manages state transitions internally
environment = World(grid_shape, start, fat_loc, pro_loc)
# Algorithm parameters
alpha = 0.5 # Learning rate
# Number of time steps in continuing task
max_steps = 75000
# Number of runs to average over per reward scheme
nruns = 10
# Take geometrically spaced steps
time_steps = np.geomspace(100, max_steps - 1, 100, dtype=int)
# Pre-declare array to hold datapoints for a single reward scheme run
datapoints = np.zeros((nruns, 100))
# Output location
PATH_OUT = "../results/"
# Reward schemes to be experimented on, naming convention of paper followed
# Key: baseline --> reward = fitness increment
# Other rewards --> Reward for 00 state hungry wrt both is fixed small negative value
# reward for other 2 is unconstrained but between -1 and 1
reward_schemes = ["baseline", "other"]
# Reward array, follows same coding order as
# the Q[s][a] matrix
# State coding is 0 --> Un sat in both, 1 --> sat in fat only,
# 2 --> sat in pro only, 3 --> sat in both so 4 possible states


# Generate results for all reward schemes
for scheme in reward_schemes:
	# List to hold all reward values tested under current scheme
	rewards = []
	# List to hold corresponding cummulative fitness value data-points
	data_points = []
	# List to hold steps spent breakdown across reward schemes
	step_breakdown = []
	# List to hold Q values associated with each reward scheme
	Q_values_list = []
	# Get reward space discretization specific to scheme
	# We will always use the same reward for states 1 (01) and 2 (10)
	disc_0, disc1_2, disc_3 = get_discretization(scheme)
	# Get output summary file name
	file_out = PATH_OUT + scheme
	nschemes = len(disc_0)*len(disc1_2)*len(disc_3)
	# Run over all possible rewards under current reward scheme
	scheme_num = 0
	for rew0 in disc_0:
		for rew1_2 in disc1_2:
			for rew3 in disc_3:
				rew_array = np.array([rew0, rew1_2, rew1_2, rew3])
				print("Processing Reward scheme {0} of {1} for the scheme family {2} ".format(scheme_num+1, nschemes, scheme))
				scheme_num += 1
				# Array to hold number of steps spent in each state (also averaged)
				steps_per_state = np.zeros(4)
				# Average cummulative fitness values over 10 runs
				for rs in range(nruns):
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
					ctr = 0
					print("Performing run {0} of {1} for {2} steps .... ".format(rs+1, nruns, max_steps))
					for t in range(max_steps):
						# Get epsilon-greedy action wrt current state
						action = eps_greedy(Q_values, row, col, sat_state, nactions)
						steps_per_state[sat_state] += 1
						# Take action, get next state and incremental fitness that will be gained on moving to that state
						fitness, (next_row, next_col, next_sat_state) = \
							environment.update_state(action, bool(rands_fat[t]), bool(rands_pro[t]))
						# Get reward associated with making this transition (to next state)
						rew = rew_array[next_sat_state]
						# Make Q-learning(0) update to Q values
						Q_values[row][col][sat_state][action] += \
							alpha * (rew + gamma * np.max(Q_values[next_row][next_col][next_sat_state]) - Q_values[row][col][sat_state][action])
						(row, col, sat_state) = (next_row, next_col, next_sat_state)
						cum_fit += fitness
						# If current time stamp present in list
						# Record the data point
						if t in time_steps:
							datapoints[rs][ctr] = cum_fit
							# Go to next index
							ctr += 1
				rewards.append(rew_array)
				data_points.append(np.mean(datapoints, axis=0))
				step_breakdown.append(steps_per_state/nruns)
				# Arbitarlily append the Q value obtained in the last (10th run)
				Q_values_list.append(Q_values)
				# print(rewards)
				# print(data_points)
				# print(step_breakdown)
				# print(Q_values_list)
	np.save(file_out + "_scheme.npy", rewards)
	np.save(file_out + "_data_points.npy", data_points)
	np.save(file_out + "_steps_per_state.npy", step_breakdown)
	np.save(file_out + "_Q_learned.npy", Q_values_list)
