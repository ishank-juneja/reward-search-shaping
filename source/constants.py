# Parameters needed across all files


# Define constants
grid_shape = (6, 6)
# Start location
start = (0, 5)
# Fat position in grid
fat_loc = (5, 0)
# Protein location in grid
pro_loc = (0, 5)
# discount factor, < 1 avoids blow up
gamma = 0.99
# Action space: Up, Down, Right, Left, Eat
nactions = 5
# Decay probability of satiatedness
prob_fat = 0.1
prob_pro = 0.1
# Declare constants
pro_fit = 0.01
fat_fit = 0.01
both_fit = 1
none_fit = 0
