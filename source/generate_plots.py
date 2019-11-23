from matplotlib import pyplot as plt


# Input files
# Reward schemes to be experimented on, naming convention of paper followed
# Key: baseline --> reward = fitness increment
# simple_fitnees --> 0 reward for unsatiated in both, some arbitrary positive reward
# for being satiated in a single food group and some other arbitrary positive reward
# for being satiated with respect to both food groups
# Other rewards --> No restrictions except for the fact that the reward associated
# with the 01 and 10 (binary) states should be the same
baselines_schemes = "../results/baseline_scheme.npy"
baseline_data = "../results/baseline_data_points.npy"
other_schemes = "../results/baseline_scheme.npy"
other_data = "../results/baseline_data_points.npy"


# Plot 1 - Standard Cummulative Fitness vs Horizon plots
# Specifically plot Baseline, Best Simple fitness based - No negative penalty allowed

