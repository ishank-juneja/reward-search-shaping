from matplotlib import pyplot as plt
import numpy as np


# Input files
# Reward schemes to be experimented on, naming convention of paper followed
# Key: baseline --> reward = fitness increment
# simple_fitnees --> 0 reward for unsatiated in both, some arbitrary positive reward
# for being satiated in a single food group and some other arbitrary positive reward
# for being satiated with respect to both food groups
# Other rewards --> No restrictions except for the fact that the reward associated
# with the 01 and 10 (binary) states should be the same
baselines_schemes_path = "../results/baseline_scheme.npy"
baselines_schemes = np.load(baselines_schemes_path)
baseline_data_path = "../results/baseline_data_points.npy"
baseline_data = np.load(baseline_data_path)
baseline_Qval_path = "../results/baseline_Q_learned.npy"
baseline_Qval = np.load(baseline_Qval_path)
baseline_steps_per_state_path = "../results/baseline_steps_per_state.npy"
baseline_steps_per_state = np.load(baseline_steps_per_state_path)
other_schemes_path = "../results/other_scheme.npy"
other_schemes = np.load(other_schemes_path)
other_data_path = "../results/other_data_points.npy"
other_data = np.load(other_data_path)
other_Qval_path = "../results/other_Q_learned.npy"
other_Qval = np.load(other_Qval_path)
other_steps_per_state_path = "../results/other_steps_per_state.npy"
other_steps_per_state = np.load(other_steps_per_state_path)
# Take geometrically spaced steps
max_steps = 75000
time_steps = np.geomspace(100, max_steps - 1, 100, dtype=int)


# Plot 1 - Standard Cummulative Fitness vs Horizon plots
# Special lines to be included are
# 1. Baseline plot, 2. Best Simple fitness reward (No negative rewards, and 0 for Hungry in both)
# 3. Generally just all the rewards with very thin lines, 4. Globally optimal reward
plt.figure(figsize=(6, 6))
plt.title("Fitness Growth for Reward schemes")
plt.xlabel("Time Step in Agent Lifetime")
plt.ylabel("Cumulative Fitness")

# Best Reward plot, get index of best reward by looking at terminal cumulative fitness
best_rew = np.argmax(other_data[:, -1])
# Drop a few samples for smoother plot
plt.plot(time_steps[::2], other_data[best_rew][::2], linewidth=2, linestyle='-', color='k')
# plt.plot(time_steps, other_data[best_rew], linewidth=2, linestyle='-', color='k')

# Best Fitness based reward, again based on terminal cumulative reward
max_fit = 0
max_index = 0
for i in range(other_data.shape[0]):
    if other_schemes[i][0] == 0 and other_schemes[i][1] > 0 and other_schemes[i][3] > 0 and other_data[i][-1] > max_fit:
        max_fit = other_data[i][-1]
        max_index = i
plt.plot(time_steps, other_data[max_index], linewidth=1, linestyle='--', color='k')

#  Baseline plot
# plt.plot(time_steps, baseline_data[0], linewidth=1, linestyle=':', color='k')

# All other plots
for i in range(other_data.shape[0]):
    if i != max_index and i != best_rew:
        plt.plot(time_steps, other_data[i], linewidth=0.1, linestyle='-', color='k')


plt.legend(["Best-reward", "Best-Fitness-based/Baseline", "other-rewards"])
#plt.savefig("../results/cum_fitness_vs_time_steps.png")
plt.close()

# Print summary results
print("The best reward index is {0}".format(best_rew))
print("The optimal reward values obtained from search are: Reward for unsatiated {0}, Reward for unisatiated ins\n "
      .format(other_schemes[best_rew][0], other_schemes[best_rew][1], other_schemes[best_rew][3]))


# Plot 2 - Fitness vs. Partial Satiatedness penalty for fixed hungry reward of 0
plt.figure(figsize=(6, 6))
plt.title("Relation between fitness and unbalanced diet penalty")
plt.xlabel("Unbalanced Diet Penalty Value")
plt.ylabel("Cumulative Fitness")
plt.scatter(other_schemes[:, 3] - other_schemes[:, 1], other_data[:, -1], color='k', s=3)
plt.scatter(other_schemes[161, 3] - other_schemes[161, 1], other_data[161, -1], color='g', s=15, marker='v')
plt.scatter(baselines_schemes[:, 3] - baselines_schemes[:, 1], baseline_data[:, -1], color='red', s=15, marker='v')
plt.text(other_schemes[161, 3] - other_schemes[161, 1] - 0.2, other_data[161, -1] - 100, "Optimal Reward", ma='center', size = 7)
plt.text(baselines_schemes[:, 3] - baselines_schemes[:, 1] + 0.02, baseline_data[:, -1] + 50, "Baseline", ma='center', size=7)
plt.axvline(baselines_schemes[:, 3] - baselines_schemes[:, 1], linewidth=0.5, linestyle='--', color='k', alpha=0.5)
plt.axhline(baseline_data[:, -1], linewidth=0.5, linestyle='--', color='k', alpha=0.5)
#plt.savefig("../results/cumm_fitness_vs_penalty_scatter.png")
plt.close()


# Plot 3 - Time for which unsatiated/Hungry
plt.figure(figsize=(6, 6))
plt.ylabel("Fraction of time spent completely Hungry")
plt.xlabel("Reward for being satiated in single state")
plt.title("Fraction of time spent Hungry as a function of reward scheme")
for i in range(other_data.shape[0]):
    if i != best_rew:
        plt.scatter(other_schemes[:, 2], other_steps_per_state[:, 0]/75000, color='k', s=3)
plt.scatter(other_schemes[best_rew, 2], other_steps_per_state[best_rew, 0]/75000, color='g', s=25, marker='v')
plt.scatter(baselines_schemes[:, 2], baseline_steps_per_state[:, 0]/75000, color='red', s=25, marker='v')
plt.text(other_schemes[best_rew, 2] + 0.02, other_steps_per_state[best_rew, 0]/75000 - 0.01, "Optimal Reward", ma='center', size = 7)
plt.text(baselines_schemes[:, 2] - 0.02, baseline_steps_per_state[:, 0]/75000 + 0.02, "Baseline", ma='center', size=7)
#plt.savefig("../results/fraction_hungry_vs_hungry_in_one_reward.png")
plt.close()


# Print Optimal policy that has been learned by training under the optimal reward function
# Print the index of the maximizing actions
print(np.transpose(np.argmax(other_Qval[best_rew], axis=3),  (2, 0, 1)))
