from random import random, randint
import numpy as np
from balanced_diet_gridworld import World
from policy import eps_greedy
from scipy.stats import bernoulli
import sys
import matplotlib.pyplot as plt


grid_shape= (6,6)
# Fat position in grid
fat_loc = (5, 0)
# Protein location in grid
pro_loc = (0, 5)

dist_betn_foods= 10

p_decay=0.1

##Potential function will be the same size as the reward matrix, there's a potential associated with each (state X location)
pot_func1 = np.zeros((grid_shape[0], grid_shape[1], 4))
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        # dist_from_fat= (5-x) + (y)
        # dist_from_pro= (x) + (5-y)
        # #If state is 0, we first need to go to food at min distance, get that food item and then go forward
        # d_min=min(dist_from_fat, dist_from_pro)
        # pot_func[x][y][0]= -float(d_min) - (float(dist_betn_foods)/np.power(float(1-p_decay), dist_betn_foods))
        # ### If state is 1, it needs to go go to protein
        # pot_func[x][y][1] = -(float(dist_from_pro)/np.power(float(1-p_decay), dist_from_pro)) 
        # ### If state is 2, it needs to go to fat
        # pot_func[x][y][2] = -(float(dist_from_fat)/np.power(float(1-p_decay), dist_from_fat))
        # ###If state is 3, doesn't need to explicitly go anywhere
        # pot_func[x][y][3] = 0
        if(x+y <7 and x+y>3):
            pot_func1[x][y][0] = 1.0
            pot_func1[x][y][1] = 1.0
            pot_func1[x][y][2] = 1.0
            pot_func1[x][y][3] = 1.0
        else:
            pot_func1[x][y][0] = 0.0
            pot_func1[x][y][1] = 0.0
            pot_func1[x][y][2] = 0.0
            pot_func1[x][y][3] = 0.0

pot_func2 = np.zeros((grid_shape[0], grid_shape[1], 4))
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        # dist_from_fat= (5-x) + (y)
        # dist_from_pro= (x) + (5-y)
        # #If state is 0, we first need to go to food at min distance, get that food item and then go forward
        # d_min=min(dist_from_fat, dist_from_pro)
        # pot_func[x][y][0]= -float(d_min) - (float(dist_betn_foods)/np.power(float(1-p_decay), dist_betn_foods))
        # ### If state is 1, it needs to go go to protein
        # pot_func[x][y][1] = -(float(dist_from_pro)/np.power(float(1-p_decay), dist_from_pro)) 
        # ### If state is 2, it needs to go to fat
        # pot_func[x][y][2] = -(float(dist_from_fat)/np.power(float(1-p_decay), dist_from_fat))
        # ###If state is 3, doesn't need to explicitly go anywhere
        # pot_func[x][y][3] = 0
        if(x+y <7 and x+y>4):
            pot_func2[x][y][0] = 1.0
            pot_func2[x][y][1] = 1.0
            pot_func2[x][y][2] = 1.0
            pot_func2[x][y][3] = 1.0
        else:
            pot_func2[x][y][0] = 0.0
            pot_func2[x][y][1] = 0.0
            pot_func2[x][y][2] = 0.0
            pot_func2[x][y][3] = 0.0



pot_func3 = np.zeros((grid_shape[0], grid_shape[1], 4))
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        # dist_from_fat= (5-x) + (y)
        # dist_from_pro= (x) + (5-y)
        # #If state is 0, we first need to go to food at min distance, get that food item and then go forward
        # d_min=min(dist_from_fat, dist_from_pro)
        # pot_func[x][y][0]= -float(d_min) - (float(dist_betn_foods)/np.power(float(1-p_decay), dist_betn_foods))
        # ### If state is 1, it needs to go go to protein
        # pot_func[x][y][1] = -(float(dist_from_pro)/np.power(float(1-p_decay), dist_from_pro)) 
        # ### If state is 2, it needs to go to fat
        # pot_func[x][y][2] = -(float(dist_from_fat)/np.power(float(1-p_decay), dist_from_fat))
        # ###If state is 3, doesn't need to explicitly go anywhere
        # pot_func[x][y][3] = 0
        if(x==0 or y==0):
            pot_func3[x][y][0] = 1.0
            pot_func3[x][y][1] = 1.0
            pot_func3[x][y][2] = 1.0
            pot_func3[x][y][3] = 1.0
        else:
            pot_func3[x][y][0] = 0.0
            pot_func3[x][y][1] = 0.0
            pot_func3[x][y][2] = 0.0
            pot_func3[x][y][3] = 0.0


pot_func4 = np.zeros((grid_shape[0], grid_shape[1], 4))
for x in range(grid_shape[0]):
    for y in range(grid_shape[1]):
        dist_from_fat= (5-x) + (y)
        dist_from_pro= (x) + (5-y)
        #If state is 0, we first need to go to food at min distance, get that food item and then go forward
        d_min=min(dist_from_fat, dist_from_pro)
        pot_func4[x][y][0]= -float(d_min) - (float(dist_betn_foods)/np.power(float(1-p_decay), dist_betn_foods))
        ### If state is 1, it needs to go go to protein
        pot_func4[x][y][1] = -(float(dist_from_pro)/np.power(float(1-p_decay), dist_from_pro)) 
        ### If state is 2, it needs to go to fat
        pot_func4[x][y][2] = -(float(dist_from_fat)/np.power(float(1-p_decay), dist_from_fat))
        ###If state is 3, doesn't need to explicitly go anywhere
        pot_func4[x][y][3] = 0


# ###Now that potential function is declared, we only need to add the these to the original rewards in the learning setup.
# for x in range(grid_shape[0]):
#     for y in range(grid_shape[1]):
#         pot_func[x][y][0]+=0.0
#         pot_func[x][y][1]+=0.0
#         pot_func[x][y][2]+=0.0
#         pot_func[x][y][3]+=0.0
#         # print(pot_func[x][y])

##Now, this is the reward associated with each state, after the process of reward shaping. Now, we'll apply the
##same learning arrangement with agent.py ##Everything is just copy pasted with change to the learning setup

# Start location
start = (0, 5)

environment = World(grid_shape, start, fat_loc, pro_loc)
# Algorithm parameters
alpha = 0.5 # Learning rate
# Number of time steps in continuing task
max_steps = 125000
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
#rew_array = np.array([-0.1, -0.05, -0.05, 1]) #There's no more need for a reward array
rew_array = np.array([0.0, 0.01, 0.01, 1])
rew_array/=10


pot_bool = 1 # to enable reward_shaping


#TO store fitness values for every timestep
fitness_array = np.zeros((6,max_steps))

# time taken to reach satiated state


if __name__ == '__main__':
    # Declare array to hold Q_values Q(s, a), init with zeros
    # State coding is 0 --> Un sat in both, 1 --> sat in fat only,
    # 2 --> sat in pro only, 3 --> sat in both so 4 possible states
    for i in range(6):


        if i == 0:
            pot_func = np.zeros((grid_shape[0], grid_shape[1], 4))
        elif i==1:
            pot_func = pot_func1
        elif i==2:
            pot_func = pot_func2
        elif i==3:
            pot_func = pot_func3
        elif i==4:
            pot_func = pot_func4;
        else:
            pot_func = np.zeros((grid_shape[0], grid_shape[1], 4))
            rew_array = np.array([-0.25, -0.2, -0.2, 1.0])


       

        for rs in range(10):

            Q_values = np.zeros((grid_shape[0], grid_shape[1], 4, nactions))
            # Initialise state as seen by agent
            row = start[0]
            col = start[1]
            sat_state = 0
            # Intialise cummulative fitness
            cum_fit = 0
            # Pre-generate sequence of bernousat_statesat_statelli random numbers

            X1 = bernoulli(prob_fat)
            rands_fat = X1.rvs(max_steps, random_state=rs+1)
            # Use a different random seed for un-correlatedness
            X2 = bernoulli(prob_pro)
            rands_pro = X2.rvs(max_steps, random_state=rs)
            action = eps_greedy(Q_values, row, col, sat_state, nactions)
            for t in range(max_steps):
                # Get epsilon-greedy action wrt current state
                # action = eps_greedy(Q_values, row, col, sat_state, nactions)
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
                    print("Success") ##The agent is satiated in both
                # Make Q-learning(0) update to Q values
                next_action=eps_greedy(Q_values, next_row, next_col, next_sat_state, nactions) ##Using SARSA
                Q_values[row][col][sat_state][action] += \
                    alpha * (rew + pot_bool*(gamma*pot_func[next_row][next_col][next_sat_state] - pot_func[row][col][sat_state]) + gamma * (np.max(Q_values[next_row][next_col][next_sat_state])) - Q_values[row][col][sat_state][action])
                (row, col, sat_state, action) = (next_row, next_col, next_sat_state, next_action)
                cum_fit += fitness
                fitness_array[i][t]+=cum_fit

        print(cum_fit)
       # print(size(fitness_array))

    y0 = np.asarray(fitness_array[0]/10)
    y1 = np.asarray(fitness_array[1]/10)
    y2 = np.asarray(fitness_array[2]/10)
    y3 = np.asarray(fitness_array[3]/10)
    y4 = np.asarray(fitness_array[4]/10)
    y5 = np.asarray(fitness_array[5]/10)
    np.save('y0',y0)
    np.save('y1',y1)
    np.save('y2',y2)
    np.save('y3',y3)
    np.save('y4',y4)
    np.save('y5',y5)
    x = [i for i in range(max_steps)]
    plt.plot(x, y1, '-', linewidth='3', label='Broad Diagonal Potential', color='green') 
    plt.plot(x, y2, '--', linewidth='1', label='Thin Diagonal Potential', color='green') 
    plt.plot(x, y3, '--',label='Edge Potential', color='green') 
    
    plt.plot(x, y5, '-',label='Optimal Reward Search', color='red') 
    plt.plot(x, y0, '-',label='Baseline', color='black') 
    plt.plot(x, y4, '-',label='Value Based Potential', color='y')
plt.ylabel('Cummulative Fitness')
plt.xlabel('Time step in Agent lifetime')
plt.title('Fitness growth for reward shaping')
plt.legend()
plt.show()



