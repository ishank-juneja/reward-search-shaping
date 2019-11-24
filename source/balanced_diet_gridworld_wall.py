# Declare constants
pro_fit = 0.01
fat_fit = 0.01
both_fit = 1
none_fit = 0

extra_rew = 0
# Class to implement balanced diet problem environment
# Environment provides "Fitness" instead of the usual rewards
class World:
    # Constructor to init environment parameters
    def __init__(self, shape, born, fat, protein):
        # Read in size of grid world as a tuple
        self.shape = shape
        # Read in born/start state
        self.start = born
        # Satiated in Fat ?
        self.fat_sat = False
        # Satiated in Protein ?
        self.pro_sat = False
        # Read in Fat location
        self.fat_loc = fat
        # Read in Protein location
        self.pro_loc = protein
        # Reset current position to start
        self.cur_row = self.start[0]
        self.cur_col = self.start[1]
        self.extra_rew = 0

    # Reset state in which object ois
    def reset(self):
        # Reset current state to default start/born
        self.cur_row = self.start[0]
        self.cur_col = self.start[1]
        # Satiated in Fat ?
        self.fat_sat = False
        # Satiated in Protein
        self.pro_sat = False
        return

    # Action number vs outcome mapping
    # 0 = N, 1 = S, 2 = E, 3 = W, 4 = Eat
    def update_state(self, action, rand_fat, rand_pro):
        # Change satiatedness based on random event occurred
        # if random() < 0.2:
        #     self.fat_sat = False
        # if random() < 0.2:
        if rand_fat:
            self.fat_sat = False
        if rand_pro:
            self.pro_sat = False
        # Else do nothing

        if action == 0:
            # Up
            #self.cur_row += 1
            if(self.cur_row == 2 and (self.cur_col == 2 or self.cur_col == 3)):
                self.cur_row += 0
                self.extra_rew = 1

            else:
                self.cur_row += 1
                

        elif action == 1:
            # Down
            # self.cur_row += -1
            if(self.cur_row == 3 and (self.cur_col == 2 or self.cur_col == 3)):
                self.cur_row += 0
                self.extra_rew =1
            else:
                self.cur_row += -1

        elif action == 2:
            # Right
            #self.cur_col += 1
            if(self.cur_col == 2 and (self.cur_row == 2 or self.cur_row == 3)):
                self.cur_col += 0
                self.extra_rew =1
            else:
                self.cur_col += 1


        elif action == 3:
            # Left
            #self.cur_col += -1
            if(self.cur_col == 3 and (self.cur_row == 2 or self.cur_row == 3)):
                self.cur_col += 0
                self.extra_rew =1
            else:
                self.cur_col += -1


        elif action == 4:
            # Eat action, don't move, update satiatedness of on food location
            if (self.cur_row, self.cur_col) == self.fat_loc:
                self.fat_sat = True
            elif (self.cur_row, self.cur_col) == self.pro_loc:
                self.pro_sat = True
            # Else do nothing
        else:
            print("Incorrect action {0} encountered, aborting".format(action))
            exit(0)
        # Clip the effect of actions to within the grid limits
        self.cur_row = max(min(self.cur_row, self.shape[0] - 1), 0)
        self.cur_col = max(min(self.cur_col, self.shape[1] - 1), 0)
        # Fitness increment based on constants defined above
        # Encode satiated state for return
        if self.pro_sat and self.fat_sat:
            fit_incr = both_fit
            sat_state = 3
        elif self.pro_sat:
            fit_incr = pro_fit
            sat_state = 2
        elif self.fat_sat:
            fit_incr = fat_fit
            sat_state = 1
        else:
            fit_incr = none_fit
            sat_state = 0
        # Return fitness increment based on Satiatedness and also return next state to agent


        if(self.cur_col==0 or self.cur_row==0):

            self.extra_rew = -200

        fit_incr +=-0.2*(self.extra_rew)
        self.extra_rew=0



        return fit_incr/200, (self.cur_row, self.cur_col, sat_state)
