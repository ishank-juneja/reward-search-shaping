from constants import *
import numpy as np


def get_discretization(scheme):
    # Basline case, rewards are the same
    # as underlygin sparse fitness values themselves
    if scheme == "baseline":
        disc_0 = [none_fit]
        disc_1_2 = [fat_fit]
        disc_3 = [both_fit]
    elif scheme == "other":
        disc_0 = np.array([-1, -0.5, -0.25, 0, 0.01, 0.02])
        disc_1_2 = np.linspace(-1, 1, 11)
        disc_3 = np.linspace(0, 1, 6)
    else:
        print("Incorrect Reward Scheme")
        exit(0)
    return disc_0, disc_1_2, disc_3

