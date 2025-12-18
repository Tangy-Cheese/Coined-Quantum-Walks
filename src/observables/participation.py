import numpy as np

def participation_value(state, t):
    return 1 / (np.sum(np.abs(state)**4) * (2*t + 1))
