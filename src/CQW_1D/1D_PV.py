import numpy as np
import pandas as pd
from numba import prange, njit
import sys #For scripting runs

from operators import shift_1d, G_1d

# Parameters
time_steps = [10, 25, 50, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2

# Realisations
j = 10 # This is fine for local runs 

# Just saying here if we wanna change the variable j, can do it in the SLURM script easily by adding another argument
if len(sys.argv) == 2:
    j = int(sys.argv[1])

#Evenly spaced disorder strengths
disorder_strength = np.linspace(0, 1, 11)

# Data storage
data = {}

for t in time_steps:
    participation_values = []

    for W in disorder_strength:
        total_participation = 0

        for iteration in range(j):
            random_disorder = np.random.uniform(-1, 1, size)
            state = np.zeros((size, 2), dtype=complex)
            state[initial_position, :] = [1/np.sqrt(2), 1j/np.sqrt(2)]

            for step in range(t):
                for x in range(size):
                    state[x, :] = np.dot(G_1d(random_disorder[x], W), state[x, :])
                state = shift_1d(state)

            participation_value = 1 / (np.sum(np.abs(state)**4) * ((2*t)+1))
            total_participation += participation_value


        participation_values.append(total_participation / j)

    data[f't={t}'] = participation_values

# Saving data as CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'

# Add j to the data as another column. This is just for convenience though.
df["j"] = j
df.to_csv(r"../../output/data/1D_PV.csv")
