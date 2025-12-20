import numpy as np
import pandas as pd
import sys

import numba

from operators import shift_3d, G_3d
from observables import participation_3d

# Parameters
time_steps = [1, 2, 5, 10]
size = 2 * max(time_steps) + 1
initial_position = size // 2

# Realisations
j = 1  # sensible for 3D

# Allow override from SLURM / command line
if len(sys.argv) == 2:
    j = int(sys.argv[1])

# Disorder strengths
disorder_strength = np.linspace(0, 1, 3)

# Data storage
data = {}

for t in time_steps:
    participation_values = []

    for W in disorder_strength:
        total_participation = 0

        for iteration in range(j):
            # Generate random disorder
            random_disorder = np.random.uniform(
                -1, 1, (size, size, size, 8)
            )

            # Initialise state
            state = np.zeros((size, size, size, 8), dtype=np.complex128)
            state[
                initial_position,
                initial_position,
                initial_position,
                :
            ] = [
                1/2, 1j/2, 1j/2, -1/2,
                1j/2, -1/2, -1/2, -1j/2
            ]

            # Time evolution
            for step in range(t):
                for x in range(size):
                    for y in range(size):
                        for z in range(size):
                            state[x, y, z, :] = np.dot(
                                G_3d(random_disorder[x, y, z, :], W),
                                state[x, y, z, :]
                            )

                state = shift_3d(state)

            # Participation ratio
            participation_value = participation_3d(state, t)
            total_participation += participation_value

        participation_values.append(total_participation / j)

    data[f"t={t}"] = participation_values

# Save to CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = "Disorder Strength"
df["j"] = j

df.to_csv("../../output/data/3D_PV.csv")
