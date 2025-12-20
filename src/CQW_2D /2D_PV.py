import numpy as np
import pandas as pd
import sys  # For scripting runs

from operators import shift_2d, G_2d

from observables import participation_2d

# Parameters
time_steps = [10, 20, 30]
size = 2 * max(time_steps) + 1
initial_position = size // 2

# Realisations
j = 1  # Fine for local runs

# Allow overriding j from SLURM / command line
if len(sys.argv) == 2:
    j = int(sys.argv[1])

# Evenly spaced disorder strengths
disorder_strength = np.linspace(0, 1, 11)

# Data storage
data = {}

for t in time_steps:
    participation_values = []

    for W in disorder_strength:
        total_participation = 0

        for iteration in range(j):
            # Generate random disorder
            random_disorder = np.random.uniform(-1, 1, (size, size, 4))

            # Initialise state
            state = np.zeros((size, size, 4), dtype=np.complex128)
            state[initial_position, initial_position, :] = [
                1/2, 1j/2, 1j/2, -1/2
            ]

            # Time evolution
            for step in range(t):
                for x in range(size):
                    for y in range(size):
                        state[x, y, :] = np.dot(
                            G_2d(random_disorder[x, y], W),
                            state[x, y, :]
                        )

                state = shift_2d(state)

            # Participation ratio
            participation_value = participation_2d(state, t)
            total_participation += participation_value

        participation_values.append(total_participation / j)

    data[f"t={t}"] = participation_values

# Saving data as CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = "Disorder Strength"

# Add j for convenience
df["j"] = j
df.to_csv(r"../../output/data/2D_PV.csv")
