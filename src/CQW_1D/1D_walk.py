import numpy as np
import pandas as pd
from numba import prange, njit
import sys

# Outlining Variables:
time_steps = 100  # Number of max time steps
size = 2 * time_steps + 1
initial_position = size // 2  # Start at Middle

j = 250  # Number of iterations for averaging

if len(sys.argv) == 3:
    j = int(sys.argv[1])
    time_steps = int(sys.argv[2])

disorder_strength = np.linspace(0, 1, 6)  # Disorder strength parameter W

# Shift operator 
def apply_shift_operator(state):
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]  # down state shift
    new_state[:-1, 1] = state[1:, 1]  # up state shift
    return new_state

def G_gate(disorder, w):

    r_x = 0.5 * (1 + w * disorder)

    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ])
    return G

data = {}

for W in disorder_strength:
    # Run the walk for j iterations
    for iteration in range(j):

        random_disorder = np.random.uniform(-1, 1, size)    
        initial_state = np.zeros((size, 2), dtype=complex)
        initial_state[initial_position] = [1 / np.sqrt(2), 1j / np.sqrt(2)]  # Initial Pauli Y Gate Eigenvectors

        state = np.copy(initial_state)  # Reset state for each iteration

        # Loop over time steps
        for step in range(time_steps):
            # Apply coin operator with site-specific disorder
            state = np.array([np.dot(G_gate(random_disorder[step], W), state[pos]) for pos in range(size)])
            state = apply_shift_operator(state)  # Apply shift operator

            # If this time step is in time_to_plot, store the probability distribution
            if step + 1 in time_steps:
                probability_distribution = np.sum(np.abs(state) ** 2, axis=1)  # Eqn. 1
                time_index = time_steps.index(step + 1)
                average_probability_distributions[time_index] += probability_distribution  # Accumulate

        # Divide by the number of iterations to get the average
        average_probability_distributions /= j

    data[f't={time_steps}'] = average_probability_distributions

positions = np.arange(-initial_position, initial_position)


# Saving data as CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'

# Add j to the data as another column. This is just for convenience though.
df["j"] = j
df.to_csv(r"../../output/data/1D_walk.csv")
