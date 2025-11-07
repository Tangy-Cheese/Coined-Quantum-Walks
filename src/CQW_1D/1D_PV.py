import numpy as np
import pandas as pd

# Parameters
time_steps = [20, 40, 60, 80, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 10
disorder_strength = np.linspace(0, 1, 10)

# Coin operator with disorder
def G_gate(disorder, w):
    r_x = 0.5 * (1 + w * disorder)
    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ])
    return G

# Shift operator
def apply_shift(state):
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]
    new_state[:-1, 1] = state[1:, 1]
    return new_state

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
                    state[x, :] = np.dot(G_gate(random_disorder[x], W), state[x, :])
                state = apply_shift(state)

            participation_value = 1 / (np.sum(np.abs(state)**4) * ((2*t)+1))
            total_participation += participation_value


        participation_values.append(total_participation / j)

    data[f't={t}'] = participation_values

# Saving data as CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'
df.to_csv('participation_data.csv')
