import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import numba

# Parameters
time_steps = [20, 40, 60, 80, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 2000
disorder_strength = np.linspace(0, 1, 50)

# Coin operator with disorder
@numba.njit
def G_gate(disorder, w):
    r_x = 0.5 * (1 + w * disorder)
    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ], dtype=np.complex128)
    return G.astype(np.complex128)

# Shift operator
@numba.njit
def apply_shift(state):
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]
    new_state[:-1, 1] = state[1:, 1]
    return new_state

# Data storage
@numba.njit
def run_quantum_walk(W, t):
    """Simulate the quantum walk for a given disorder strength W and time t."""
    total_participation = 0

    for _ in range(j):
        # Generate random disorder
        random_disorder = np.random.uniform(-1, 1, (size))

        # Reset state
        state = np.zeros((size, 2), dtype=np.complex128)
        state[initial_position, :] = [
            1/np.sqrt(2), (1j)/np.sqrt(2)
        ]

        # Time evolution
        for step in range(t):
            # Apply coin operator with disorder
            for x in range(size):
                state[x, :] = np.dot(G_gate(random_disorder[x], W), state[x, :])

            # Apply shift operator
            state = apply_shift(state)

        # Compute participation ratio
        participation_value = 1 / (np.sum(np.abs(state) ** 4) * ((2 * t) + 1))
        total_participation += participation_value

    return total_participation / j


# Use parallel processing for different disorder strengths
data = {}
for t in time_steps:
    participation_values = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_quantum_walk)(W, t) for W in disorder_strength
    )
    data[f't={t}'] = participation_values

# Saving data as CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'
df.to_csv('participation_data.csv')
