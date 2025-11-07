import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import numba

# Parameters
time_steps = [20, 40, 60, 80, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 500
disorder_strength = np.linspace(0, 1, 50)

# Initial state: walker at the center in a superposition
initial_state = np.zeros((size, size, 4), dtype=np.complex128)
initial_state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, -1/2]

@numba.njit
def G_gate(disorder, w):
    """Generate the site-specific G coin gate with disorder."""
    rx = 0.5 * (1 + w * disorder) 

    G1 = np.array([
        [np.sqrt(rx[0]), np.sqrt(1-rx[0])],
        [np.sqrt(1-rx[0]), -np.sqrt(rx[0])]
    ], dtype=np.complex128)  # <-- Ensure complex128

    G2 = np.array([
        [np.sqrt(rx[1]), np.sqrt(1-rx[1])],
        [np.sqrt(1-rx[1]), -np.sqrt(rx[1])]
    ], dtype=np.complex128)  # <-- Ensure complex128

    G = np.kron(G1, G2).astype(np.complex128)  # <-- Ensure complex128

    return G

@numba.njit
def apply_shift_2d(state):
    """Shift operator for 2D quantum walk."""
    new_state = np.zeros_like(state)
    new_state[:-1, 1:, 0] = state[1:, :-1, 0]  # up - right
    new_state[1:, 1:, 1] = state[:-1, :-1, 1]  # down - right
    new_state[:-1, :-1, 2] = state[1:, 1:, 2]  # up - left
    new_state[1:, :-1, 3] = state[:-1, 1:, 3]  # down - left
    return new_state

@numba.njit
def run_quantum_walk(W, t):
    """Simulate the quantum walk for a given disorder strength W and time t."""
    total_participation = 0

    for _ in range(j):
        # Generate random disorder
        random_disorder = np.random.uniform(-1, 1, (size, size, 4))

        # Reset state
        state = np.zeros((size, size, 4), dtype=np.complex128)
        state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, -1/2]

        # Time evolution
        for step in range(t):
            # Apply coin operator with disorder
            for x in range(size):
                for y in range(size):
                    # Ensure G_gate() output is complex128
                    state[x, y, :] = np.dot(G_gate(random_disorder[x, y], W).astype(np.complex128), state[x, y, :])

            # Apply shift operator
            state = apply_shift_2d(state)

        # Compute participation ratio
        participation_value = 1 / (np.sum(np.abs(state) ** 4) * ((2 * t) ** 2 + 1))
        total_participation += participation_value

    return total_participation / j



# Use parallel processing for different disorder strengths
data = {}
for t in time_steps:
    participation_values = Parallel(n_jobs=-1, backend="loky")(delayed(run_quantum_walk)(W, t) for W in disorder_strength)
    data[f't={t}'] = participation_values

# Save to CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'
df.to_csv('participation_data_2d.csv')

