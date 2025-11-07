import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import numba

# Parameters
time_steps = [20, 40, 60, 80, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 125
disorder_strength = np.linspace(0, 1, 50)

# Initial state: walker at the center in a superposition
initial_state = np.zeros((size, size, size, 8), dtype=np.complex128)
initial_state[initial_position, initial_position, initial_position, :] = np.array(
    [1/2, (1j)/2, (1j)/2, -1/2, (1j)/2, -1/2, -1/2, -1j/2], dtype=np.complex128
)

@numba.njit
def create_tunable_coin(disorder_values, W):
    """Generate site-specific G coin gate with disorder (vectorized)."""
    rx = 0.5 * (1 + W * disorder_values)
    
    # Precompute square roots once
    sqrt_rx = np.sqrt(rx)
    sqrt_1_rx = np.sqrt(1 - rx)

    # Generate G matrices efficiently
    G1 = np.array([[sqrt_rx[0], sqrt_1_rx[0]], [sqrt_1_rx[0], -sqrt_rx[0]]], dtype=np.complex128)
    G2 = np.array([[sqrt_rx[1], sqrt_1_rx[1]], [sqrt_1_rx[1], -sqrt_rx[1]]], dtype=np.complex128)
    G3 = np.array([[sqrt_rx[2], sqrt_1_rx[2]], [sqrt_1_rx[2], -sqrt_rx[2]]], dtype=np.complex128)

    # Use efficient Kronecker product
    return np.kron(G1, np.kron(G2, G3))

@numba.njit(parallel=True)
def shift_3d(state):
    """Optimized shift operator using NumPy slicing (in-place updates)."""
    new_state = np.zeros_like(state)

    # Use in-place shifts to reduce memory overhead
    new_state[1:, 1:, 1:, 0] = state[:-1, :-1, :-1, 0]  # (+x, +y, +z)
    new_state[1:, 1:, :-1, 1] = state[:-1, :-1, 1:, 1]  # (+x, +y, -z)
    new_state[1:, :-1, 1:, 2] = state[:-1, 1:, :-1, 2]  # (+x, -y, +z)
    new_state[1:, :-1, :-1, 3] = state[:-1, 1:, 1:, 3]  # (+x, -y, -z)
    new_state[:-1, 1:, 1:, 4] = state[1:, :-1, :-1, 4]  # (-x, +y, +z)
    new_state[:-1, 1:, :-1, 5] = state[1:, :-1, 1:, 5]  # (-x, +y, -z)
    new_state[:-1, :-1, 1:, 6] = state[1:, 1:, :-1, 6]  # (-x, -y, +z)
    new_state[:-1, :-1, :-1, 7] = state[1:, 1:, 1:, 7]  # (-x, -y, -z)

    return new_state

@numba.njit(parallel=True)
def run_quantum_walk(W, t):
    """Optimized quantum walk simulation."""
    total_participation = 0

    for _ in numba.prange(j):  # Parallelize over disorder realizations
        # Generate random disorder in advance
        random_disorder = np.random.uniform(-1, 1, (size, size, size, 8))

        # Create coin gates for all sites in advance
        coin_gates = np.empty((size, size, size, 8, 8), dtype=np.complex128)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    coin_gates[x, y, z] = create_tunable_coin(random_disorder[x, y, z], W)

        # Reset state
        state = np.zeros((size, size, size, 8), dtype=np.complex128)
        state[initial_position, initial_position, initial_position, :] = np.array(
            [1/2, (1j)/2, (1j)/2, -1/2, (1j)/2, -1/2, -1/2, -1j/2], dtype=np.complex128
        )

        # Time evolution
        for step in range(t):
            # Apply all coin operations in parallel
            for x in numba.prange(size):
                for y in numba.prange(size):
                    for z in numba.prange(size):
                        state[x, y, z, :] = np.dot(coin_gates[x, y, z], state[x, y, z, :])

            # Apply shift operator
            state = shift_3d(state)

        # Compute participation ratio (optimized)
        participation_value = 1 / (np.sum(np.abs(state) ** 4) * ((2 * t) ** 3 + 1))
        total_participation += participation_value

    return total_participation / j

# Parallelize execution over disorder strengths and time steps
data = {}
for t in time_steps:
    participation_values = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_quantum_walk)(W, t) for W in disorder_strength
    )
    
    data[f't={t}'] = participation_values

# Save results to CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'
df.to_csv('optimized_participation_data_3d.csv')

