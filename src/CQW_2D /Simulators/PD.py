import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from joblib import Parallel, delayed
import numba

# Parameters
t = 25  # Number of time steps
size = 2 * t + 1  # Grid size
initial_position = size // 2  # Starting at the center
j = 1 # Number of iterations for averaging
disorder_strength = np.array(0, 0.2, 0.4, 0.6, 0.8, 1) # Disorder strength parameter W


# Initial state: walker at the center in a superposition
state = np.zeros((size, size, 4), dtype=complex)  # State has 4 components (up, down, left, right)
state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, -1/2]

# Coin operator with disorder
@numba.jit
def G_gate(disorder, W):
    rx = 0.5 * (1 + disorder_strength * disorder)
    
    G1 = np.array([
        [np.sqrt(rx[0]), np.sqrt(1-rx[0])],
        [np.sqrt(1-rx[0]) , -np.sqrt(rx[0])]
    ], dtype=np.complex128)

    G2 = np.array([
        [np.sqrt(rx[1]), np.sqrt(1-rx[1])],
        [np.sqrt(1-rx[1]) , -np.sqrt(rx[1])]
    ], dtype=np.complex128)

    G = np.kron(G1, G2).astype(np.complex128)

    return G


# Shift operator
@numba.jit
def apply_shift_2d(state):
    new_state = np.zeros_like(state)
    new_state[:-1, 1:, 0] = state[1:, :-1, 0]  # up - right
    new_state[1:, 1:, 1] = state[:-1, :-1, 1]  # down - right
    new_state[:-1, :-1, 2] = state[1:, 1:, 2]  # up - left
    new_state[1:, :-1, 3] = state[:-1, 1:, 3]  # down - left
    return new_state

@numba.jit
def disordered_walk(W, t):
    total_probability = np.zeros((size, size))

    for _ in range(j):
        random_disorder = np.random.uniform(-1, 1, (size, size, 4))

        state = np.zeros((size, size, 4), dtype = np.complex128)
        state[initial_position, initial_position, :] = np.array([1/2, (1j)/2, (1j)/2, -1/2], dtype = np.complex128)

        for step in range(t):
            for x in range(size):
                for y in range(size):

                    state[x, y, :] = np.dot(G_gate(random_disorder[x, y], W).astype(np.complex128), state[x, y, :])


            state = apply_shift_2d(state)

        probability = np.sum(np.abs(state)**2, axis = 2)
        total_probability += probability

    return total_probability / j 


# Function to compute and save probability distribution for a given W
def compute_and_save(W):
    PD = disordered_walk(W, t)  # Compute probability distribution

    # Convert to DataFrame
    df = pd.DataFrame(PD)
    df.index.name = 'X'
    df.columns.name = 'Y'

    # Save to CSV file
    filename = f'Walk_data_2d_W{W}.csv'
    df.to_csv(filename)

    print(f'Saved {filename}')  # Confirmation message
    return filename  # Returning filename for debugging if needed

# Use parallel processing to iterate over disorder_strength values
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(compute_and_save)(W) for W in disorder_strength
)
