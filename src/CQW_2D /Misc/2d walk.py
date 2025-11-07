import numpy as np
import math
import matplotlib.pyplot as plt
t = 100 # Number of time steps
size = 2 * t + 1
state = np.zeros((size, size, 4), dtype=complex)
initial_position = size // 2

# Initial state: walker at the center in a superposition
state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, - 1/2]

# Coin operator: Generalized 4-state Hadamard
def create_tunable_coin(theta1, beta, gamma):
    """Create an 8x8 coin operator from three 2x2 blocks (C1, C2, C3)."""
    C1 = np.array([
        [np.cos(theta1),                  np.sin(theta1)*np.exp(1j*beta)],
        [np.sin(theta1)*np.exp(1j*gamma),-np.cos(theta1)*np.exp(1j*(gamma+beta))]
    ], dtype=complex)

    C2 = np.array([
        [np.cos(theta1),                  np.sin(theta1)*np.exp(1j*beta)],
        [np.sin(theta1)*np.exp(1j*gamma),-np.cos(theta1)*np.exp(1j*(gamma+beta))]
    ], dtype=complex)

     # Kronecker products:  C1 ⊗ C2 ⊗ C3  =>  8x8
    coin_operator = np.kron(C1, C2)
    return coin_operator

# Example Grover-like coin with angles = pi/4, no phases:
theta1 = np.pi/4
beta = 0
gamma = 0
coin_operator = create_tunable_coin(theta1, beta, gamma)


# Shift operator
def apply_shift_2d(state):
    new_state = np.zeros_like(state)
    new_state[:-1, 1:, 0] = state[1:, :-1, 0]  # up - right
    new_state[1:, 1:, 1] = state[:-1, :-1, 1]  # down - right
    new_state[:-1, :-1, 2] = state[1:, 1:, 2]  # up - left
    new_state[1:, :-1, 3] = state[:-1, 1:, 3]  # down - left
    return new_state

# Quantum walk evolution
for step in range(t):
    # Apply coin operator
    for x in range(size):
        for y in range(size):
            state[x, y, :] = np.dot(coin_operator, state[x, y, :])
    # Apply shift operator
    state = apply_shift_2d(state)

# Compute probability distribution
probability = np.sum(np.abs(state)**2, axis=2)

# Plot heatmap of the final probability distribution
plt.figure(figsize=(8, 6))
plt.imshow(probability, extent=(-t, t, -t, t), origin='lower', cmap='viridis', interpolation='nearest')
plt.colorbar(label="Probability")
plt.title("2D Quantum Walk Probability Distribution (Ballistic Spread)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(False)
plt.show()