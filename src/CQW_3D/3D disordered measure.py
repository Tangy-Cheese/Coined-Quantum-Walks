import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 15  # Number of time steps
size = 2 * t + 1  # Grid size
initial_position = size // 2  # Starting at the center
j = 1  # Number of iterations for averaging
disorder_strength = 0.90  # Disorder strength parameter W

# Initial state: walker at the center in a superposition
state = np.zeros((size, size, size, 8), dtype=complex)  # State has 8 components (3D walk)
state[initial_position, initial_position, initial_position, :] = [
    1/2, (1j)/2, (1j)/2, -1/2, (1j)/2, -1/2, -1/2, -1j/2
]

# Coin operator with disorder per coin state
def create_tunable_coin(disorder_values):
    rx = 0.5 * (1 + disorder_strength * disorder_values)  # Disorder-modulated r-values
    
    G1 = np.array([
        [np.sqrt(rx[0]), np.sqrt(1 - rx[0])],
        [np.sqrt(1 - rx[0]), -np.sqrt(rx[0])]
    ], dtype=complex)
    G2 = np.array([
        [np.sqrt(rx[1]), np.sqrt(1 - rx[1])],
        [np.sqrt(1 - rx[1]), -np.sqrt(rx[1])]
    ], dtype=complex)
    G3 = np.array([
        [np.sqrt(rx[2]), np.sqrt(1 - rx[2])],
        [np.sqrt(1 - rx[2]), -np.sqrt(rx[2])]
    ], dtype=complex)
    
    # Kronecker products:  C1 ⊗ C2 ⊗ C3  =>  8x8
    temp = np.kron(G1, G2)
    coin_operator = np.kron(temp, G3)
    return coin_operator

# Shift operator
def shift_3d(state):
    new_state = np.zeros_like(state)
    new_state[1:, 1:, 1:, 0] = state[:-1, :-1, :-1, 0]  # (+x, +y, +z)
    new_state[1:, 1:, :-1, 1] = state[:-1, :-1, 1:, 1]  # (+x, +y, -z)
    new_state[1:, :-1, 1:, 2] = state[:-1, 1:, :-1, 2]  # (+x, -y, +z)
    new_state[1:, :-1, :-1, 3] = state[:-1, 1:, 1:, 3]  # (+x, -y, -z)
    new_state[:-1, 1:, 1:, 4] = state[1:, :-1, :-1, 4]  # (-x, +y, +z)
    new_state[:-1, 1:, :-1, 5] = state[1:, :-1, 1:, 5]  # (-x, +y, -z)
    new_state[:-1, :-1, 1:, 6] = state[1:, 1:, :-1, 6]  # (-x, -y, +z)
    new_state[:-1, :-1, :-1, 7] = state[1:, 1:, 1:, 7]  # (-x, -y, -z)
    return new_state

# Accumulate probability distributions for averaging
average_probability = np.zeros((size, size, size))

# Run the walk for j iterations
for iteration in range(j):
    # Assign random disorder to each grid point for each coin state
    random_disorder = np.random.uniform(-1, 1, (size, size, size, 8))
    state = np.zeros((size, size, size, 8), dtype=complex)
    state[initial_position, initial_position, initial_position, :] = [
        1/2, (1j)/2, (1j)/2, -1/2, (1j)/2, -1/2, -1/2, -1j/2
    ]

    # Loop over time steps
    for step in range(t):
        # Apply coin operator with site-specific disorder
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    state[x, y, z, :] = np.dot(create_tunable_coin(random_disorder[x, y, z, :]), state[x, y, z, :])
        
        # Apply shift operator
        state = shift_3d(state)

    # Compute the probability distribution
    probability = np.sum(np.abs(state)**2, axis=-1)
    average_probability += probability  # Accumulate probability distribution

# Average the probability distribution over all iterations
average_probability /= j

# 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
coords = np.arange(-t, t+1)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

# Flatten for scatter
x_plot, y_plot, z_plot, p_plot = X.ravel(), Y.ravel(), Z.ravel(), average_probability.ravel()
mask = (p_plot >= 0.003)
ax.scatter(x_plot[mask], y_plot[mask], z_plot[mask], c=p_plot[mask], cmap="viridis", s=10, alpha=0.7)

# Add colorbar and labels
cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
cbar.set_label("Probability", fontsize=12)
ax.set_title("3D Probability Distribution After 3D Coined Quantum Walk", fontsize=15)
ax.set_xlabel("X-axis", fontsize=12)
ax.set_ylabel("Y-axis", fontsize=12)
ax.set_zlabel("Z-axis", fontsize=12)
plt.show()
