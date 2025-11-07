import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) Parameters
###############################################################################
t = 20
size = 2*t + 1  # So that we can go naturally from -t to +t
initial_position = t  # Center of the grid

###############################################################################
# 2) Allocate state array
###############################################################################
# state[x, y, z, :] holds amplitudes for 8 coin states at position (x, y, z)
state = np.zeros((size, size, size, 8), dtype=complex)

###############################################################################
# 3) Define the initial coin amplitudes at the center
###############################################################################
# The amplitudes you mentioned:
# [  1*(1/(2√2)),   1j*(1/(2√2)),  -1*(1/(2√2)),   1j*(1/(2√2)),
#    1j*(1/(2√2)),  -1*(1/(2√2)),  -1j*(1/(2√2)),  -1*(1/(2√2)) ]

init_amp = np.array([
    1/(2*np.sqrt(2)),
    1j/(2*np.sqrt(2)),
    1j/(2*np.sqrt(2)),
    -1/(2*np.sqrt(2)),
    1j/(2*np.sqrt(2)),
    -1/(2*np.sqrt(2)),
   -1/(2*np.sqrt(2)),
   -1j/(2*np.sqrt(2))
], dtype=complex)

# Place it in the 3D array at the center
state[initial_position, initial_position, initial_position, :] = init_amp

###############################################################################
# 4) Define a tunable 8x8 coin operator via 2x2 Kronecker products
###############################################################################
def create_tunable_coin(theta1, theta2, theta3, beta, gamma):
    """Create an 8x8 coin operator from three 2x2 blocks (C1, C2, C3)."""
    C1 = np.array([
        [np.cos(theta1),                  np.sin(theta1)*np.exp(1j*beta)],
        [np.sin(theta1)*np.exp(1j*gamma),-np.cos(theta1)*np.exp(1j*(gamma+beta))]
    ], dtype=complex)
    C2 = np.array([
        [np.cos(theta2),                  np.sin(theta2)*np.exp(1j*beta)],
        [np.sin(theta2)*np.exp(1j*gamma),-np.cos(theta2)*np.exp(1j*(gamma+beta))]
    ], dtype=complex)
    C3 = np.array([
        [np.cos(theta3),                  np.sin(theta3)*np.exp(1j*beta)],
        [np.sin(theta3)*np.exp(1j*gamma),-np.cos(theta3)*np.exp(1j*(gamma+beta))]
    ], dtype=complex)
    
    # Kronecker products:  C1 ⊗ C2 ⊗ C3  =>  8x8
    temp = np.kron(C1, C2)
    coin_operator = np.kron(temp, C3)
    return coin_operator

CNOT = np.array([
    [1,  0,  0,  0,  0,  0,  0,  0],
    [0,  1,  0,  0,  0,  0,  0,  0],
    [0,  0,  1,  0,  0,  0,  0,  0],
    [0,  0,  0,  1,  0,  0,  0,  0],
    [0,  0,  0,  0,  1,  0,  0,  0],
    [0,  0,  0,  0,  0,  1,  0,  0],
    [0,  0,  0,  0,  0,  0,  0,  1],
    [0,  0,  0,  0,  0,  0,  1,  0]
])


# Example Grover-like coin with angles = pi/4, no phases:
theta1 = theta2 = theta3 = np.pi/4
beta = 0
gamma = 0
coin_operator = np.dot(create_tunable_coin(theta1, theta2, theta3, beta, gamma), CNOT)

###############################################################################
# 5) Define the 3D shift operator
###############################################################################
# We label coin states 0..7 as movement in the directions:
#   0: (+x, +y, +z)
#   1: (+x, +y, -z)
#   2: (+x, -y, +z)
#   3: (+x, -y, -z)
#   4: (-x, +y, +z)
#   5: (-x, +y, -z)
#   6: (-x, -y, +z)
#   7: (-x, -y, -z)
#
# This consistent labeling helps produce the "cube" expansion for symmetrical states.

def shift_3d(old_state):
    new_state = np.zeros_like(old_state)

    #(+x, +y, +z)
    new_state[1:, 1:, 1:, 0] = old_state[:-1, :-1, :-1, 0]
    
    #(+x, +y, -z)
    new_state[1:, 1:, :-1, 1] = old_state[:-1, :-1, 1:, 1]
    
    #(+x, -y, +z)
    new_state[1:, :-1, 1:, 2] = old_state[:-1, 1:, :-1, 2]
    
    #(+x, -y, -z)
    new_state[1:, :-1, :-1, 3] = old_state[:-1, 1:, 1:, 3]
    
    #(-x, +y, +z)
    new_state[:-1, 1:, 1:, 4] = old_state[1:, :-1, :-1, 4]
    
    #(-x, +y, -z)
    new_state[:-1, 1:, :-1, 5] = old_state[1:, :-1, 1:, 5]
    
    #(-x, -y, +z)
    new_state[:-1, :-1, 1:, 6] = old_state[1:, 1:, :-1, 6]
    
    #(-x, -y, -z)
    new_state[:-1, :-1, :-1, 7] = old_state[1:, 1:, 1:, 7]
    
    return new_state

###############################################################################
# 6) Evolve the walk for t steps
###############################################################################
for step in range(t):
    # (a) Apply coin operator at each site
    #     We do it site by site for clarity (though vectorized is faster)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                state[x, y, z, :] = np.dot(coin_operator, state[x, y, z, :])
    
    # (b) Apply the shift operator
    state = shift_3d(state)

###############################################################################
# 7) Compute and plot the final probability distribution
###############################################################################
probability = np.sum(np.abs(state)**2, axis=-1)

# Mask threshold (adjust as you like)
average_prob = 0.0003

# We want to plot from -t to +t along each axis
coords = np.arange(-t, t+1)  # length = size
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

# Flatten for scatter
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = Z.ravel()
p_flat = probability.ravel()

# Apply mask
mask = (p_flat >= average_prob)
x_plot = x_flat[mask]
y_plot = y_flat[mask]
z_plot = z_flat[mask]
p_plot = p_flat[mask]

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x_plot, y_plot, z_plot, c=p_plot, cmap="viridis", s=10, alpha=0.7)

# Add colorbar and labels
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("Probability", fontsize=12)
ax.set_title("3D Probability Distribution After 3D Coined Quantum Walk", fontsize=15)
ax.set_xlabel("X-axis", fontsize=12)
ax.set_ylabel("Y-axis", fontsize=12)
ax.set_zlabel("Z-axis", fontsize=12)

plt.show()