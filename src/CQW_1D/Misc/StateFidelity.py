import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 100  # Number of time steps
num_positions = 201  # Number of positions on the line (odd for symmetry)
initial_position = num_positions // 2  # Start at the middle
j = 100  # Number of iterations for averaging
disorder_strength = 1  # Disorder strength parameter W

# Initial state (Eqn. 5)
initial_state = np.zeros((num_positions, 2), dtype=complex)
initial_state[initial_position] = [1 / np.sqrt(2), 1j / np.sqrt(2)]  # Pauli-Y eigenvector

# Shift operator (Eqn. 4)
def apply_shift_operator(state):
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]  # Down state shift
    new_state[:-1, 1] = state[1:, 1]  # Up state shift
    return new_state

# Coin operator (G_gate) with site-specific disorder
def G_gate(position, r_values):
    r_x = r_values[position]  # Use site-specific r value
    G = np.array([[np.sqrt(r_x), np.sqrt(1 - r_x)], [np.sqrt(1 - r_x), -np.sqrt(r_x)]])
    return G

# Function to compute the fidelity between disordered state and Hadamard state
def compute_state_fidelity(state_disordered, state_hadamard):
    overlap = np.sum(state_disordered.conj() * state_hadamard)  # Inner product of the states
    fidelity = np.abs(overlap)**2  # Squared magnitude of the overlap
    return fidelity

# Initialize state fidelity values
fidelity_values = np.zeros(t)

# Run the walk for j iterations
for iteration in range(j):
    # Assign random disorder
    random_disorder = np.random.uniform(-1, 1, num_positions)
    r_values = 0.5 * (1 + disorder_strength * random_disorder)  # Scale disorder

    # Reset states for each iteration
    state_disordered = np.copy(initial_state)
    state_hadamard = np.copy(initial_state)

    # Loop over time steps
    for step in range(t):
        # Apply coin operator and shift for disordered state
        state_disordered = np.array([np.dot(G_gate(pos, r_values), state_disordered[pos]) 
                                      for pos in range(num_positions)])
        state_disordered = apply_shift_operator(state_disordered)

        # Apply Hadamard coin operator and shift for Hadamard state
        hadamard_coin = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], 
                                  [1 / np.sqrt(2), -1 / np.sqrt(2)]])  # Hadamard operator
        state_hadamard = np.array([np.dot(hadamard_coin, state_hadamard[pos]) 
                                   for pos in range(num_positions)])
        state_hadamard = apply_shift_operator(state_hadamard)

        # Calculate state fidelity
        fidelity_values[step] += compute_state_fidelity(state_disordered, state_hadamard)

# Average the fidelity values across iterations
fidelity_values /= j

# Plot the state fidelity as a function of time
plt.figure(figsize=(10, 6))
plt.plot(range(1, t + 1), fidelity_values, label="State Fidelity")
plt.xlabel("Time (t)")
plt.ylabel("F(t)")
plt.title(f"State Fidelity vs. Time for Disordered Quantum Walk (W={disorder_strength})")
plt.legend()
plt.grid(True)
plt.show()
