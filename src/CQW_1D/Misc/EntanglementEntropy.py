import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 100  # Number of time steps
num_positions = 201  # Number of positions on the line (odd for symmetry)
initial_position = num_positions // 2  # Start at the middle
time_to_plot = [100]  # Specific times to plot
j = 50  # Number of iterations for averaging
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

# Define the reduced density matrix for the coin
def compute_density_matrix(state):
    # Initialize the reduced density matrix
    rho_c = np.zeros((2, 2), dtype=complex)
    
    # Sum over all positions to trace out position degrees of freedom
    for pos in range(state.shape[0]):
        coin_state = state[pos, :].reshape(2, 1)  # Coin state at position
        rho_c += np.outer(coin_state, coin_state.conj())  # Outer product contributes to reduced rho
    return rho_c

# Define the Entanglement Entropy property
def EE(rho):
    eigenvalues = np.linalg.eigvalsh(rho)  # Get eigenvalues of the reduced density matrix
    eigenvalues = eigenvalues[eigenvalues > 0]  # Filter out zero eigenvalues for log calculation
    return -np.sum(eigenvalues * np.log(eigenvalues))  # Von Neumann entropy

# Initialize EE values
EE_values = np.zeros(t)

# Run the walk for j iterations
for iteration in range(j):
    # Assign random disorder
    random_disorder = np.random.uniform(-1, 1, num_positions)
    r_values = 0.5 * (1 + disorder_strength * random_disorder)  # Scale disorder

    state = np.copy(initial_state)  # Reset state for each iteration

    # Loop over time steps
    for step in range(t):
        # Apply coin operator with site-specific disorder
        state = np.array([np.dot(G_gate(pos, r_values), state[pos]) for pos in range(num_positions)])
        state = apply_shift_operator(state)  # Apply shift operator

        # Calculate the reduced density matrix for the coin
        rho = compute_density_matrix(state)

        # Calculate Entanglement Entropy for this time step
        EE_values[step] += EE(rho)

# Average the EE values across iterations
EE_values /= j

# Plot the EE values as a function of time
plt.figure(figsize=(10, 6))
plt.plot(range(1, t + 1), EE_values, label="Entanglement Entropy")
plt.xlabel("Time (t)")
plt.ylabel("S")
plt.title(f"Entanglement Entropy as a function of time for Disordered Quantum Walk (W={disorder_strength})")
plt.legend()
plt.grid(True)
plt.show()
