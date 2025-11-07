import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
disorder_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5 different disorder strengths
j_max = 500             # Number of disorder realizations
t = 100                 # Number of time steps
num_positions = 2 * t + 1  # Number of positions (ensuring odd for symmetry)

# Path (adjust as needed)
data_dir = r"C:\Users\Louis\Physics_Project\Programming\Data\Data_Mined_Sj"

# Function to compute the single-particle (coin) density matrix
def compute_density_matrix(state):
    # state is of shape (num_positions, 2)
    rho_c = np.zeros((2, 2), dtype=complex)
    for pos in range(state.shape[0]):
        coin_state = state[pos, :].reshape(2, 1)
        rho_c += np.outer(coin_state, coin_state.conj())
    return rho_c

# Function to compute the entanglement entropy (von Neumann entropy)
def compute_entanglement_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out any non-positive eigenvalues due to numerical precision
    eigenvalues = eigenvalues[eigenvalues > 0]
    return -np.sum(eigenvalues * np.log(eigenvalues))

plt.figure(figsize=(10, 6))

# Loop over disorder strengths and calculate entanglement entropy vs time
for W in disorder_strengths:
    EE_sum_all_j = np.zeros(t)
    num_realizations_used = 0

    for j in range(j_max):
        # Construct filename
        filename = f"disorder_data_W{W:.3f}_j={j}_t={t}.npz"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found. Skipping.")
            continue

        # Load data
        data = np.load(file_path)
        wavefunction = data["wavefunction"]  # Shape: (num_positions, 2, t)

        # Compute entanglement entropy at each time step
        for step in range(t):
            # Extract wavefunction at time step 'step'
            wf_step = wavefunction[:, :, step]  # (num_positions, 2)
            rho = compute_density_matrix(wf_step)
            EE_sum_all_j[step] += compute_entanglement_entropy(rho)

        num_realizations_used += 1

    # Average over the number of realizations used
    if num_realizations_used > 0:
        EE_sum_all_j /= num_realizations_used
        # Plot the entanglement entropy vs time for this disorder strength
        plt.plot(range(1, t + 1), EE_sum_all_j, label=f"W={W:.2f}")

plt.xlabel("Time (t)")
plt.ylabel("Entanglement Entropy (S)")
plt.title(f"Entanglement Entropy vs Time for Various Disorder Strengths\nAveraged Over {j_max} Realizations")
plt.legend()
plt.grid(True)
plt.show()
