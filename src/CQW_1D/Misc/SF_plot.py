import numpy as np
import matplotlib.pyplot as plt

# Load the data (two columns: time, fidelity)
data = np.loadtxt("fidelity_data.txt")
time = data[:, 0]
fidelity = data[:, 1]

# Plot the fidelity as a function of time
plt.figure(figsize=(10, 6))
plt.plot(time, fidelity, marker='o', label="State Fidelity")
plt.xlabel("Time (t)")
plt.ylabel("F(t)")
plt.title("State Fidelity vs. Time for Disordered Quantum Walk (W=1)")
plt.legend()
plt.grid(True)
plt.show()