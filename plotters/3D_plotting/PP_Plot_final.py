import numpy as np
import matplotlib.pyplot as plt

# Load the first dataset
data_1 = np.load(r"participation_data_3dFINALFINAL.npz")

# Load the second dataset containing t=100
data_2 = np.load(r"participation_data_3dt100j20.npz")  # Adjust filename as needed

# Recreate the disorder strength array (must match the simulation)
disorder_strength_3D = np.linspace(0, 1, 30)  # 3D

# Define the time step keys as saved in the npz files
time_steps_3D = ["t=20", "t=40", "t=60", "t=80"]  # 3D
time_step_t100 = "t=100"  # Key for the second file

# Define markers, line styles, and colors for each time step
markers = ["o", "s", "D", "^", 'D', 'X']  # Added 'X' for t=100
lines = ["solid", "dashed", (0, (3, 1, 1, 1, 1, 1)), "dotted", "-.", "--"]
colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "darkred", "#ff7f0e"]  # Added orange for t=100

# Number of samples for error calculation
N = 80  # For the first dataset
N_t100 = 50  # Adjust this based on the second dataset

plt.figure(figsize=(11, 7))

legend_entries = []

# Plot each time step's participation data versus disorder strength with error bars
for i, t in enumerate(time_steps_3D):
    mean_values = data_1[t]
    std_dev = np.std(mean_values, ddof=1)  # Sample standard deviation
    error_bars = std_dev / np.sqrt(N)      # Standard error

    print(f"Standard deviation for {t}: {std_dev:.5f}")

    plt.errorbar(disorder_strength_3D, mean_values, yerr=error_bars, fmt=markers[i], linestyle=lines[i], 
                 markersize=6, color=colors[i], linewidth=1.5, markerfacecolor=colors[i], capsize=4)
    legend_entries.append(f"{t}, σ={std_dev:.2f}")

# Now add t=100 data
mean_values_t100 = data_2[time_step_t100]
std_dev_t100 = np.std(mean_values_t100, ddof=1)  # Sample standard deviation
error_bars_t100 = std_dev_t100 / np.sqrt(N_t100)  # Standard error

print(f"Standard deviation for {time_step_t100}: {std_dev_t100:.5f}")

plt.errorbar(disorder_strength_3D, mean_values_t100, yerr=error_bars_t100, fmt=markers[-1], linestyle=lines[-1], 
             markersize=6, color=colors[-1], linewidth=1.5, markerfacecolor=colors[-1], capsize=4)

legend_entries.append(f"{time_step_t100}, σ={std_dev_t100:.2f}")

# Labels, legend, and tick adjustments
font = 14.4
math_p = r'$\mathcal{P}$'  # Unicode for 𝒫

plt.xlabel("W", fontsize=font)
plt.ylabel(math_p + "(W)", fontsize=font)
plt.legend(legend_entries, fontsize=font)

# Remove gridlines
plt.grid(False)

# Format ticks: inward-facing and same size on all four sides
plt.tick_params(direction='in', length=6, width=1.2, labelsize=font, 
                bottom=True, top=True, left=True, right=True)

# Set minor ticks on all sides
plt.minorticks_on()
plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                bottom=True, top=True, left=True, right=True)

# Show the plot
plt.show()
