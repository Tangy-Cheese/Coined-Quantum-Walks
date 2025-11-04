import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV file
df = pd.read_csv('participation_data_finale.csv')

# Define columns corresponding to time steps
time_steps = ["t=25", "t=50", "t=100", "t=250", "t=500", "t=1000"]

# Define different markers for each time step
markers = ["o", "s", "D", "^", "v", ">"]
lines = ["solid", "dashed", (0, (3, 1, 1, 1, 1, 1)), "dotted", "dashdot", (5, (10, 3))]
colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# Number of samples (for error calculation)
N = 2000

plt.figure(figsize=(11,7))

legend_entries = []

# Plot each time step against Disorder Strength with error bars
for i, t in enumerate(time_steps):
    mean_values = df[t]
    std_dev = np.std(mean_values, ddof=1)  # Sample standard deviation
    error_bars = std_dev / np.sqrt(N)      # Standard error
    
    print(f"Standard deviation for {t}: {std_dev:.3f}")
    
    plt.errorbar(df["Disorder Strength"], mean_values, yerr=error_bars, fmt=markers[i], linestyle=lines[i], 
                 markersize=6, color=colors[i], linewidth=1.5, markerfacecolor=colors[i], capsize=4)
    
    legend_entries.append(f"{t}, σ={std_dev:.2f}")

font = 14.4
math_p =  r'$\mathcal{P}$'  # Unicode for 𝒫
# Add plot details
plt.xlabel("W", fontsize=font)
plt.ylabel(math_p + "(W)", fontsize=font)
plt.legend(legend_entries, fontsize=font)

# Remove gridlines
plt.grid(False)

# Format ticks: inward-facing and same size on all four sides
plt.tick_params(direction='in', length=6, width=1.2, labelsize=font, 
                bottom=True, top=True, left=True, right=True)  # Major ticks on all sides

# Set minor ticks on all sides
plt.minorticks_on()
plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                bottom=True, top=True, left=True, right=True)  # Minor ticks on all sides

# Show plot
plt.show()
