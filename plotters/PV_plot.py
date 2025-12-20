import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Please change path to output manually or from simluation directory directly.

# Change the name of the file needed to plot
df = pd.read_csv('../../output/data/2D_PV.csv')

time_steps = df.columns[1:-1]
list(time_steps)

t_dim = len(time_steps)

colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(t_dim)]

# Number of realizations (for error calculation)
j = df["j"].iloc[0]

legend_entries = []

# Plot each time step against Disorder Strength with error bars
for i, t in enumerate(time_steps):
    mean_values = df[t]
    std_dev = np.std(mean_values, ddof=1)  # Sample standard deviation
    error_bars = std_dev / np.sqrt(j)      # Standard error
    
    print(f"Standard deviation for {t}: {std_dev:.3f}")
    
    plt.errorbar(df["Disorder Strength"], mean_values, yerr=error_bars, fmt=".", linestyle="-", markersize=5, color=colors[i], linewidth=1.5, markerfacecolor=colors[i], capsize=4)
    
    legend_entries.append(f"{t}, σ={std_dev:.2f}")

math_p =  r'$\mathcal{P}$' 
# Add plot details
plt.xlabel("W")
plt.ylabel(math_p + "(W)")
plt.legend(legend_entries)

# Format ticks: inward-facing and same size on all four sides
plt.tick_params(direction='in', length=6, width=1.2, bottom=True, top=True, left=True, right=True)  # Major ticks on all sides

# Set minor ticks on all sides
plt.minorticks_on()
plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                bottom=True, top=True, left=True, right=True)  # Minor ticks on all sides

plt.title(f"Occupation of space as function of disorder strength for {j} realizations")
# Show plot
plt.show()
