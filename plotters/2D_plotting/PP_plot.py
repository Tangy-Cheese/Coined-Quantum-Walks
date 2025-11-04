import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('participation_data_2d_final_1000.csv')

# Plotting
plt.figure(figsize=(10.5, 7))

# Define columns corresponding to time steps
time_steps = ["t=20", "t=40", "t=60", "t=80", "t=100"]

# Define different markers for each time step
markers = ["o", "s", "D", "^", "v"]
lines = ["solid", "dashed", (0,(3, 1, 1, 1, 1, 1)), "dotted", "dashdot"]
colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

# Plot each time step against Disorder Strength
for i, t in enumerate(time_steps):
    plt.plot(df["Disorder Strength"], df[t], marker=markers[i], linestyle=lines[i], markersize = 6, label=t, color = colors[i], linewidth = "0.95",
    markerfacecolor = colors[i]
    
    )

# Add plot details
plt.xlabel("W", fontsize = "15")
plt.ylabel("Participation Value", fontsize = "15")
plt.legend(fontsize = "15")

# Format ticks: inward-facing on all four sides
plt.tick_params(direction='in', length=6, width=1.2, labelsize=15, 
                bottom=True, top=True, left=True, right=True)  # Major ticks on all sides

# Set minor ticks
plt.minorticks_on()
plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                bottom=True, top=True, left=True, right=True)  # Minor ticks on all sides


# Show plot
plt.show()

