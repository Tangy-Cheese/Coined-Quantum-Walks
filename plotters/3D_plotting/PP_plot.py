import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('participation_data_3d_tester (2).csv')

# Plotting
plt.figure(figsize=(10, 7))

# Define columns corresponding to time steps
time_steps = ["t=20", "t=40", "t=60", "t=80", "t=100"]

# Plot each time step against Disorder Strength
for t in time_steps:
    plt.plot(df["Disorder Strength"], df[t], marker='o', linestyle='--', label=t)

# Add plot details
plt.xlabel("Disorder Strength (W)")
plt.ylabel("Participation Value")
plt.title("Participation Value vs Disorder Strength for Different Time Steps")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
