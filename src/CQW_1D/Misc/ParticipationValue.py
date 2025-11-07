import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 100  # Number of time steps
size = 2 * t + 1  # Grid size
initial_position = size // 2  # Starting at the center
j = 1  # Number of iterations for averaging
disorder_strength = np.linspace(0, 1, 10) # Disorder strength parameter W


# Initial state: walker at the center in a superposition
state = np.zeros((size, 2), dtype=complex)  # State has 4 components (up, down, left, right)
state[initial_position, :] = [1/np.sqrt(2), 1j/np.sqrt(2)]

# Coin operator with disorder
def G_gate(disorder, w):
    r_x = 0.5 * (1 + w * disorder)
    G = np.array([
        [np.sqrt(r_x),  np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ])
    return G

# Shift operator
def apply_shift(state):
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]  
    new_state[:-1, 1] = state[1:, 1]  
    return new_state

participation_values = []

for W in disorder_strength:
    total_participation = 0
    total_normalised_participation = 0  # Reset for each W

# Run the walk for j iterations
    for iteration in range(j):
        # Assign random disorder to each grid point
        random_disorder = np.random.uniform(-1, 1, (size))
        state = np.zeros((size, 2), dtype=complex)
        state[initial_position, :] = [1/np.sqrt(2), 1j/np.sqrt(2)]

        # Loop over time steps
        for step in range(t):
            # Apply coin operator with site-specific disorder
            for x in range(size):
                    state[x, :] = np.dot(G_gate(random_disorder[x], W), state[x, :])
        
            # Apply shift operator
            state = apply_shift(state)

        # Compute participation ratio
            participation_value = 1/(np.sum(np.abs(state)**4)*(size))
            total_participation += participation_value
        
        #total_normalised_participation = (1/(size**2))*(total_participation)

    # Average over j iterations and store
    participation_values.append(total_participation / j)

# Plot participation value against disorder strength W
plt.figure(figsize=(8, 6))
plt.plot(disorder_strength, participation_values, marker='o', linestyle='--')
plt.xlabel("Disorder Strength (W)")
plt.ylabel("Participation Value")
plt.title("Participation Value vs Disorder Strength")
plt.grid(True)
plt.show()