import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Parameters
time_steps = [25, 50, 75, 100, 250, 500]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 250
disorder_strength = np.linspace(0, 1, 50)

# Initial state: walker at the center in a superposition
initial_state = np.zeros((size, size, 4), dtype=np.complex128)
initial_state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, -1/2]

def precompute_G_gate_2d(size, r1, r2):
    """
    Vectorized precomputation of the coin gate for every site in 2D.
    For each site, we construct two 2×2 coin matrices G1 and G2 and take their tensor product.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    sqrt_r1   = np.sqrt(r1)[..., np.newaxis, np.newaxis]
    sqrt_1mr1 = np.sqrt(1 - r1)[..., np.newaxis, np.newaxis]
    G1 = sqrt_r1 * Z + sqrt_1mr1 * X  # shape: (size, size, 2, 2)
    
    sqrt_r2   = np.sqrt(r2)[..., np.newaxis, np.newaxis]
    sqrt_1mr2 = np.sqrt(1 - r2)[..., np.newaxis, np.newaxis]
    G2 = sqrt_r2 * Z + sqrt_1mr2 * X  # shape: (size, size, 2, 2)
    
    # Compute the tensor product for each site.
    # This creates an array with shape (size, size, 2, 2, 2, 2) that we reshape into (size, size, 4, 4)
    coin = np.einsum('...ij,...kl->...ikjl', G1, G2).reshape(size, size, 4, 4)
    return coin

def apply_shift_operator_2d(state):
    """
    Vectorized shift operator for the 2D quantum walk using np.roll.
    The shifts are chosen to mimic:
      - Coin 0: new_state[x,y] = state[x+1, y-1]  → shift=(-1, +1)
      - Coin 1: new_state[x,y] = state[x-1, y-1]  → shift=(+1, +1)
      - Coin 2: new_state[x,y] = state[x+1, y+1]  → shift=(-1, -1)
      - Coin 3: new_state[x,y] = state[x-1, y+1]  → shift=(+1, -1)
    (Note that np.roll applies periodic shifts, so boundaries become periodic.)
    """
    shifted_list = [
        np.roll(state[..., 0], shift=(-1, +1), axis=(0, 1)),
        np.roll(state[..., 1], shift=(+1, +1), axis=(0, 1)),
        np.roll(state[..., 2], shift=(-1, -1), axis=(0, 1)),
        np.roll(state[..., 3], shift=(+1, -1), axis=(0, 1))
    ]
    return np.stack(shifted_list, axis=-1)

def run_quantum_walk(W, t):
    """
    Simulate the 2D quantum walk for a given disorder strength W and time t,
    using the vectorized coin and shift operators.
    """
    total_participation = 0.0

    for _ in range(j):
        # Generate two random disorder fields (one for each coin operator)
        rd1 = np.random.uniform(-1, 1, (size, size))
        rd2 = np.random.uniform(-1, 1, (size, size))
        r1 = 0.5 * (1 + W * rd1)
        r2 = 0.5 * (1 + W * rd2)
        
        # Precompute the coin operator for every site (shape: size x size x 4 x 4)
        coin = precompute_G_gate_2d(size, r1, r2)
        
        # Reset state
        state = np.zeros((size, size, 4), dtype=np.complex128)
        state[initial_position, initial_position, :] = [1/2, (1j)/2, (1j)/2, -1/2]
        
        # Time evolution
        for step in range(t):
            # Apply coin operator simultaneously at every site
            state = np.einsum('...ij,...j->...i', coin, state)
            # Apply the shift operator
            state = apply_shift_operator_2d(state)
        
        # Compute participation ratio
        participation_value = 1 / (np.sum(np.abs(state)**4) * ((2 * t)**2 + 1))
        total_participation += participation_value

    return total_participation / j

# Use parallel processing for different disorder strengths
data = {}
for t in time_steps:
    participation_values = Parallel(n_jobs=-1)(
        delayed(run_quantum_walk)(W, t) for W in disorder_strength
    )
    data[f't={t}'] = participation_values

# Save results to CSV
df = pd.DataFrame(data, index=disorder_strength)
df.index.name = 'Disorder Strength'
df.to_csv('participation_data_2d_finale.csv')
