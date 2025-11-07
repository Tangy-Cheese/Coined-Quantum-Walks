import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Parameters
time_steps = [20, 40, 60, 80, 100]
size = 2 * max(time_steps) + 1
initial_position = size // 2
j = 125
disorder_strength = np.linspace(0, 1, 50)

# Initial state: walker at the center in a superposition
initial_state = np.zeros((size, size, size, 8), dtype=np.complex128)
initial_state[initial_position, initial_position, initial_position, :] = [
    1/(2*np.sqrt(2)), (1j)/(2*np.sqrt(2)), (1j)/(2*np.sqrt(2)),
    -1/(2*np.sqrt(2)), (1j)/(2*np.sqrt(2)), -1/(2*np.sqrt(2)),
    -1/(2*np.sqrt(2)), -1j/(2*np.sqrt(2))
]

def precompute_G_gate_3d(Nx, r1, r2, r3):
    """Vectorized precomputation of the coin gate for every site."""
    # Define coin matrices (Pauli-like)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    sqrt_r1 = np.sqrt(r1)[..., np.newaxis, np.newaxis]
    sqrt_1mr1 = np.sqrt(1 - r1)[..., np.newaxis, np.newaxis]
    G1 = sqrt_r1 * Z + sqrt_1mr1 * X
    
    sqrt_r2 = np.sqrt(r2)[..., np.newaxis, np.newaxis]
    sqrt_1mr2 = np.sqrt(1 - r2)[..., np.newaxis, np.newaxis]
    G2 = sqrt_r2 * Z + sqrt_1mr2 * X
    
    sqrt_r3 = np.sqrt(r3)[..., np.newaxis, np.newaxis]
    sqrt_1mr3 = np.sqrt(1 - r3)[..., np.newaxis, np.newaxis]
    G3 = sqrt_r3 * Z + sqrt_1mr3 * X
    
    # Compute the Kronecker products in a vectorized manner
    c12 = np.einsum('...ij,...kl->...ikjl', G1, G2).reshape(Nx, Nx, Nx, 4, 4)
    c123 = np.einsum('...ij,...kl->...ikjl', c12, G3).reshape(Nx, Nx, Nx, 8, 8)
    return c123

def apply_shift_operator_3d(state):
    """Vectorized shift operator using np.roll."""
    shifted_list = [
        np.roll(np.roll(np.roll(state[..., 0], +1, axis=0), +1, axis=1), +1, axis=2),
        np.roll(np.roll(np.roll(state[..., 1], +1, axis=0), +1, axis=1), -1, axis=2),
        np.roll(np.roll(np.roll(state[..., 2], +1, axis=0), -1, axis=1), +1, axis=2),
        np.roll(np.roll(np.roll(state[..., 3], +1, axis=0), -1, axis=1), -1, axis=2),
        np.roll(np.roll(np.roll(state[..., 4], -1, axis=0), +1, axis=1), +1, axis=2),
        np.roll(np.roll(np.roll(state[..., 5], -1, axis=0), +1, axis=1), -1, axis=2),
        np.roll(np.roll(np.roll(state[..., 6], -1, axis=0), -1, axis=1), +1, axis=2),
        np.roll(np.roll(np.roll(state[..., 7], -1, axis=0), -1, axis=1), -1, axis=2)
    ]
    return np.stack(shifted_list, axis=-1)

def run_quantum_walk(W, t):
    """
    Simulate the quantum walk for a given disorder strength W and time t
    using the vectorized coin and shift operators.
    """
    total_participation = 0.0

    for _ in range(j):
        # Generate three random arrays for the coin operator disorder (one per coin)
        rd1 = np.random.uniform(-1, 1, (size, size, size))
        rd2 = np.random.uniform(-1, 1, (size, size, size))
        rd3 = np.random.uniform(-1, 1, (size, size, size))
        r1 = 0.5 * (1.0 + W * rd1)
        r2 = 0.5 * (1.0 + W * rd2)
        r3 = 0.5 * (1.0 + W * rd3)
        
        # Precompute the coin gate for every site
        coin_xyz = precompute_G_gate_3d(size, r1, r2, r3)  # shape: (size, size, size, 8, 8)
        
        # Reset the state to the initial condition
        state = np.zeros((size, size, size, 8), dtype=np.complex128)
        state[initial_position, initial_position, initial_position, :] = [
            1/2, (1j)/2, (1j)/2, -1/2, (1j)/2, -1/2, -1/2, -1j/2
        ]
        
        # Time evolution
        for step in range(t):
            # Apply the coin operator at all sites simultaneously
            state = np.einsum('...ij,...j->...i', coin_xyz, state)
            # Apply the shift operator
            state = apply_shift_operator_3d(state)
        
        # Compute the participation ratio
        participation_value = 1 / (np.sum(np.abs(state)**4) * ((2 * t)**3 + 1))
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
df.to_csv('participation_data_3d_finale.csv')
