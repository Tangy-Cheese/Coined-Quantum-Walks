import numpy as np
import concurrent.futures

# Parameters
TIME_STEPS = [100]        # List of time steps to simulate
NUM_ITER = 50                            # Number of realizations per disorder strength
DISORDER_STRENGTH = np.linspace(0, 1, 30)   # Array of disorder strengths to explore
SIZE = 2 * max(TIME_STEPS) + 1              # Lattice size in each dimension
INITIAL_POSITION = SIZE // 2              # Center of the lattice

def precompute_G_gate_3d(Nx, r1, r2, r3):
    """
    Precompute the coin gate for every site in the 3D lattice in a vectorized fashion.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    sqrt_r1   = np.sqrt(r1)[..., np.newaxis, np.newaxis]
    sqrt_1mr1 = np.sqrt(1 - r1)[..., np.newaxis, np.newaxis]
    G1 = sqrt_r1 * Z + sqrt_1mr1 * X

    sqrt_r2   = np.sqrt(r2)[..., np.newaxis, np.newaxis]
    sqrt_1mr2 = np.sqrt(1 - r2)[..., np.newaxis, np.newaxis]
    G2 = sqrt_r2 * Z + sqrt_1mr2 * X

    sqrt_r3   = np.sqrt(r3)[..., np.newaxis, np.newaxis]
    sqrt_1mr3 = np.sqrt(1 - r3)[..., np.newaxis, np.newaxis]
    G3 = sqrt_r3 * Z + sqrt_1mr3 * X

    # Compute Kronecker products in a vectorized way
    c12 = np.einsum('...ij,...kl->...ikjl', G1, G2).reshape(Nx, Nx, Nx, 4, 4)
    c123 = np.einsum('...ij,...kl->...ikjl', c12, G3).reshape(Nx, Nx, Nx, 8, 8)
    return c123

def apply_shift_operator_3d(state):
    """
    Apply the shift operator in 3D using np.roll for each coin component.
    """
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

def run_quantum_walk_single(W, t):
    """
    Run a single realization of the quantum walk for a given disorder strength W and time t.
    Returns the participation ratio computed as:
        1 / [ (sum |ψ|^4) * ((2*t)^3 + 1) ]
    """
    # Generate random disorder arrays in one call (3 arrays for r1, r2, r3)
    rd = np.random.uniform(-1, 1, (3, SIZE, SIZE, SIZE))
    r1 = 0.5 * (1.0 + W * rd[0])
    r2 = 0.5 * (1.0 + W * rd[1])
    r3 = 0.5 * (1.0 + W * rd[2])
    
    # Precompute the coin gate for every lattice site
    coin_xyz = precompute_G_gate_3d(SIZE, r1, r2, r3)
    
    # Initialize state: walker at the center with a defined superposition
    state = np.zeros((SIZE, SIZE, SIZE, 8), dtype=np.complex128)
    state[INITIAL_POSITION, INITIAL_POSITION, INITIAL_POSITION, :] = [1/(2*np.sqrt(2)),1j/(2*np.sqrt(2)),-1/(2*np.sqrt(2)),1j/(2*np.sqrt(2)),1j/(2*np.sqrt(2)), -1/(2*np.sqrt(2)),-1j/(2*np.sqrt(2)),-1/(2*np.sqrt(2))]
    # Time evolution loop
    

    for step in range(t):
        state = np.einsum('...ij,...j->...i', coin_xyz, state)
        state = apply_shift_operator_3d(state)
    
    # Calculate the participation ratio
    participation_ratio = 1 / (np.sum(np.abs(state)**4) * ((2 * t)**3 + 1))
    return participation_ratio

def aggregate_quantum_walk_participation():
    """
    Run simulations for all time steps and disorder strengths.
    For each time step t and each disorder strength W, NUM_ITER realizations are run in parallel.
    The average participation ratio is then stored in a dictionary and saved as an NPZ file.
    """
    results = {}

    for t in TIME_STEPS:
        print(f"Starting simulations for time step t = {t}")
        pr_values_for_t = []
        for i, W in enumerate(DISORDER_STRENGTH):
            print(f"  Processing disorder strength W = {W:.3f} ({i+1}/{len(DISORDER_STRENGTH)}) for t = {t}")
            # Run NUM_ITER independent realizations in parallel for the current disorder strength W
            with concurrent.futures.ProcessPoolExecutor() as executor:
                pr_results = list(executor.map(run_quantum_walk_single, [W]*NUM_ITER, [t]*NUM_ITER))
            avg_pr = np.mean(pr_results)
            pr_values_for_t.append(avg_pr)
            print(f"    Completed W = {W:.3f}: Avg Participation Ratio = {avg_pr:.6e}")
        # Store the results as a NumPy array for the given time step
        results[f't={t}'] = np.array(pr_values_for_t)
        print(f"Completed simulations for time step t = {t}\n")
    
    output_filename = 'participation_data_3dt100.npz'
    np.savez(output_filename, **results)
    print(f"Saved aggregated participation ratio data to {output_filename}")

if __name__ == "__main__":
    aggregate_quantum_walk_participation()

