import numpy as np

def entanglement_1d(state: np.ndarray) -> float:
    """
    Compute the entanglement entropy between the coin and position
    degrees of freedom for a 1D coined quantum walk.

    The entanglement entropy is defined as the von Neumann entropy
    of the reduced coin density matrix obtained by tracing out
    the position degrees of freedom.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N, 2), where N is the
        number of lattice sites and the second axis corresponds
        to the coin states.

    Returns
    -------
    float
        The entanglement entropy S = -Tr(ρ_c log ρ_c).
    """
    rho_c = state.conj().T @ state

    eigenvalues = np.linalg.eigvalsh(rho_c)
    eigenvalues = eigenvalues[eigenvalues > 0]

    return -np.sum(eigenvalues * np.log(eigenvalues))

if __name__ == "__main__":

    pass