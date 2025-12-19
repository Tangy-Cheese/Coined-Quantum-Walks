import numpy as np

def G_1d(disorder: float, W: float) -> np.ndarray:
    """
    Construct the 1D disordered coin operator for a coined quantum walk.

    This function implements the quenched disordered Hadamard-type coin,
    where the disorder enters through a site-dependent parameter
    r_x = 1/2 (1 + W ξ_x). For W = 0 the operator reduces to the standard
    Hadamard coin, while increasing W introduces stronger disorder.

    Parameters
    ----------
    disorder : float
        Local disorder value ξ_x sampled from a uniform distribution
        in the interval [-1, 1].
    W : float
        Disorder strength parameter controlling the magnitude of the
        quenched disorder.

    Returns
    -------
    np.ndarray
        A 2 x 2 complex-valued unitary matrix representing the disordered
        coin operator at a single lattice site.
    """
    r_x = 0.5 * (1 + W * disorder)

    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ], dtype=np.complex128)
    return G

def G_2d(disorder: np.ndarray, W: float) -> np.ndarray:
    """
    Construct the 2D disordered coin operator for a coined quantum walk
    on a square lattice.

    The 2D coin is formed as the tensor product of two independent
    1D disordered coin operators acting along the x and y directions.
    The disorder enters through site-dependent parameters
    r_x = 1/2 (1 + W ξ_x) and r_y = 1/2 (1 + W ξ_y), where ξ_x and ξ_y
    are independent random variables sampled from a uniform
    distribution in the interval [-1, 1].

    Parameters
    ----------
    disorder : np.ndarray
        Array of length 2 containing the local disorder values
        [ξ_x, ξ_y] for the x and y directions.
    W : float
        Disorder strength parameter controlling the magnitude of the
        quenched disorder.

    Returns
    -------
    np.ndarray
        A 4 x 4 complex-valued unitary matrix representing the
        disordered 2D coin operator.
    """
    rx = 0.5 * (1 + W * disorder[0])
    ry = 0.5 * (1 + W * disorder[1])
    
    Gx = np.array([
        [np.sqrt(rx), np.sqrt(1-rx)],
        [np.sqrt(1-rx) , -np.sqrt(rx)]
    ], dtype=np.complex128)

    Gy = np.array([
        [np.sqrt(ry), np.sqrt(1-ry)],
        [np.sqrt(1-ry) , -np.sqrt(ry)]
    ], dtype=np.complex128)

    G = np.kron(Gx, Gy)
    return G

def G_3d(disorder: np.ndarray, W: float) -> np.ndarray:
    """
    Construct the 3D disordered coin operator for a coined quantum walk
    on a cubic lattice.

    The 3D coin is formed as the tensor product of three independent
    1D disordered coin operators acting along the x, y, and z directions.
    The disorder enters through site-dependent parameters
    r_x = 1/2 (1 + W ξ_x), r_y = 1/2 (1 + W ξ_y), and
    r_z = 1/2 (1 + W ξ_z), where ξ_x, ξ_y, and ξ_z are independent
    random variables sampled from a uniform distribution in the
    interval [-1, 1].

    Parameters
    ----------
    disorder : np.ndarray
        Array of length 3 containing the local disorder values
        [ξ_x, ξ_y, ξ_z] for the x, y, and z directions.
    W : float
        Disorder strength parameter controlling the magnitude of the
        quenched disorder.

    Returns
    -------
    np.ndarray
        An 8 x 8 complex-valued unitary matrix representing the
        disordered 3D coin operator.
    """
    rx = 0.5 * (1 + W * disorder[0])
    ry = 0.5 * (1 + W * disorder[1])
    rz = 0.5 * (1 + W * disorder[2])
    
    Gx = np.array([
        [np.sqrt(rx), np.sqrt(1 - rx)],
        [np.sqrt(1 - rx), -np.sqrt(rx)]
    ], dtype=np.complex128)

    Gy = np.array([
        [np.sqrt(ry), np.sqrt(1 - ry)],
        [np.sqrt(1 - ry), -np.sqrt(ry)]
    ], dtype=np.complex128)

    Gz = np.array([
        [np.sqrt(rz), np.sqrt(1 - rz)],
        [np.sqrt(1 - rz), -np.sqrt(rz)]
    ], dtype=np.complex128)
    
    G = np.kron(np.kron(Gx, Gy), Gz)
    return G

if __name__ == "__main__":
    G1 = G_1d(1, 0)
    print(G1)