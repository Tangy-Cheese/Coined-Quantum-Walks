import numpy as np

def participation_1d(state: np.ndarray, t: int) -> float:
    """
    Compute the participation ratio for a 1D coined quantum walk.

    The participation ratio quantifies the effective number of
    lattice sites occupied by the quantum walk. It is defined as
    the inverse of the sum of the squared probability density,
    normalised by the 1D system size (2t + 1).

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N, 2), where N is the
        number of lattice sites and the second axis corresponds
        to the coin states.
    t : int
        Number of time steps evolved, used to determine the
        effective 1D system size.

    Returns
    -------
    float
        The participation ratio for the 1D quantum walk at time t.
        Values close to zero indicate strong localisation, while
        larger values correspond to more extended states.
    """ 
    return 1 / (np.sum(np.abs(state)**4) * (2*t + 1))


def participation_2d(state: np.ndarray, t: int) -> float:
    """
    Compute the participation ratio for a 2D coined quantum walk.

    The participation ratio is defined as the inverse of the sum
    of the squared probability density over the lattice, normalised
    by the effective 2D system size (2t)^2 + 1.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N_x, N_y, 4), where
        (N_x, N_y) define the lattice dimensions and the third
        axis corresponds to the four coin states.
    t : int
        Number of time steps evolved, used to determine the
        effective 2D system size.

    Returns
    -------
    float
        The participation ratio for the 2D quantum walk at time t.
        Smaller values indicate localisation, while larger values
        indicate diffusive or extended behaviour.
    """
    return 1 / (np.sum(np.abs(state)**4) * ((2 * t)**2 + 1))


def participation_3d(state: np.ndarray, t: int) -> float:
    """
    Compute the participation ratio for a 3D coined quantum walk.

    The participation ratio is defined as the inverse of the sum
    of the squared probability density over the lattice, normalised
    by the effective 3D system size (2t)^3 + 1.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N_x, N_y, N_z, 8), where
        (N_x, N_y, N_z) define the lattice dimensions and the last
        axis corresponds to the eight coin states.
    t : int
        Number of time steps evolved, used to determine the
        effective 3D system size.

    Returns
    -------
    float
        The participation ratio for the 3D quantum walk at time t.
        Smaller values indicate localisation, while larger values
        indicate extended or diffusive behaviour.
    """
    return 1 / (np.sum(np.abs(state)**4) * ((2 * t)**3 + 1))


if __name__ == "__main__":

    pass