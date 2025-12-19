import numpy as np

def shift_1d(state: np.ndarray) -> np.ndarray:
    """
    Apply the 1D conditional shift operator for a coined quantum walk.

    The shift propagates the walker according to the coin state:
    coin state 0 moves the walker one site to the right (+x),
    and coin state 1 moves the walker one site to the left (-x).
    Amplitudes that would move outside the lattice are discarded,
    corresponding to hard-wall boundary conditions.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N, 2), where N is the
        number of lattice sites and the second axis corresponds
        to the two coin states.

    Returns
    -------
    np.ndarray
        Updated quantum walk state after applying the 1D shift
        operator. The returned array has the same shape as the
        input state.
    """ 
    new_state = np.zeros_like(state)

    new_state[1:, 0] = state[:-1, 0]
    new_state[:-1, 1] = state[1:, 1]
    return new_state


def shift_2d(state: np.ndarray) -> np.ndarray:
    """
    Apply the 2D conditional shift operator for a coined quantum walk
    on a square lattice with diagonal propagation.

    The shift moves the walker conditionally on the coin state along
    the lattice diagonals. The coin states are interpreted as:
        0 -> (+x, -y)
        1 -> (-x, -y)
        2 -> (+x, +y)
        3 -> (-x, +y)

    Amplitudes that would move outside the lattice are discarded,
    corresponding to hard-wall boundary conditions.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N_x, N_y, 4), where
        (N_x, N_y) define the lattice dimensions and the third
        axis corresponds to the four coin states.

    Returns
    -------
    np.ndarray
        Updated quantum walk state after applying the 2D shift
        operator. The returned array has the same shape as the
        input state.
    """  
    new_state = np.zeros_like(state)
    
    new_state[:-1, 1:, 0] = state[1:, :-1, 0]  
    new_state[1:, 1:, 1] = state[:-1, :-1, 1]  
    new_state[:-1, :-1, 2] = state[1:, 1:, 2]  
    new_state[1:, :-1, 3] = state[:-1, 1:, 3]  
    return new_state


def shift_3d(state):
    """
    Apply the 3D conditional shift operator for a coined quantum walk
    on a cubic lattice with diagonal propagation.

    The shift moves the walker conditionally on the coin state along
    the body diagonals of the cube. The coin states are interpreted as:
        0 -> (+x, +y, +z)
        1 -> (+x, +y, -z)
        2 -> (+x, -y, +z)
        3 -> (+x, -y, -z)
        4 -> (-x, +y, +z)
        5 -> (-x, +y, -z)
        6 -> (-x, -y, +z)
        7 -> (-x, -y, -z)

    Amplitudes that would move outside the lattice are discarded,
    corresponding to hard-wall boundary conditions.

    Parameters
    ----------
    state : np.ndarray
        Quantum walk state array of shape (N_x, N_y, N_z, 8), where
        (N_x, N_y, N_z) define the cubic lattice dimensions and the
        last axis corresponds to the eight coin states.

    Returns
    -------
    np.ndarray
        Updated quantum walk state after applying the 3D shift
        operator. The returned array has the same shape as the
        input state.
    """
    new_state = np.zeros_like(state)
    new_state[1:, 1:, 1:, 0] = state[:-1, :-1, :-1, 0]  
    new_state[1:, 1:, :-1, 1] = state[:-1, :-1, 1:, 1]  
    new_state[1:, :-1, 1:, 2] = state[:-1, 1:, :-1, 2]  
    new_state[1:, :-1, :-1, 3] = state[:-1, 1:, 1:, 3]  
    new_state[:-1, 1:, 1:, 4] = state[1:, :-1, :-1, 4]  
    new_state[:-1, 1:, :-1, 5] = state[1:, :-1, 1:, 5]  
    new_state[:-1, :-1, 1:, 6] = state[1:, 1:, :-1, 6]  
    new_state[:-1, :-1, :-1, 7] = state[1:, 1:, 1:, 7]  
    return new_state


if __name__ == "__main__":

    pass