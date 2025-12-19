import numpy as np

def shift_1d(state: np.ndarray) -> np.ndarray:
    new_state = np.zeros_like(state)
    new_state[1:, 0] = state[:-1, 0]
    new_state[:-1, 1] = state[1:, 1]
    return new_state

if __name__ == "__main__":

    pass