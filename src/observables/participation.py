import numpy as np

def participation_value(state: np.ndarray, t: int) -> float:
    return 1 / (np.sum(np.abs(state)**4) * (2*t + 1))

if __name__ == "__main__":

    pass