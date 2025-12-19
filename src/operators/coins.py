import numpy as np

def G_1d(disorder: float, w: float) -> np.ndarray:

    r_x = 0.5 * (1 + w * disorder)

    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ])
    return G

if __name__ == "__main__":
    G1 = G_1d(1, 0)
    print(G1)