import numpy as np

def G_1d(disorder, w):

    r_x = 0.5 * (1 + w * disorder)

    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ])
    return G

if __name__ == "__main__":

    pass