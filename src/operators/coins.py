import numpy as np

def G_1d(disorder: float, W: float) -> np.ndarray:

    r_x = 0.5 * (1 + W * disorder)

    G = np.array([
        [np.sqrt(r_x), np.sqrt(1 - r_x)],
        [np.sqrt(1 - r_x), -np.sqrt(r_x)]
    ], dtype=np.complex128)
    return G

def G_2d(disorder: np.ndarray, W: float) -> np.ndarray:

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