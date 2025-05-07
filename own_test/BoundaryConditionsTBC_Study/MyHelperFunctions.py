import numpy as np

def create_tridiagonal_matrix(n, md, od):
    """
    Creates an NxN tridiagonal matrix.
    
    Args:
        n (int): Size of the matrix
        md (float): Value for the main diagonal
        od (float): Value for the offset diagonals
    
    Returns:
        numpy.ndarray: NxN tridiagonal matrix
    """
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, md)
    np.fill_diagonal(matrix[1:], od)  # superdiagonal
    np.fill_diagonal(matrix[:, 1:], od)  # subdiagonal
    return matrix


# Analytical solution
def u_gaussian(x, t, σ=0.3, c=1):
    return np.exp( -((x-4)-c*t)**2/(2*σ**2) )


def ut_gaussian(x, t, σ=0.3, c=1):
    return c * ((x - 4) - c * t) / (σ ** 2) * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))


def utt_gaussian(x, t, σ=0.3, c=1):
    term1 = -c ** 2 / (σ ** 2)
    term2 = 1 - ((x - 4) - c * t) ** 2 / (σ ** 2)
    return term1 * term2 * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))


def uttt_gaussian(x, t, σ=0.3, c=1):
    term1 = c ** 3 / (σ ** 4)
    term2 = 3 * ((x - 4) - c * t) - ((x - 4) - c * t) ** 3 / (σ ** 2)
    return term1 * term2 * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))
