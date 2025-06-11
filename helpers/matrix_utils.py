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