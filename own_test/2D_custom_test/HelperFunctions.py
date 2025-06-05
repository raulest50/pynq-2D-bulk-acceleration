import numpy as np

def create_tridiagonal_matrix(main_diagonal, offset_diagonal):
    """
    Creates a symmetric tridiagonal matrix from the main diagonal and offset diagonal.
    
    Parameters:
    -----------
    main_diagonal : array_like
        The values for the main diagonal of the matrix.
    offset_diagonal : array_like
        The values for the first sub-diagonal and first super-diagonal of the matrix.
        
    Returns:
    --------
    numpy.ndarray
        A symmetric tridiagonal matrix where:
        - The main diagonal contains the values from main_diagonal
        - The first sub-diagonal and first super-diagonal contain the values from offset_diagonal
        - All other elements are zero
        
    Examples:
    ---------
    >>> main_diag = np.array([2, 2, 2, 2])
    >>> offset_diag = np.array([-1, -1, -1])
    >>> create_tridiagonal_matrix(main_diag, offset_diag)
    array([[ 2, -1,  0,  0],
           [-1,  2, -1,  0],
           [ 0, -1,  2, -1],
           [ 0,  0, -1,  2]])
    """
    # Convert inputs to numpy arrays if they aren't already
    main_diagonal = np.asarray(main_diagonal)
    offset_diagonal = np.asarray(offset_diagonal)
    
    # Check that dimensions are compatible
    n = len(main_diagonal)
    if len(offset_diagonal) != n - 1:
        raise ValueError("Length of offset_diagonal must be one less than the length of main_diagonal")
    
    # Create a zero matrix of the appropriate size
    matrix = np.zeros((n, n))
    
    # Set the main diagonal
    np.fill_diagonal(matrix, main_diagonal)
    
    # Set the first sub-diagonal and first super-diagonal
    for i in range(n-1):
        matrix[i, i+1] = offset_diagonal[i]  # Super-diagonal
        matrix[i+1, i] = offset_diagonal[i]  # Sub-diagonal (symmetric)
    
    return matrix