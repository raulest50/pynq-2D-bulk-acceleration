import numpy as np
from HelperFunctions import create_tridiagonal_matrix

# Test case 1: Simple 4x4 matrix
main_diag = np.array([2, 2, 2, 2])
offset_diag = np.array([-1, -1, -1])

matrix1 = create_tridiagonal_matrix(main_diag, offset_diag)
print("Test case 1 - 4x4 matrix:")
print(matrix1)
print()

# Test case 2: Different values on diagonals
main_diag2 = np.array([4, 5, 6, 7, 8])
offset_diag2 = np.array([1, 2, 3, 4])

matrix2 = create_tridiagonal_matrix(main_diag2, offset_diag2)
print("Test case 2 - 5x5 matrix with different values:")
print(matrix2)
print()

# Test case 3: Verify symmetry
print("Test case 3 - Verify symmetry:")
is_symmetric = np.allclose(matrix2, matrix2.T)
print(f"Is the matrix symmetric? {is_symmetric}")

# Test case 4: Verify it's truly tridiagonal
print("\nTest case 4 - Verify it's truly tridiagonal:")
# Create a larger matrix for better visualization
n = 6
main_diag3 = np.ones(n) * 3
offset_diag3 = np.ones(n-1) * -1
matrix3 = create_tridiagonal_matrix(main_diag3, offset_diag3)
print(matrix3)