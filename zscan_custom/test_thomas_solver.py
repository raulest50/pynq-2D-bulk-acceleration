import numpy as np
from operator_solvers import custom_thomas_solver

def test_custom_thomas_solver():
    """
    Test the custom_thomas_solver function with a simple example.
    """
    # Create a test case with known solution
    n = 5
    dp = 2.0  # Main diagonal value
    dp1 = 2.5  # First element
    dp2 = 2.5  # Last element
    do = -1.0  # Off-diagonal value
    
    # Create the full matrix for verification
    A = np.zeros((n, n))
    
    # Fill the main diagonal
    A[0, 0] = dp1
    for i in range(1, n-1):
        A[i, i] = dp
    A[n-1, n-1] = dp2
    
    # Fill the off-diagonals
    for i in range(n-1):
        A[i, i+1] = do  # Upper diagonal
        A[i+1, i] = do  # Lower diagonal
    
    # Create a right-hand side vector
    b = np.ones(n)
    
    # Solve using custom_thomas_solver
    x_custom = custom_thomas_solver(dp, dp1, dp2, do, b)
    
    # Solve using numpy's linalg.solve for comparison
    x_numpy = np.linalg.solve(A, b)
    
    # Check if the solutions are close
    is_close = np.allclose(x_custom, x_numpy, rtol=1e-10, atol=1e-10)
    
    print(f"Test passed: {is_close}")
    if not is_close:
        print(f"Custom solution: {x_custom}")
        print(f"NumPy solution: {x_numpy}")
        print(f"Difference: {x_custom - x_numpy}")
    
    return is_close

# Run the test
if __name__ == "__main__":
    test_custom_thomas_solver()
    
    # Test with complex numbers
    print("\nTesting with complex numbers:")
    n = 5
    dp = 2.0 + 1.0j
    dp1 = 2.5 + 0.5j
    dp2 = 2.5 - 0.5j
    do = -1.0 + 0.2j
    
    # Create the full matrix for verification
    A = np.zeros((n, n), dtype=complex)
    
    # Fill the main diagonal
    A[0, 0] = dp1
    for i in range(1, n-1):
        A[i, i] = dp
    A[n-1, n-1] = dp2
    
    # Fill the off-diagonals
    for i in range(n-1):
        A[i, i+1] = do  # Upper diagonal
        A[i+1, i] = do  # Lower diagonal
    
    # Create a right-hand side vector
    b = np.ones(n, dtype=complex)
    
    # Solve using custom_thomas_solver
    x_custom = custom_thomas_solver(dp, dp1, dp2, do, b)
    
    # Solve using numpy's linalg.solve for comparison
    x_numpy = np.linalg.solve(A, b)
    
    # Check if the solutions are close
    is_close = np.allclose(x_custom, x_numpy, rtol=1e-10, atol=1e-10)
    
    print(f"Test passed: {is_close}")
    if not is_close:
        print(f"Custom solution: {x_custom}")
        print(f"NumPy solution: {x_numpy}")
        print(f"Difference: {x_custom - x_numpy}")