import numpy as np
from operator_solvers import compute_b_vector

def test_compute_b_vector():
    """
    Test the compute_b_vector function with a simple example.
    """
    # Create a test case
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
    
    # Create a test vector
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Compute b using compute_b_vector
    b_custom = compute_b_vector(dp, dp1, dp2, do, x0)
    
    # Compute b using direct matrix multiplication
    b_direct = A @ x0
    
    # Check if the results are close
    is_close = np.allclose(b_custom, b_direct, rtol=1e-10, atol=1e-10)
    
    print(f"Test passed: {is_close}")
    if not is_close:
        print(f"Custom result: {b_custom}")
        print(f"Direct result: {b_direct}")
        print(f"Difference: {b_custom - b_direct}")
    
    return is_close

def test_compute_b_vector_complex():
    """
    Test the compute_b_vector function with complex numbers.
    """
    # Create a test case with complex numbers
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
    
    # Create a test vector with complex values
    x0 = np.array([1.0+0.5j, 2.0-1.0j, 3.0+0.0j, 4.0-0.5j, 5.0+1.0j])
    
    # Compute b using compute_b_vector
    b_custom = compute_b_vector(dp, dp1, dp2, do, x0)
    
    # Compute b using direct matrix multiplication
    b_direct = A @ x0
    
    # Check if the results are close
    is_close = np.allclose(b_custom, b_direct, rtol=1e-10, atol=1e-10)
    
    print(f"Complex test passed: {is_close}")
    if not is_close:
        print(f"Custom result: {b_custom}")
        print(f"Direct result: {b_direct}")
        print(f"Difference: {b_custom - b_direct}")
    
    return is_close

def test_edge_cases():
    """
    Test edge cases for the compute_b_vector function.
    """
    # Test with a single element vector
    dp = 2.0
    dp1 = 2.5
    dp2 = 2.5
    do = -1.0
    x0 = np.array([3.0])
    
    # For a single element, only dp1 (which is also dp2) matters
    b_custom = compute_b_vector(dp, dp1, dp2, do, x0)
    b_expected = np.array([dp1 * x0[0]])
    
    is_close = np.allclose(b_custom, b_expected, rtol=1e-10, atol=1e-10)
    print(f"Single element test passed: {is_close}")
    
    # Test with a two-element vector
    x0 = np.array([2.0, 3.0])
    A = np.array([[dp1, do], [do, dp2]])
    
    b_custom = compute_b_vector(dp, dp1, dp2, do, x0)
    b_direct = A @ x0
    
    is_close = np.allclose(b_custom, b_direct, rtol=1e-10, atol=1e-10)
    print(f"Two-element test passed: {is_close}")
    
    return is_close

# Run all tests
if __name__ == "__main__":
    print("Testing compute_b_vector function:")
    test_compute_b_vector()
    test_compute_b_vector_complex()
    test_edge_cases()