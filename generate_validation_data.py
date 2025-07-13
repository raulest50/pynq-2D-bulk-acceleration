import numpy as np
import os
import argparse

# Import the functions we need to test
from deep_tissue_imaging.propagators.step_operators import (
    adi_x, adi_y, half_nonlinear, half_linear_absorption, half_2photon_absorption
)

def save_complex_matrix(filename, matrix):
    """Save a complex matrix to a .dat file with real and imaginary parts on each line."""
    with open(filename, 'w') as f:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                f.write(f"{matrix[i, j].real:.8e} {matrix[i, j].imag:.8e}\n")
    print(f"Saved matrix to {filename}")

def load_complex_matrix(filename, shape):
    """Load a complex matrix from a .dat file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                real, imag = float(parts[0]), float(parts[1])
                data.append(complex(real, imag))
    
    # Reshape the data into the original matrix shape
    return np.array(data, dtype=np.complex64).reshape(shape)

def main(output_dir="validationData", seed=42):
    """
    Generate validation data for testing C++ HLS implementations of BPM functions.
    
    Args:
        output_dir (str): Directory to save validation data
        seed (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters based on deep_tissue_imaging_1.py
    NDX, NDY = 46, 46
    DX, DY = np.float32(1.5e-5), np.float32(1.5e-5)
    DZ = np.float32(1e-4)
    k = np.float32(7853981.6339)
    n0 = np.float32(1.0)
    n2 = np.float32(2.5e-20)
    alpha = np.float32(0.1)  # Example value for linear absorption
    beta = np.float32(1e-12)  # Example value for 2-photon absorption
    eps = np.float32(1e-12)   # Small value for adi_x and adi_y
    
    # Generate random complex matrix
    np.random.seed(seed)  # For reproducibility
    real_part = np.random.rand(NDX+1, NDY+1).astype(np.float32)
    imag_part = np.random.rand(NDX+1, NDY+1).astype(np.float32)
    input_matrix = real_part + 1j * imag_part
    
    # Save input matrix
    input_file = os.path.join(output_dir, "in.dat")
    save_complex_matrix(input_file, input_matrix)
    
    # Process and save output for each function
    
    # 1. adi_x
    output_adi_x = adi_x(input_matrix, NDY+1, eps, k, DZ, DX)
    save_complex_matrix(os.path.join(output_dir, "adi_x_out.dat"), output_adi_x)
    
    # 2. adi_y
    output_adi_y = adi_y(input_matrix, NDX+1, eps, k, DZ, DY)
    save_complex_matrix(os.path.join(output_dir, "adi_y_out.dat"), output_adi_y)
    
    # 3. half_nonlinear
    output_nonlinear = half_nonlinear(input_matrix, k, n2, DZ)
    save_complex_matrix(os.path.join(output_dir, "half_nonlinear_out.dat"), output_nonlinear)
    
    # 4. half_linear_absorption
    output_linear_abs = half_linear_absorption(input_matrix, alpha, DZ)
    save_complex_matrix(os.path.join(output_dir, "half_linear_absorption_out.dat"), output_linear_abs)
    
    # 5. half_2photon_absorption
    output_2photon_abs = half_2photon_absorption(input_matrix, beta, DZ)
    save_complex_matrix(os.path.join(output_dir, "half_2photon_absorption_out.dat"), output_2photon_abs)
    
    print("\nValidation data generation complete!")
    print(f"Input file: {input_file}")
    print("Output files:")
    print(f"  - {os.path.join(output_dir, 'adi_x_out.dat')}")
    print(f"  - {os.path.join(output_dir, 'adi_y_out.dat')}")
    print(f"  - {os.path.join(output_dir, 'half_nonlinear_out.dat')}")
    print(f"  - {os.path.join(output_dir, 'half_linear_absorption_out.dat')}")
    print(f"  - {os.path.join(output_dir, 'half_2photon_absorption_out.dat')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation data for C++ HLS testing")
    parser.add_argument("--output-dir", default="validationData", help="Directory to save validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args.output_dir, args.seed)