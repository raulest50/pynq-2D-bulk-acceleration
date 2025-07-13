import numpy as np
import os
import argparse

# Import necessary modules and functions
from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain

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

def main(output_dir="validation_data_main", seed=42):
    """
    Generate validation data for testing C++ HLS implementations of full_step_within_tissue.
    
    Args:
        output_dir (str): Directory to save validation data
        seed (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Parameters based on deep_tissue_imaging_1.py
    Lz = np.float32(361e-6)  # 361um
    Nz = 361
    dz = np.float32(Lz / Nz)  # 1um
    
    Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
    Nx, Ny = 256, 256
    dx = np.float32(Lx / Nx)  # 0.176um
    dy = np.float32(Ly / Ny)  # 0.176um
    
    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    k0 = np.float32(2*np.pi / laser.wavelength)
    k = np.float32(k0 * tejido.n_0)
    sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
    sigma_x = np.float32(5e-6)  # Typical value for brain tissue (5 μm)
    
    # Create domain object
    domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)
    
    # Generate initial field
    phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)
    
    # Save initial field
    save_complex_matrix(os.path.join(output_dir, "initial_field.dat"), phi0)
    
    # Test single step of full_step_within_tissue
    phi_single_step = prop.full_step_within_tissue(phi0.copy(), tejido, domain)
    save_complex_matrix(os.path.join(output_dir, "full_step_within_tissue_out.dat"), phi_single_step)
    
    # Test 120 steps of full_step_within_tissue (without applying random phase masks)
    phi = phi0.copy()
    for i in range(120):
        phi = prop.full_step_within_tissue(phi, tejido, domain)
        
        # Save intermediate results at specific steps if needed
        if i in [0, 39, 79, 119]:
            save_complex_matrix(os.path.join(output_dir, f"full_step_within_tissue_step_{i+1}.dat"), phi)
    
    # Save final result after 120 steps
    save_complex_matrix(os.path.join(output_dir, "full_step_within_tissue_120_steps.dat"), phi)
    
    print("\nValidation data generation complete!")
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print(f"  - {os.path.join(output_dir, 'initial_field.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_out.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_step_1.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_step_40.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_step_80.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_step_120.dat')}")
    print(f"  - {os.path.join(output_dir, 'full_step_within_tissue_120_steps.dat')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation data for C++ HLS testing of full_step_within_tissue")
    parser.add_argument("--output-dir", default="validation_data_main", help="Directory to save validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args.output_dir, args.seed)