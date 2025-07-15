import numpy as np
import os
import argparse

from deep_tissue_imaging.elementos.lasers import fuente_microscopia_1 as laser, campo_tem00
from deep_tissue_imaging.elementos.tejidos import cerebro_emb_pez_cebra as tejido
import deep_tissue_imaging.propagators.propagation as prop
import deep_tissue_imaging.elementos.domain as Domain
from deep_tissue_imaging.propagators.step_operators import (
    adi_x, adi_y, half_nonlinear, half_linear_absorption, half_2photon_absorption,
    custom_thomas_solver, compute_b_vector
)


def save_complex_matrix(filename, matrix):
    """Save a complex matrix to a .dat file with real and imaginary parts."""
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
    return np.array(data, dtype=np.complex64).reshape(shape)


def generate_individual_operators_data(output_dir, seed=42):
    """Generate validation data for individual operators."""
    print("\n=== Generating validation data for individual operators ===")

    np.random.seed(seed)

    # Parameters from deep_tissue_imaging_1.py
    Lx = np.float32(45e-6)
    Ly = np.float32(45e-6)
    Nx = 256
    Ny = 256
    dx = np.float32(Lx / Nx)
    dy = np.float32(Ly / Ny)
    dz = np.float32(1e-4)
    k = np.float32(7853981.6339)
    n2 = np.float32(2.5e-20)
    alpha = np.float32(0.1)
    beta = np.float32(1e-12)
    eps = np.float32(1e-12)

    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    input_matrix = campo_tem00(X, Y, laser.w0, laser.I_peak)

    save_complex_matrix(os.path.join(output_dir, "individual_ops_in.dat"), input_matrix)

    output_adi_x = adi_x(input_matrix, Ny, eps, k, dz, dx)
    save_complex_matrix(os.path.join(output_dir, "adi_x_out.dat"), output_adi_x)

    output_adi_y = adi_y(input_matrix, Nx, eps, k, dz, dy)
    save_complex_matrix(os.path.join(output_dir, "adi_y_out.dat"), output_adi_y)

    output_half_nl = input_matrix.copy()
    output_half_nl = half_2photon_absorption(output_half_nl, beta, dz)
    output_half_nl = half_nonlinear(output_half_nl, k, n2, dz)
    output_half_nl = half_linear_absorption(output_half_nl, alpha, dz)
    save_complex_matrix(os.path.join(output_dir, "half_nonlinear_ops_combined_out.dat"), output_half_nl)

    # Scalars for the tridiagonal coefficients
    dp = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(10.0, 100.0))
    dp1 = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(10.0, 100.0))
    dp2 = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(10.0, 100.0))
    do = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(10.0, 100.0))

    # Vector for the input
    sign = np.random.choice([-1.0, 1.0], size=Nx)
    x0 = sign * np.random.uniform(10.0, 100.0, Nx).astype(np.float32)

    np.savetxt(os.path.join(output_dir, "b_vector_dp_in.dat"), [[dp, 0.0]], fmt='%.8e')
    np.savetxt(os.path.join(output_dir, "b_vector_dp1_in.dat"), [[dp1, 0.0]], fmt='%.8e')
    np.savetxt(os.path.join(output_dir, "b_vector_dp2_in.dat"), [[dp2, 0.0]], fmt='%.8e')
    np.savetxt(os.path.join(output_dir, "b_vector_do_in.dat"), [[do, 0.0]], fmt='%.8e')
    np.savetxt(os.path.join(output_dir, "b_vector_x0_in.dat"), np.column_stack((x0, np.zeros_like(x0))), fmt='%.8e')

    b_vector = compute_b_vector(dp, dp1, dp2, do, x0)
    np.savetxt(os.path.join(output_dir, "b_vector_out.dat"), np.column_stack((b_vector.real, np.zeros_like(b_vector))), fmt='%.8e')

    thomas_result = custom_thomas_solver(dp, dp1, dp2, do, b_vector)
    np.savetxt(os.path.join(output_dir, "thomas_solver_out.dat"), np.column_stack((thomas_result.real, np.zeros_like(thomas_result))), fmt='%.8e')

    print("Individual operators validation data generation complete!")


def generate_full_step_data(output_dir, seed=42):
    """Generate validation data for full step propagation."""
    print("\n=== Generating validation data for full step propagation ===")
    np.random.seed(seed)

    Lz = np.float32(361e-6)
    Nz = 361
    dz = np.float32(Lz / Nz)
    Lx = np.float32(45e-6)
    Ly = np.float32(45e-6)
    Nx = 256
    Ny = 256
    dx = np.float32(Lx / Nx)
    dy = np.float32(Ly / Ny)

    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    k0 = np.float32(2*np.pi / laser.wavelength)
    k = np.float32(k0 * tejido.n_0)
    sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
    sigma_x = np.float32(5e-6)

    domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)
    phi0 = campo_tem00(X, Y, laser.w0, laser.I_peak)

    save_complex_matrix(os.path.join(output_dir, "initial_field.dat"), phi0)

    phi_single_step = prop.full_step_within_tissue(phi0.copy(), tejido, domain)
    save_complex_matrix(os.path.join(output_dir, "full_step_within_tissue_out.dat"), phi_single_step)

    phi = phi0.copy()
    for i in range(120):
        phi = prop.full_step_within_tissue(phi, tejido, domain)
        if i in [0, 39, 79, 119]:
            save_complex_matrix(os.path.join(output_dir, f"full_step_within_tissue_step_{i+1}.dat"), phi)

    save_complex_matrix(os.path.join(output_dir, "full_step_within_tissue_120_steps.dat"), phi)
    print("Full step validation data generation complete!")


def main(output_dir="validation_data_main", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    generate_individual_operators_data(output_dir, seed)
    generate_full_step_data(output_dir, seed)
    print("\n=== All validation data generation complete! ===")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation data for C++ HLS testing")
    parser.add_argument("--output-dir", default="validation_data_main", help="Directory to save validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args.output_dir, args.seed)
