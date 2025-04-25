
import numpy as np


"""
        THESE 3 METHODS ARE USED ONCE AT THE BEGINING OF THE ROUTINE
"""

def Sellmeir_Fcy_Response(c, f):
    """
    Sellmeir equation for calculating the refractive index (n_omega)
    for fused silica given the speed of light (c) and frequency (f).

    Parameters:
    c (float): Speed of light in m/s
    f (float): Frequency in Hz

    Returns:
    n_omega (float): Refractive index at the given frequency
    """

    # Omega = 2 * np.pi * f
    Lambda = c / f
    Lambda_in_Micras = Lambda * 1e6

    # Sellmeir Equation coefficients for fused silica
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794

    C1 = 0.0684043
    C2 = 0.1162414
    C3 = 9.896161

    nsq = 1 + ((B1 * Lambda_in_Micras ** 2) / (Lambda_in_Micras ** 2 - C1 ** 2)) + \
          ((B2 * Lambda_in_Micras ** 2) / (Lambda_in_Micras ** 2 - C2 ** 2)) + \
          ((B3 * Lambda_in_Micras ** 2) / (Lambda_in_Micras ** 2 - C3 ** 2))

    n_omega = np.sqrt(nsq)

    return n_omega

def Gaussian_BEAM_Solution_Saleh(Eo,wo,ko,XX,YY,Z):
    zo = ko * (wo ** 2) / 2
    w_z = wo * np.sqrt(1 + (Z / zo) ** 2)

    # To avoid division by zero, replace Z=0 with a small epsilon
    if Z == 0:
        Z = np.finfo(float).eps

    R_z = Z * (1 + (zo / Z) ** 2)
    psi_z = np.arctan(Z / zo)

    Eout = Eo * (wo / w_z) * np.exp(-((XX ** 2 + YY ** 2) / w_z ** 2)
                                    - 1j * ko * Z
                                    - 1j * (ko * (XX ** 2 + YY ** 2) / (2 * R_z))
                                    + 1j * psi_z)

    return Eout, w_z

def Gaussian_BEAM_Solution_Saleh1D(Eo,wo,ko,R,Z):
    zo = ko * (wo ** 2) / 2
    w_z = wo * np.sqrt(1 + (Z / zo) ** 2)

    # To avoid division by zero in R_z calculation, ensure Z is not zero
    if Z == 0:
        Z = np.finfo(float).eps

    R_z = Z * (1 + (zo / Z) ** 2)
    psi_z = np.arctan(Z / zo)

    Eout = Eo * (wo / w_z) * np.exp(-(R ** 2) / (w_z ** 2)
                                    - 1j * ko * Z
                                    - 1j * (ko * (R ** 2) / (2 * R_z))
                                    + 1j * psi_z)

    return Eout, w_z


"""
        FOR ACCELERATION >>>>>>>>>
"""

def BPM_First_half_TBC(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2):
    """
    Split-step half-stage with transparent boundary conditions (TBC).
    Applies diffraction and Kerr nonlinearity over a half-step of size DZ in z.
    Parameters:
        PHI_m       : 2D complex field at current z-plane, shape (NDX+1, NDY+1)
        PHI_m_auxNL : 2D complex field used for nonlinear term, same shape
        k           : propagation constant
        n0          : linear refractive index at this step
        NDX, NDY    : number of grid points in x and y
        DX, DY      : spatial grid spacing in x and y
        DZ          : propagation step size in z
        n2          : 2D Kerr nonlinear index array, same shape as PHI_m
    Returns:
        PHI_aux     : 2D complex field after half-step propagation
    """
    # Precompute coefficients
    A = 1
    B = -1j / (2 * k * n0)
    C = -1j / (2 * k * n0)
    Delta_nNL = n2 * np.abs(PHI_m_auxNL)**2
    D_NL = -1j * k * Delta_nNL

    # Initialize output field
    PHI_aux = np.zeros((NDX + 1, NDY + 1), dtype=complex)

    # Loop over each row (y-index)
    for l in range(NDY + 1):
        # Tridiagonal coefficients in x-direction
        alfa = -B / (2 * DX**2)
        beta = (A / DZ) + (B / DX**2) - (D_NL[:, l] / 4)
        gamma = alfa

        # Build tridiagonal matrix MMx
        diag_main  = beta
        diag_lower = alfa * np.ones(NDX)
        diag_upper = gamma * np.ones(NDX)
        MMx = (
            np.diag(diag_main) +
            np.diag(diag_lower, -1) +
            np.diag(diag_upper,  1)
        )

        # Transparent BC at X boundaries
        # Left boundary
        if PHI_m[0, l] != 0 and PHI_m[1, l] != 0:
            Gamma_L = 1 / (PHI_m[1, l] / PHI_m[0, l])
            Gamma_L = np.nan_to_num(Gamma_L)
            Gamma_L = np.real(Gamma_L) + 1j * abs(np.imag(Gamma_L))
        else:
            Gamma_L = 0
        MMx[0, 0] += alfa * Gamma_L

        # Right boundary
        if PHI_m[NDX, l] != 0 and PHI_m[NDX - 1, l] != 0:
            Gamma_R = 1 / (PHI_m[NDX - 1, l] / PHI_m[NDX, l])
            Gamma_R = np.nan_to_num(Gamma_R)
            Gamma_R = np.real(Gamma_R) - 1j * abs(np.imag(Gamma_R))
        else:
            Gamma_R = 0
        MMx[NDX, NDX] += alfa * Gamma_R

        # Build RHS vector d for y-boundaries or interior
        if l == 0:
            # bottom edge
            Gamma_Y = 1 / (PHI_m[:, 1] / PHI_m[:, 0])
            Gamma_Y = np.nan_to_num(Gamma_Y)
            Gamma_Y = np.real(Gamma_Y) - 1j * abs(np.imag(Gamma_Y))
            PHI_bound = Gamma_Y * PHI_m[:, 0]
            d = (
                (C / (2 * DY**2)) * PHI_bound +
                ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] +
                (C / (2 * DY**2)) * PHI_m[:, l + 1]
            )
        elif l == NDY:
            # top edge
            Gamma_Y = 1 / (PHI_m[:, NDY - 1] / PHI_m[:, NDY])
            Gamma_Y = np.nan_to_num(Gamma_Y)
            PHI_bound = Gamma_Y * PHI_m[:, NDY]
            d = (
                (C / (2 * DY**2)) * PHI_m[:, l - 1] +
                ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] +
                (C / (2 * DY**2)) * PHI_bound
            )
        else:
            # interior rows
            d = (
                (C / (2 * DY**2)) * PHI_m[:, l - 1] +
                ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] +
                (C / (2 * DY**2)) * PHI_m[:, l + 1]
            )

        # Solve tridiagonal system for this row
        PHI_aux[:, l] = np.linalg.solve(MMx, d)
    return PHI_aux

def BPM_Second_half_TBC(PHI_pm, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2):
    """
    Split-step second half-stage with transparent boundary conditions (TBC).
    Applies Kerr nonlinearity then diffraction over a half-step of size DZ in z.

    Parameters:
        PHI_pm      : 2D complex field at mid z-plane, shape (NDX+1, NDY+1)
        PHI_m_auxNL : 2D complex field used for nonlinear term, same shape
        k           : propagation constant
        n0          : linear refractive index at this step
        NDX, NDY    : number of grid points in x and y
        DX, DY      : spatial grid spacing in x and y
        DZ          : propagation step size in z
        n2          : 2D Kerr nonlinear index array, same shape as PHI_pm

    Returns:
        PHI_aux     : 2D complex field after second half-step propagation
    """
    # Precompute coefficients
    A = 1
    B = -1j / (2 * k * n0)
    C = -1j / (2 * k * n0)
    Delta_nNL = n2 * np.abs(PHI_m_auxNL)**2
    D_NL = -1j * k * Delta_nNL

    # Initialize output field
    PHI_aux = np.zeros((NDX + 1, NDY + 1), dtype=complex)

    # Loop over each column (x-index)
    for i in range(NDX + 1):
        # Tridiagonal coefficients in y-direction
        alfa = -C / (2 * DY**2)
        beta = (A / DZ) + (C / DY**2) - (D_NL[i, :] / 4)
        gamma = alfa

        # Build tridiagonal matrix MMy
        diag_main  = beta
        diag_lower = alfa * np.ones(NDY)
        diag_upper = gamma * np.ones(NDY)
        MMy = (
            np.diag(diag_main) +
            np.diag(diag_lower, -1) +
            np.diag(diag_upper,  1)
        )

        # Transparent BC at Y boundaries
        # Bottom boundary
        if PHI_pm[i, 0] != 0 and PHI_pm[i, 1] != 0:
            Gamma_B = 1 / (PHI_pm[i, 1] / PHI_pm[i, 0])
            Gamma_B = np.nan_to_num(Gamma_B)
            Gamma_B = np.real(Gamma_B) + 1j * abs(np.imag(Gamma_B))
        else:
            Gamma_B = 0
        MMy[0, 0] += alfa * Gamma_B

        # Top boundary
        if PHI_pm[i, NDY] != 0 and PHI_pm[i, NDY - 1] != 0:
            Gamma_T = 1 / (PHI_pm[i, NDY - 1] / PHI_pm[i, NDY])
            Gamma_T = np.nan_to_num(Gamma_T)
            Gamma_T = np.real(Gamma_T) - 1j * abs(np.imag(Gamma_T))
        else:
            Gamma_T = 0
        MMy[NDY, NDY] += alfa * Gamma_T

        # Build RHS vector r for interior or boundaries
        if i == 0:
            # left edge
            Gamma_X = 1 / (PHI_pm[1, :] / PHI_pm[0, :])
            Gamma_X = np.nan_to_num(Gamma_X)
            Gamma_X = np.real(Gamma_X) - 1j * abs(np.imag(Gamma_X))
            PHI_edge = PHI_pm[0, :] * Gamma_X
            r = (
                (B / (2 * DX**2)) * PHI_pm[i + 1, :] +
                ((A / DZ) - (B / DX**2) + (D_NL[i, :] / 4)) * PHI_pm[i, :] +
                (B / (2 * DX**2)) * PHI_edge
            )
        elif i == NDX:
            # right edge
            Gamma_X = 1 / (PHI_pm[NDX - 1, :] / PHI_pm[NDX, :])
            Gamma_X = np.nan_to_num(Gamma_X)
            PHI_edge = PHI_pm[NDX, :] * Gamma_X
            r = (
                (B / (2 * DX**2)) * PHI_edge +
                ((A / DZ) - (B / DX**2) + (D_NL[i, :] / 4)) * PHI_pm[i, :] +
                (B / (2 * DX**2)) * PHI_pm[i - 1, :]
            )
        else:
            # interior columns
            r = (
                (B / (2 * DX**2)) * PHI_pm[i + 1, :] +
                ((A / DZ) - (B / DX**2) + (D_NL[i, :] / 4)) * PHI_pm[i, :] +
                (B / (2 * DX**2)) * PHI_pm[i - 1, :]
            )

        # Solve tridiagonal system for this column
        PHI_aux[i, :] = np.linalg.solve(MMy, r)

    return PHI_aux


def BPM_2D_Prop_NL_var_alongZ(PHI_m, k, NDX, NDY, NDZ, DX, DY, DZ, n_medium, n_sample, n2_sample, Zi_sample, Stu):
    """
    In Functions module, Npoints_Z_to_save is only for saving the 2d beam profile, in the main
    script was set to 15, that is to take 15 snapshots of the beam profile and to return it.
    but since i'm only interested in the transmittance by now, then i dont need that and removed
    all code related to the snapshots. This function is only return the optical field at the end of the propagation.
    I want to simplify the code as much as i can to facilitate is translation
    to C/C++ HLS.
    """

    n0 = n0_z(0, Zi_sample, Stu, n_medium, n_sample)  # first order n in the z index specified by Zis
    n2 = n2_z(0, Zi_sample, Stu, n2_sample)  # second order n in the z index specified by Zis (assuming n2 of medium is 0)

    PHI_m_auxNL_DZ_2 = PHI_m  # First NL term is calculated from the initial field
    PHI_m_half = PHI_m  # First term is calculated from the initial field

    # Propagating a first DZ/2
    PHI_pm_DZ_2 = BPM_First_half_TBC(PHI_m_half, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, 0.5 * DZ, n2)
    PHI_m_half = BPM_Second_half_TBC(PHI_pm_DZ_2, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, 0.5 * DZ, n2)
    # Phase Delay for the propagation in the step DZ/2
    PHI_m_half = PHI_m_half * np.exp(-1j * k * n0 * DZ)
    # Using this half as the initial
    PHI_m_auxNL = PHI_m_half  # First NL term is calculated from the initial field calculated at DZ/2


    for z_step in range(1, NDZ):
        n0 = n0_z(z_step, Zi_sample, Stu, n_medium, n_sample)  # first order n in the z index specified by Zis
        n2 = n2_z(z_step, Zi_sample, Stu, n2_sample)  # second order n in the z index specified by Zis (assuming n2 of medium is 0)

        # Propagating a whole DZ
        PHI_pm = BPM_First_half_TBC(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2)
        PHI_m = BPM_Second_half_TBC(PHI_pm, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2)
        # Phase Delay for the propagation in the step DZ
        PHI_m = PHI_m * np.exp(-1j * k * n0 * DZ)
        PHI_m_auxNL_DZ_2 = PHI_m  # Non-linear term is calculated from the field a half step behind

        # Propagating a whole DZ but staircased
        PHI_pm_DZ_2 = BPM_First_half_TBC(PHI_m_half, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, DZ, n2)
        PHI_m_half = BPM_Second_half_TBC(PHI_pm_DZ_2, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, DZ, n2)
        # Phase Delay for the propagation in the step DZ
        PHI_m_half = PHI_m_half * np.exp(-1j * k * n0 * DZ)
        PHI_m_auxNL = PHI_m_half  # Non-linear term is calculated from the field a half step behind

    return PHI_m


def n2_z(current_z_index, sample_location, sample_thickness_units, n2_value):
    if current_z_index < sample_location or current_z_index > sample_location + sample_thickness_units-1:
        return 0
    else:
        return n2_value

def n0_z(current_z_index, sample_location, sample_thickness_units, n_medium, n_sample):
    if current_z_index < sample_location or current_z_index > sample_location + sample_thickness_units-1:
        return n_medium
    else:
        return n_sample