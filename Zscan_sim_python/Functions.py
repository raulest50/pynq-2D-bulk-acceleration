
import numpy as np

def Create_Volumeetric_Data(PHI_m_alongZ,XX,YY,z_to_save,Npoints_Z_to_save, normalized):
    # Initialize the volumetric data arrays
    Nzs = PHI_m_alongZ.shape[2]
    VolData = np.zeros((XX.shape[0], XX.shape[1], Nzs))
    zLL = np.zeros((XX.shape[0], XX.shape[1], Nzs))

    c_planes = 0

    for i in range(Nzs):
        # Coordinate along z, of the corresponding plane
        cz = z_to_save[i]
        zLL[:, :, c_planes] = cz + np.zeros_like(XX)

        max_c = np.max(np.abs(PHI_m_alongZ[:, :, i]))

        # Adding a new plane
        if normalized:
            VolData[:, :, c_planes] = np.abs(PHI_m_alongZ[:, :, i]) / max_c
        else:
            VolData[:, :, c_planes] = np.abs(PHI_m_alongZ[:, :, i])

        c_planes += 1

    return VolData, XX, YY, zLL

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


def BPM_First_half(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2):
    BC = 'TBC'

    #===============================================
    ##=== Evaluating from z=m to z=m+1/2     === ###
    #===============================================
    A = 1
    B = -1j / (2 * k * n0)
    C = -1j / (2 * k * n0)
    Delta_nNL = n2 * np.abs(PHI_m_auxNL)**2
    D_NL = -1j * k * Delta_nNL

    # Initialize the output array
    PHI_aux = np.zeros((NDX + 1, NDY + 1), dtype=complex)

    # Loop over Y
    for l in range(NDY + 1):
        # Assemble tridiagonal matrix for X direction
        alfa_diagx = -B / (2 * (DX**2))
        beta_diagx = (A / DZ) + (B / DX**2) - D_NL[:, l] / 4
        gamma_diagx = -B / (2 * (DX**2))

        MM_bx = np.diag(beta_diagx)
        MM_ax = np.diag(alfa_diagx * np.ones(NDX), -1)
        MM_cx = np.diag(gamma_diagx * np.ones(NDX), 1)

        MMx = MM_ax + MM_bx + MM_cx

        # Correct boundary conditions using field values from the previous step
        if BC == 'TBC':
            if PHI_m[0, l] == 0 or PHI_m[1, l] == 0:
                GammaLX0 = 0
            else:
                GammaLX0 = 1 / (PHI_m[1, l] / PHI_m[0, l])
                GammaLX0 = np.nan_to_num(GammaLX0)
                GammaLX0 = np.real(GammaLX0) + 1j * np.abs(np.imag(GammaLX0))

            if PHI_m[NDX, l] == 0 or PHI_m[NDX - 1, l] == 0:
                GammaLNDX = 0
            else:
                GammaLNDX = 1 / (PHI_m[NDX - 1, l] / PHI_m[NDX, l])
                GammaLNDX = np.nan_to_num(GammaLNDX)
                GammaLNDX = np.real(GammaLNDX) - 1j * np.abs(np.imag(GammaLNDX))

        elif BC == 'PEC':
            GammaLX0 = 0
            GammaLNDX = 0

        MMx[0, 0] += alfa_diagx * GammaLX0
        MMx[NDX, NDX] += alfa_diagx * GammaLNDX

        # Calculate vector d, based on the previous field
        if l == 0:
            if BC == 'TBC':
                GammaLY0 = 1. / (PHI_m[:, 1] / PHI_m[:, 0])
                GammaLY0 = np.nan_to_num(GammaLY0)
                GammaLY0 = np.real(GammaLY0) - 1j * np.abs(np.imag(GammaLY0))

            elif BC == 'PEC':
                GammaLY0 = 0

            PHIY0 = GammaLY0 * PHI_m[:, 0]
            d = (C / (2 * (DY**2))) * PHIY0 + ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] + (C / (2 * (DY**2))) * PHI_m[:, l + 1]

        elif l == NDY:
            if BC == 'TBC':
                GammaLYN = 1. / (PHI_m[:, NDY - 1] / PHI_m[:, NDY])
                GammaLYN = np.nan_to_num(GammaLYN)

            elif BC == 'PEC':
                GammaLYN = 0

            PHIYN = GammaLYN * PHI_m[:, NDY]
            d = (C / (2 * (DY**2))) * PHI_m[:, l - 1] + ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] + (C / (2 * (DY**2))) * PHIYN

        else:
            d = (C / (2 * (DY**2))) * PHI_m[:, l - 1] + ((A / DZ) - (C / DY**2) + (D_NL[:, l] / 4)) * PHI_m[:, l] + (C / (2 * (DY**2))) * PHI_m[:, l + 1]

        # Invert the matrix to find the new field
        PHI_aux[:, l] = np.linalg.solve(MMx, d)

    # Using this field for the next step
    PHI_pm = PHI_aux

    return PHI_pm



def BPM_Second_half(PHI_pm, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2):
    BC = 'TBC'

    #===============================================
    ##=== Evaluating from z=m+1/2 to z=m+1 === ###
    #===============================================
    A = 1
    B = -1j / (2 * k * n0)
    C = -1j / (2 * k * n0)
    Delta_nNL = n2 * np.abs(PHI_m_auxNL)**2
    D_NL = -1j * k * Delta_nNL

    # Initialize the output array
    PHI_aux2 = np.zeros((NDX + 1, NDY + 1), dtype=complex)

    # Loop over X (receives PHI_pm as input)
    for i in range(NDX + 1):
        # Assemble tridiagonal matrix for Y direction
        alfa_diagy = -C / (2 * (DY**2))
        beta_diagy = (A / DZ) + C / (DY**2) - D_NL[i, :] / 4
        gamma_diagy = -C / (2 * (DY**2))

        MM_by = np.diag(beta_diagy)
        MM_ay = np.diag(alfa_diagy * np.ones(NDY), -1)
        MM_cy = np.diag(gamma_diagy * np.ones(NDY), 1)

        MMy = MM_ay + MM_by + MM_cy

        # Correct boundary conditions using field values from the previous step
        if BC == 'TBC':
            if PHI_pm[i, 0] == 0:
                GammaLY0 = 0
            else:
                GammaLY0 = 1. / (PHI_pm[i, 1] / PHI_pm[i, 0])
                GammaLY0 = np.nan_to_num(GammaLY0)
                GammaLY0 = np.real(GammaLY0) + 1j * np.abs(np.imag(GammaLY0))

            if PHI_pm[i, NDY] == 0:
                GammaLYNDY = 0
            else:
                GammaLYNDY = 1 / (PHI_pm[i, NDY - 1] / PHI_pm[i, NDY])
                GammaLYNDY = np.nan_to_num(GammaLYNDY)
                GammaLYNDY = np.real(GammaLYNDY) - 1j * np.abs(np.imag(GammaLYNDY))

        elif BC == 'PEC':
            GammaLY0 = 0
            GammaLYNDY = 0

        MMy[0, 0] += alfa_diagy * GammaLY0
        MMy[NDY, NDY] += alfa_diagy * GammaLYNDY

        # Calculate vector R, based on the previous field
        if i == 0:
            if BC == 'TBC':
                GammaL0 = 1. / (PHI_pm[1, :] / PHI_pm[0, :])
                GammaL0 = np.nan_to_num(GammaL0)
                GammaL0 = np.real(GammaL0) - 1j * np.abs(np.imag(GammaL0))

            elif BC == 'PEC':
                GammaL0 = 0

            PHI_0 = PHI_pm[0, :] * GammaL0
            r = (B / (2 * (DX**2))) * PHI_pm[i + 1, :] + ((A / DZ) - (B / (DX**2)) + D_NL[i, :] / 4) * PHI_pm[i, :] + (B / (2 * (DX**2))) * PHI_0

        elif i == NDX:
            if BC == 'TBC':
                GammaLN = 1. / (PHI_pm[NDX - 1, :] / PHI_pm[NDX, :])
                GammaLN = np.nan_to_num(GammaLN)

            elif BC == 'PEC':
                GammaLN = 0

            PHI_NDR_p2 = PHI_pm[NDX, :] * GammaLN
            r = (B / (2 * (DX**2))) * PHI_NDR_p2 + ((A / DZ) - (B / (DX**2)) + D_NL[i, :] / 4) * PHI_pm[i, :] + (B / (2 * (DX**2))) * PHI_pm[i - 1, :]

        else:
            r = (B / (2 * (DX**2))) * PHI_pm[i + 1, :] + ((A / DZ) - (B / (DX**2)) + D_NL[i, :] / 4) * PHI_pm[i, :] + (B / (2 * (DX**2))) * PHI_pm[i - 1, :]

        # Transpose r (note: avoid conjugate transpose as in MATLAB's A')
        R = np.transpose(r)

        # Invert the matrix to find the new field and update
        PHI_aux2[i, :] = np.transpose(np.linalg.solve(MMy, R))

    # Update PHI_m for the next step
    PHI_m = PHI_aux2

    return PHI_m



def BPM_2D_LinearCrkNcolsn_for_freqProp_NL_varAlongZ(PHI_m,k,NDX,NDY,NDZ,DX,DY,DZ,Npoints_Z_to_save, nalongZ, n2alongZ):
    N_count_to_save = np.floor(NDZ / Npoints_Z_to_save)
    c_count_to_save = 0

    PHI_m_auxNL_DZ_2 = PHI_m  # First NL term is calculated from the initial field
    PHI_m_half = PHI_m  # First term is calculated from the initial field

    n0 = nalongZ[0]
    n2 = n2alongZ[0]

    # Propagating a first DZ/2
    PHI_pm_DZ_2 = BPM_First_half(PHI_m_half, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, 0.5 * DZ, n2)
    PHI_m_half = BPM_Second_half(PHI_pm_DZ_2, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, 0.5 * DZ, n2)

    # Using this half as the initial
    PHI_m_auxNL = PHI_m_half  # First NL term is calculated from the initial field calculated at DZ/2

    PHI_m_to_save = []
    z_to_save = []

    for z_step in range(NDZ):

        n0 = nalongZ[z_step]
        n2 = n2alongZ[z_step]

        # Propagating a whole DZ
        PHI_pm = BPM_First_half(PHI_m, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2)
        PHI_m = BPM_Second_half(PHI_pm, PHI_m_auxNL, k, n0, NDX, NDY, DX, DY, DZ, n2)
        PHI_m_auxNL_DZ_2 = PHI_m  # Non-linear term is calculated from the field a half step behind

        # Propagating a whole DZ but staircased
        PHI_pm_DZ_2 = BPM_First_half(PHI_m_half, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, DZ, n2)
        PHI_m_half = BPM_Second_half(PHI_pm_DZ_2, PHI_m_auxNL_DZ_2, k, n0, NDX, NDY, DX, DY, DZ, n2)

        PHI_m_auxNL = PHI_m_half  # Non-linear term is calculated from the field a half step behind

        if Npoints_Z_to_save > 0:
            if (z_step % N_count_to_save == 0) or (z_step == 0) or (z_step == NDZ - 1):
                # Save desired steps
                PHI_m_to_save.append(PHI_m.copy())
                z_to_save.append(DZ * (z_step + 1))
                c_count_to_save += 1
        else:
            PHI_m_to_save = []
            z_to_save = []

    return PHI_m, np.array(PHI_m_to_save), np.array(z_to_save)