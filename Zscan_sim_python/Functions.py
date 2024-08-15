
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

def BPM_2D_LinearCrkNcolsn_for_freqProp_NL_varAlongZ(PHI_m,k,NDX,NDY,NDZ,DX,DY,DZ,Npoints_Z_to_save, nalongZ, n2alongZ):
    pass