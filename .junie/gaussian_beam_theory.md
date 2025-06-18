# Gaussian Beam — Lecture Guide for Function Generation

## 📘 Conceptual Overview

A **Gaussian beam** is a fundamental solution of the paraxial wave equation. It is the most common model for a laser beam and is characterized by a bell-shaped transverse profile that maintains its Gaussian form during propagation.

In computational photonics, the Gaussian beam is used extensively in simulations for waveguides, laser optics, and BPM (Beam Propagation Method).

---

## 🧮 Mathematical Definition

A **1D Gaussian beam** traveling along the **z-axis** is described by:

\[
E(x, z) = E_0 \cdot \frac{w_0}{w(z)} \cdot \exp\left( -\frac{x^2}{w(z)^2} \right) \cdot \exp\left( -i \left[kz + \frac{k x^2}{2R(z)} - \psi(z)\right] \right)
\]

Where:
- \( E_0 \): initial amplitude
- \( w(z) \): beam radius at distance \( z \)
- \( w_0 \): beam waist (minimum width at focus)
- \( R(z) \): radius of curvature of the wavefronts
- \( \psi(z) \): Gouy phase shift
- \( k = \frac{2\pi}{\lambda} \): wave number

---

## 📐 Beam Parameters

- **Rayleigh Range**: \( z_R = \frac{\pi w_0^2}{\lambda} \)
- **Beam Radius**: \( w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_R}\right)^2} \)
- **Wavefront Radius**: \( R(z) = z \left[1 + \left(\frac{z_R}{z}\right)^2 \right] \)
- **Gouy Phase**: \( \psi(z) = \tan^{-1}\left(\frac{z}{z_R}\right) \)

---

## 💻 Suggested Function Signature (Python-style)

```python
def gaussian_beam(x: np.ndarray, z: float, w0: float, lambda0: float, E0: float = 1.0) -> np.ndarray:
    """
    Computes the complex field of a Gaussian beam at position z.
    
    Parameters:
    - x (np.ndarray): transverse coordinate array
    - z (float): propagation distance (in meters)
    - w0 (float): beam waist (in meters)
    - lambda0 (float): wavelength (in meters)
    - E0 (float): amplitude (default is 1.0)
    
    Returns:
    - np.ndarray: complex field E(x, z)
    """
