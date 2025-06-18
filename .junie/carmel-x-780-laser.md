# Gaussian Beam Parameters — Carmel X-780

## 🧾 Context: Laser and Manufacturer

The **Carmel X-780** is a femtosecond fiber laser developed by **Calmar Laser**, a leading company based in California specializing in ultrafast laser systems. This particular model offers a **780 nm central wavelength**, **<90 fs pulse durations**, and **exceptional beam quality (M² < 1.1)**. Designed for high-precision applications such as **multiphoton microscopy, optical metrology, and 3D microprinting**, it provides a collimated, nearly diffraction-limited output ideal for Gaussian beam modeling in computational photonics and nonlinear optics.

---

## 📊 Parameters for Gaussian Beam Modeling

| Parameter                     | Symbol          | Value                          |
|------------------------------|------------------|--------------------------------|
| Wavelength                   | λ                | 780e-9 m (780 nm)              |
| Beam Waist (estimated)       | w₀               | 0.625e-3 m (0.625 mm)\*        |
| Beam Diameter (exit)         | D                | 1.25 mm                        |
| Pulse Width (FWHM)           | Δt               | < 90 fs                        |
| Temporal Profile             | —                | sech² (autocorrelation factor) |
| Beam Quality                 | M²               | < 1.1                          |
| Beam Roundness               | —                | > 90%                          |
| Polarization Extinction      | —                | > 20 dB                        |
| Repetition Rate              | f_rep            | 80 MHz                         |
| Pulse Energy (typical)       | E_pulse          | ~12.5 nJ                       |
| Average Power                | P_avg            | ~1.0 W                         |

> \*As the beam diameter at exit is 1.25 mm, the beam waist is approximated as half of it assuming minimal divergence.

---

## 📐 Derived Quantities

- **Rayleigh Range**:  
  \[
  z_R = \frac{\pi w_0^2}{\lambda}
  \]

- **Beam Radius at z**:  
  \[
  w(z) = w_0 \cdot \sqrt{1 + \left(\frac{z}{z_R}\right)^2}
  \]

- **Wavefront Curvature**:  
  \[
  R(z) = z \cdot \left(1 + \left(\frac{z_R}{z}\right)^2\right)
  \]

- **Gouy Phase Shift**:  
  \[
  \psi(z) = \arctan\left(\frac{z}{z_R}\right)
  \]

---

## ✅ Recommended Use

These parameters are ideal for implementing in:

- Python (`numpy`)
- MATLAB
- Beam Propagation Method (BPM)
- Split-Step Fourier simulations
- Modeling for nonlinear microscopy and waveguide coupling

