
# Beam Propagation Method (BPM) — Lecture Summary

## 🧾 Context

The **Beam Propagation Method (BPM)** is a widely used numerical technique to simulate the **propagation of optical fields** in waveguides and free space under the **paraxial approximation**. It is particularly suited for **Gaussian beams**, **fiber optics**, and **integrated photonic structures**.

BPM solves the **paraxial form** of the scalar Helmholtz equation:

\[
\frac{\partial E}{\partial z} = \frac{i}{2k} \nabla_\perp^2 E - i \frac{k}{n_0} \Delta n(x, y, z) E
\]

Where:
- \( E \): electric field envelope
- \( z \): propagation axis
- \( \nabla_\perp^2 \): transverse Laplacian (e.g., ∂²/∂x² + ∂²/∂y²)
- \( \Delta n \): refractive index variation
- \( k = 2\pi / \lambda \): wave number

---

## 🧠 Physical Assumptions

- The field changes **slowly** along the z-direction (paraxial approximation)
- Time-harmonic dependence \( \exp(-i\omega t) \) is assumed
- Backward reflection and high-angle scattering are neglected

---

## 🧮 Split-Step Fourier Method (SSFM)

The **Split-Step Fourier Method** is a variant of BPM that separates the linear (diffraction) and nonlinear or inhomogeneous refractive index (potential) effects.

### 🔧 Key Idea

The propagator over a small step \( \Delta z \) is split into:

\[
E(x, z + \Delta z) \approx e^{\hat{D} \Delta z / 2} \cdot e^{\hat{N} \Delta z} \cdot e^{\hat{D} \Delta z / 2} E(x, z)
\]

Where:
- \( \hat{D} = \frac{i}{2k} \nabla_\perp^2 \) is the diffraction operator
- \( \hat{N} = -i \frac{k}{n_0} \Delta n(x, z) \) is the index potential operator

### ⚡ Implementation Steps

1. **Initialize** field \( E(x, 0) \) (e.g., Gaussian beam)
2. For each step \( \Delta z \):
   - Apply \( \exp(\hat{D} \Delta z / 2) \) in Fourier space:
     - \( \mathcal{F}^{-1} \left[ \exp\left(-i \frac{k_x^2}{2k} \Delta z / 2 \right) \cdot \mathcal{F}[E] \right] \)
   - Apply \( \exp(\hat{N} \Delta z) \) in real space:
     - \( E \leftarrow E \cdot \exp\left( -i \frac{k}{n_0} \Delta n(x, z) \Delta z \right) \)
   - Apply \( \exp(\hat{D} \Delta z / 2) \) again in Fourier space

3. Repeat until final \( z \)

---

## 💡 Advantages of SSFM

- Spectrally accurate diffraction
- Simple to implement using FFT
- Efficient for large-scale simulations in 1+1D and 2+1D

---

## 🔬 Application Scenarios

- Propagation of **Gaussian beams** in air or nonlinear media
- Modeling **self-focusing** and **soliton formation**
- Optical waveguides, tapers, or grating couplers
- Time-domain variants (e.g., pulse propagation in fibers)

---

## 📚 Reference

Based on the detailed treatment of BPM and Split-Step methods in:

> *Computational Photonics*  
> Marek S. Wartak, Cambridge University Press  
> ISBN: 9781107030433

