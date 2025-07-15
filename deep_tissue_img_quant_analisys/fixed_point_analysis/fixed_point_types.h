// Auto-generated fixed-point type definitions for deep tissue imaging
#include <ap_fixed.h>
#include <complex>

// ADI_X - SNR: 165.02 dB
typedef ap_fixed<32, 16> adi_x_t;
// ADI_Y - SNR: 164.96 dB
typedef ap_fixed<32, 16> adi_y_t;
// Nonlinear - SNR: 178.11 dB
typedef ap_fixed<32, 16> nonlinear_t;
// LinearAbsorption - SNR: 178.11 dB
typedef ap_fixed<32, 16> linearabsorption_t;
// TwoPhotonAbsorption - SNR: 178.11 dB
typedef ap_fixed<32, 16> twophotonabsorption_t;

// Complex number types
typedef std::complex<adi_x_t> adi_x_complex_t;
typedef std::complex<adi_y_t> adi_y_complex_t;
typedef std::complex<nonlinear_t> nonlinear_complex_t;
typedef std::complex<linearabsorption_t> linearabsorption_complex_t;
typedef std::complex<twophotonabsorption_t> twophotonabsorption_complex_t;
