import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional
import os
import pickle

class FixedPointAnalyzer:
    """
    A class for analyzing and recommending fixed-point parameters for the deep tissue imaging algorithm.
    This helps optimize FPGA resource usage while maintaining precision.
    """
    
    def __init__(self, save_dir: str = "./fixed_point_analysis"):
        """
        Initialize the analyzer with a directory to save results.
        
        Args:
            save_dir: Directory to save analysis results
        """
        self.save_dir = save_dir
        self.value_stats = {}
        self.recommendations = {}
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def analyze_range(self, variable: np.ndarray, name: str) -> Dict:
        """
        Analyze the range of values in a variable to determine appropriate bit-width.
        
        Args:
            variable: NumPy array to analyze
            name: Name of the variable for reporting
            
        Returns:
            Dictionary with statistics and bit-width recommendations
        """
        # Handle complex numbers
        if np.iscomplexobj(variable):
            real_part = np.real(variable)
            imag_part = np.imag(variable)
            real_stats = self._analyze_real_array(real_part, f"{name}_real")
            imag_stats = self._analyze_real_array(imag_part, f"{name}_imag")
            
            # Combine recommendations
            int_bits = max(real_stats['int_bits'], imag_stats['int_bits'])
            
            stats = {
                'real': real_stats,
                'imag': imag_stats,
                'int_bits': int_bits,
                'recommended_total_bits': int_bits + 16  # Default 16 fractional bits
            }
        else:
            stats = self._analyze_real_array(variable, name)
        
        # Store statistics
        self.value_stats[name] = stats
        
        return stats
    
    def _analyze_real_array(self, variable: np.ndarray, name: str) -> Dict:
        """
        Analyze a real-valued array.
        
        Args:
            variable: NumPy array to analyze
            name: Name of the variable
            
        Returns:
            Dictionary with statistics
        """
        min_val = np.min(variable)
        max_val = np.max(variable)
        abs_max = max(abs(min_val), abs(max_val))
        mean_val = np.mean(variable)
        std_val = np.std(variable)
        
        # Calculate required integer bits (including sign bit)
        if abs_max <= 0:
            int_bits = 1  # Just sign bit if value is 0
        else:
            int_bits = int(np.ceil(np.log2(abs_max))) + 1  # +1 for sign bit
        
        # Calculate histogram for distribution analysis
        hist, bin_edges = np.histogram(variable, bins=50)
        
        stats = {
            'min': min_val,
            'max': max_val,
            'abs_max': abs_max,
            'mean': mean_val,
            'std': std_val,
            'int_bits': int_bits,
            'histogram': {
                'counts': hist,
                'bin_edges': bin_edges
            }
        }
        
        print(f"{name}: min={min_val:.6e}, max={max_val:.6e}, mean={mean_val:.6e}, std={std_val:.6e}")
        print(f"Recommended integer bits: {int_bits}")
        
        return stats
    
    def simulate_fixed_point(self, value: np.ndarray, total_bits: int, int_bits: int) -> np.ndarray:
        """
        Simulate fixed-point quantization on a value.
        
        Args:
            value: Input array
            total_bits: Total bits in fixed-point representation
            int_bits: Integer bits in fixed-point representation
            
        Returns:
            Quantized array simulating fixed-point arithmetic
        """
        frac_bits = total_bits - int_bits
        scale = 2.0 ** frac_bits
        
        # Quantize by scaling, rounding, and scaling back
        quantized = np.round(value * scale) / scale
        
        # Simulate saturation
        max_val = 2.0 ** (int_bits - 1) - 2.0 ** (-frac_bits)
        min_val = -2.0 ** (int_bits - 1)
        
        return np.clip(quantized, min_val, max_val)
    
    def simulate_complex_fixed_point(self, value: np.ndarray, total_bits: int, int_bits: int) -> np.ndarray:
        """
        Simulate fixed-point quantization on a complex value.
        
        Args:
            value: Input complex array
            total_bits: Total bits in fixed-point representation
            int_bits: Integer bits in fixed-point representation
            
        Returns:
            Quantized complex array
        """
        real_part = np.real(value)
        imag_part = np.imag(value)
        
        quantized_real = self.simulate_fixed_point(real_part, total_bits, int_bits)
        quantized_imag = self.simulate_fixed_point(imag_part, total_bits, int_bits)
        
        return quantized_real + 1j * quantized_imag
    
    def analyze_operator(self, 
                         operator_func, 
                         input_data: np.ndarray, 
                         operator_name: str,
                         bit_width_range: List[int] = [16, 20, 24, 28, 32],
                         int_bits_range: Optional[List[int]] = None) -> Dict:
        """
        Analyze an operator function with different fixed-point configurations.
        
        Args:
            operator_func: Function that implements the operator
            input_data: Input data for the operator
            operator_name: Name of the operator
            bit_width_range: List of total bit widths to test
            int_bits_range: List of integer bit widths to test (if None, will be determined automatically)
            
        Returns:
            Dictionary with analysis results
        """
        # Get floating point reference result
        reference_output = operator_func(input_data.copy())
        
        # Analyze input and output ranges
        input_stats = self.analyze_range(input_data, f"{operator_name}_input")
        output_stats = self.analyze_range(reference_output, f"{operator_name}_output")
        
        # If int_bits_range not provided, generate based on analysis
        if int_bits_range is None:
            max_int_bits = max(input_stats['int_bits'], output_stats['int_bits'])
            int_bits_range = list(range(max(1, max_int_bits - 2), max_int_bits + 3))
        
        results = {
            'input_stats': input_stats,
            'output_stats': output_stats,
            'configurations': []
        }
        
        # Test different fixed-point configurations
        for total_bits in bit_width_range:
            for int_bits in int_bits_range:
                if int_bits >= total_bits:
                    continue  # Skip invalid configurations
                
                frac_bits = total_bits - int_bits
                
                # Apply fixed-point quantization to input
                if np.iscomplexobj(input_data):
                    quantized_input = self.simulate_complex_fixed_point(input_data, total_bits, int_bits)
                else:
                    quantized_input = self.simulate_fixed_point(input_data, total_bits, int_bits)
                
                # Run operator on quantized input
                quantized_output = operator_func(quantized_input)
                
                # Calculate error metrics
                if np.iscomplexobj(reference_output):
                    abs_error = np.abs(quantized_output - reference_output)
                    rel_error = abs_error / (np.abs(reference_output) + 1e-10)
                else:
                    abs_error = np.abs(quantized_output - reference_output)
                    rel_error = abs_error / (np.abs(reference_output) + 1e-10)
                
                mean_abs_error = np.mean(abs_error)
                max_abs_error = np.max(abs_error)
                mean_rel_error = np.mean(rel_error)
                max_rel_error = np.max(rel_error)
                
                # Calculate SNR (Signal-to-Noise Ratio)
                if np.iscomplexobj(reference_output):
                    signal_power = np.mean(np.abs(reference_output)**2)
                    noise_power = np.mean(np.abs(quantized_output - reference_output)**2)
                else:
                    signal_power = np.mean(reference_output**2)
                    noise_power = np.mean((quantized_output - reference_output)**2)
                
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
                
                config_result = {
                    'total_bits': total_bits,
                    'int_bits': int_bits,
                    'frac_bits': frac_bits,
                    'mean_abs_error': mean_abs_error,
                    'max_abs_error': max_abs_error,
                    'mean_rel_error': mean_rel_error,
                    'max_rel_error': max_rel_error,
                    'snr_db': snr_db
                }
                
                results['configurations'].append(config_result)
                
                print(f"Config {total_bits}/{int_bits}: SNR={snr_db:.2f}dB, Mean Rel Error={mean_rel_error:.6f}")
        
        # Sort configurations by SNR
        results['configurations'].sort(key=lambda x: x['snr_db'], reverse=True)
        
        # Store best configuration as recommendation
        best_config = results['configurations'][0]
        self.recommendations[operator_name] = {
            'total_bits': best_config['total_bits'],
            'int_bits': best_config['int_bits'],
            'frac_bits': best_config['frac_bits'],
            'snr_db': best_config['snr_db']
        }
        
        print(f"\nBest configuration for {operator_name}:")
        print(f"  ap_fixed<{best_config['total_bits']}, {best_config['int_bits']}> (SNR: {best_config['snr_db']:.2f}dB)")
        
        return results
    
    def plot_error_vs_bitwidth(self, results: Dict, operator_name: str, metric: str = 'snr_db'):
        """
        Plot error metrics vs bit width configurations.
        
        Args:
            results: Results dictionary from analyze_operator
            operator_name: Name of the operator
            metric: Metric to plot ('snr_db', 'mean_rel_error', etc.)
        """
        plt.figure(figsize=(10, 6))
        
        # Group by total_bits
        total_bits_values = sorted(set(c['total_bits'] for c in results['configurations']))
        
        for total_bits in total_bits_values:
            configs = [c for c in results['configurations'] if c['total_bits'] == total_bits]
            configs.sort(key=lambda x: x['int_bits'])
            
            x_values = [c['int_bits'] for c in configs]
            y_values = [c[metric] for c in configs]
            
            plt.plot(x_values, y_values, 'o-', label=f'Total bits: {total_bits}')
        
        plt.xlabel('Integer Bits')
        plt.ylabel('SNR (dB)' if metric == 'snr_db' else metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Bit Configuration for {operator_name}')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, f"{operator_name}_{metric}.png"))
        plt.close()
    
    def save_analysis(self, filename: str = "fixed_point_analysis.pkl"):
        """
        Save the analysis results to a file.
        
        Args:
            filename: Name of the file to save results
        """
        results = {
            'value_stats': self.value_stats,
            'recommendations': self.recommendations
        }
        
        with open(os.path.join(self.save_dir, filename), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save a text summary
        with open(os.path.join(self.save_dir, "recommendations.txt"), 'w') as f:
            f.write("# Fixed-Point Recommendations for Deep Tissue Imaging\n\n")
            
            for op_name, rec in self.recommendations.items():
                f.write(f"## {op_name}\n")
                f.write(f"- Total bits: {rec['total_bits']}\n")
                f.write(f"- Integer bits: {rec['int_bits']}\n")
                f.write(f"- Fractional bits: {rec['frac_bits']}\n")
                f.write(f"- SNR: {rec['snr_db']:.2f} dB\n")
                f.write(f"- Recommended type: ap_fixed<{rec['total_bits']}, {rec['int_bits']}>\n\n")
    
    def load_analysis(self, filename: str = "fixed_point_analysis.pkl"):
        """
        Load analysis results from a file.
        
        Args:
            filename: Name of the file to load results from
        """
        with open(os.path.join(self.save_dir, filename), 'rb') as f:
            results = pickle.load(f)
        
        self.value_stats = results['value_stats']
        self.recommendations = results['recommendations']
    
    def generate_hls_types(self) -> str:
        """
        Generate C++ HLS type definitions based on recommendations.
        
        Returns:
            String with C++ type definitions
        """
        code = "// Auto-generated fixed-point type definitions for deep tissue imaging\n"
        code += "#include <ap_fixed.h>\n"
        code += "#include <complex>\n\n"
        
        # Add typedefs for each operator
        for op_name, rec in self.recommendations.items():
            type_name = op_name.lower().replace(' ', '_')
            code += f"// {op_name} - SNR: {rec['snr_db']:.2f} dB\n"
            code += f"typedef ap_fixed<{rec['total_bits']}, {rec['int_bits']}> {type_name}_t;\n"
        
        # Add complex types
        code += "\n// Complex number types\n"
        for op_name, rec in self.recommendations.items():
            type_name = op_name.lower().replace(' ', '_')
            code += f"typedef std::complex<{type_name}_t> {type_name}_complex_t;\n"
        
        return code


def analyze_beam_profile(phi0, phi_final, analyzer):
    """
    Analyze the beam profile (initial and final) to determine fixed-point parameters.
    
    Args:
        phi0: Initial beam profile
        phi_final: Final beam profile
        analyzer: FixedPointAnalyzer instance
    """
    # Analyze initial beam profile
    analyzer.analyze_range(phi0, "initial_beam")
    
    # Analyze final beam profile
    analyzer.analyze_range(phi_final, "final_beam")
    
    # Plot histograms of values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(phi0).flatten(), bins=50)
    plt.title("Initial Beam Amplitude Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(phi_final).flatten(), bins=50)
    plt.title("Final Beam Amplitude Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.save_dir, "beam_profile_distribution.png"))
    plt.close()


def analyze_operators(phi, tejido, domain, analyzer):
    """
    Analyze all operators in the deep tissue imaging algorithm.
    
    Args:
        phi: Sample field
        tejido: Tissue properties
        domain: Domain properties
        analyzer: FixedPointAnalyzer instance
    """
    from deep_tissue_imaging.propagators import step_operators as so
    
    # Define wrapper functions for each operator
    def adi_x_wrapper(input_phi):
        return so.adi_x(input_phi, domain.Ny, domain.eps, domain.k, domain.dz, domain.dx)
    
    def adi_y_wrapper(input_phi):
        return so.adi_y(input_phi, domain.Nx, domain.eps, domain.k, domain.dz, domain.dy)
    
    def half_nonlinear_wrapper(input_phi):
        return so.half_nonlinear(input_phi, domain.k, tejido.n2, domain.dz)
    
    def half_linear_absorption_wrapper(input_phi):
        return so.half_linear_absorption(input_phi, tejido.alpha, domain.dz)
    
    def half_2photon_absorption_wrapper(input_phi):
        return so.half_2photon_absorption(input_phi, tejido.beta, domain.dz)
    
    # Analyze each operator
    print("\n=== Analyzing ADI-X Operator ===")
    adi_x_results = analyzer.analyze_operator(adi_x_wrapper, phi.copy(), "ADI_X")
    analyzer.plot_error_vs_bitwidth(adi_x_results, "ADI_X")
    
    print("\n=== Analyzing ADI-Y Operator ===")
    adi_y_results = analyzer.analyze_operator(adi_y_wrapper, phi.copy(), "ADI_Y")
    analyzer.plot_error_vs_bitwidth(adi_y_results, "ADI_Y")
    
    print("\n=== Analyzing Nonlinear Operator ===")
    nonlinear_results = analyzer.analyze_operator(half_nonlinear_wrapper, phi.copy(), "Nonlinear")
    analyzer.plot_error_vs_bitwidth(nonlinear_results, "Nonlinear")
    
    print("\n=== Analyzing Linear Absorption Operator ===")
    linear_abs_results = analyzer.analyze_operator(half_linear_absorption_wrapper, phi.copy(), "LinearAbsorption")
    analyzer.plot_error_vs_bitwidth(linear_abs_results, "LinearAbsorption")
    
    print("\n=== Analyzing Two-Photon Absorption Operator ===")
    tpa_results = analyzer.analyze_operator(half_2photon_absorption_wrapper, phi.copy(), "TwoPhotonAbsorption")
    analyzer.plot_error_vs_bitwidth(tpa_results, "TwoPhotonAbsorption")


def main():
    """
    Main function to run the fixed-point analysis on the deep tissue imaging algorithm.
    """
    import deep_tissue_imaging.elementos.lasers as lasers
    import deep_tissue_imaging.elementos.tejidos as tejidos
    import deep_tissue_imaging.elementos.domain as Domain
    import deep_tissue_imaging.propagators.propagation as prop
    
    print("=== Deep Tissue Imaging Fixed-Point Analysis ===")
    
    # Create analyzer
    analyzer = FixedPointAnalyzer(save_dir="./fixed_point_analysis")
    
    # Setup domain parameters (similar to deep_tissue_imaging_1.py)
    Lz = np.float32(361e-6)  # 361um
    Nz = 361
    dz = np.float32(Lz / Nz)  # 1um
    
    Lx, Ly = np.float32(45e-6), np.float32(45e-6)  # 45um x 45um
    Nx, Ny = 256, 256
    dx = np.float32(Lx / Nx)  # 0.18um
    dy = np.float32(Ly / Ny)  # 0.18um
    
    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Get laser and tissue properties
    laser = lasers.fuente_microscopia_1
    tejido = tejidos.cerebro_emb_pez_cebra
    
    k0 = np.float32(2*np.pi / laser.wavelength)
    k = np.float32(k0 * tejido.n_0)
    sigma_phi = np.float32(k * tejido.Dn * tejido.l_s)
    sigma_x = np.float32(5e-6)  # Typical value for brain tissue
    
    domain = Domain.Domain(X, Y, Nx, Ny, Nz, dx, dy, dz, np.float32(1e-12), k0, k, sigma_phi, sigma_x)
    
    # Generate initial beam profile
    phi0 = lasers.campo_tem00(X, Y, laser.w0, laser.I_peak)
    
    print("Running a short propagation to get sample field states...")
    # Run a short propagation to get sample field states
    phi_history = prop.full_propagation_within_tissue(phi0, tejido, domain, mask_manager=None)
    
    # Get final beam profile
    phi_final = phi_history[-1]
    
    # Get a middle field state for operator analysis
    phi_middle = phi_history[len(phi_history)//2]
    
    # Analyze beam profiles
    print("\n=== Analyzing Beam Profiles ===")
    analyze_beam_profile(phi0, phi_final, analyzer)
    
    # Analyze operators
    analyze_operators(phi_middle, tejido, domain, analyzer)
    
    # Save analysis results
    analyzer.save_analysis()
    
    # Generate HLS type definitions
    hls_types = analyzer.generate_hls_types()
    with open(os.path.join(analyzer.save_dir, "fixed_point_types.h"), 'w') as f:
        f.write(hls_types)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {analyzer.save_dir}")
    print("Recommended fixed-point types:")
    
    for op_name, rec in analyzer.recommendations.items():
        print(f"  {op_name}: ap_fixed<{rec['total_bits']}, {rec['int_bits']}> (SNR: {rec['snr_db']:.2f}dB)")


if __name__ == "__main__":
    main()