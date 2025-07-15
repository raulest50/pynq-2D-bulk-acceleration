#!/usr/bin/env python
"""
Example script demonstrating how to use the FixedPointAnalyzer with a simple test case.
This can be used as a template for custom analysis workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from deep_tissue_img_quant_analisys.fixed_point_analyzer import FixedPointAnalyzer

def simple_operator(x):
    """A simple operator that applies a nonlinear transformation."""
    return np.exp(1j * np.abs(x)**2) * x

def main():
    """Run a simple example of fixed-point analysis."""
    # Create output directory
    output_dir = "./fixed_point_example"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create analyzer
    analyzer = FixedPointAnalyzer(save_dir=output_dir)
    
    # Generate test data: a complex Gaussian beam
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a Gaussian beam
    w0 = 2.0  # beam waist
    amplitude = 1.0
    r2 = X**2 + Y**2
    field = amplitude * np.exp(-r2/w0**2)
    
    # Add some phase variation
    phase = 0.5 * X + 0.3 * Y
    complex_field = field * np.exp(1j * phase)
    
    # Analyze the input field
    print("Analyzing input field...")
    input_stats = analyzer.analyze_range(complex_field, "input_field")
    
    # Simulate fixed-point quantization with different bit configurations
    print("\nSimulating fixed-point quantization...")
    
    # Test a few configurations
    configs = [
        (16, 4),  # 16 total bits, 4 integer bits
        (20, 4),  # 20 total bits, 4 integer bits
        (24, 4),  # 24 total bits, 4 integer bits
    ]
    
    plt.figure(figsize=(15, 5))
    
    # Original field
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(complex_field), cmap='viridis')
    plt.title("Original Field")
    plt.colorbar()
    
    # Plot quantized fields
    for i, (total_bits, int_bits) in enumerate(configs):
        quantized = analyzer.simulate_complex_fixed_point(complex_field, total_bits, int_bits)
        
        # Calculate error
        abs_error = np.abs(quantized - complex_field)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        
        # Calculate SNR
        signal_power = np.mean(np.abs(complex_field)**2)
        noise_power = np.mean(np.abs(quantized - complex_field)**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        print(f"Config {total_bits}/{int_bits}: Max Error={max_error:.6f}, Mean Error={mean_error:.6f}, SNR={snr_db:.2f}dB")
        
        # Plot
        plt.subplot(1, 4, i+2)
        plt.imshow(np.abs(quantized), cmap='viridis')
        plt.title(f"Quantized {total_bits}/{int_bits}\nSNR={snr_db:.1f}dB")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantization_comparison.png"))
    
    # Analyze the operator
    print("\nAnalyzing operator...")
    operator_results = analyzer.analyze_operator(
        simple_operator,
        complex_field,
        "SimpleNonlinear",
        bit_width_range=[16, 20, 24, 28]
    )
    
    # Plot error vs bit configuration
    analyzer.plot_error_vs_bitwidth(operator_results, "SimpleNonlinear")
    
    # Save analysis results
    analyzer.save_analysis("example_analysis.pkl")
    
    # Generate HLS type definitions
    hls_types = analyzer.generate_hls_types()
    with open(os.path.join(output_dir, "example_fixed_point_types.h"), 'w') as f:
        f.write(hls_types)
    
    print(f"\nExample complete. Results saved to {output_dir}")
    print("Recommended fixed-point types:")
    
    for op_name, rec in analyzer.recommendations.items():
        print(f"  {op_name}: ap_fixed<{rec['total_bits']}, {rec['int_bits']}> (SNR: {rec['snr_db']:.2f}dB)")

if __name__ == "__main__":
    main()