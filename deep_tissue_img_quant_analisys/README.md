# Deep Tissue Imaging Fixed-Point Analysis

This package provides tools for analyzing the deep tissue imaging algorithm and recommending optimal fixed-point parameters for FPGA implementation. The goal is to balance precision and resource usage on the FPGA.

## Overview

The fixed-point analysis tools in this package:

1. Analyze the range of values in the deep tissue imaging algorithm
2. Determine appropriate bit-widths for different operations
3. Simulate fixed-point quantization to evaluate precision
4. Provide recommendations for fixed-point parameters
5. Generate C++ HLS type definitions for FPGA implementation

## Usage

### Command-line Interface

The easiest way to run the analysis is using the command-line script:

```bash
python -m deep_tissue_img_quant_analisys.run_fixed_point_analysis
```

Command-line options:

- `--output-dir`: Directory to save analysis results (default: `./fixed_point_analysis`)
- `--bit-widths`: Comma-separated list of total bit widths to test (default: `16,20,24,28,32`)
- `--short-run`: Run a shorter propagation for faster analysis (less accurate)
- `--load-existing`: Load existing analysis results if available

### Programmatic Usage

You can also use the `FixedPointAnalyzer` class directly in your code:

```python
from deep_tissue_img_quant_analisys.fixed_point_analyzer import FixedPointAnalyzer

# Create analyzer
analyzer = FixedPointAnalyzer(save_dir="./my_analysis")

# Analyze a specific variable
stats = analyzer.analyze_range(my_variable, "my_variable_name")

# Simulate fixed-point quantization
quantized = analyzer.simulate_fixed_point(my_variable, total_bits=24, int_bits=8)

# Analyze an operator
results = analyzer.analyze_operator(
    operator_func=my_operator_function,
    input_data=my_input_data,
    operator_name="MyOperator",
    bit_width_range=[16, 20, 24, 28, 32]
)

# Save analysis results
analyzer.save_analysis()

# Generate HLS type definitions
hls_types = analyzer.generate_hls_types()
```

## Output

The analysis generates several outputs in the specified output directory:

1. **Plots**: 
   - SNR vs bit configuration for each operator
   - Beam profile distributions

2. **Text Reports**:
   - `recommendations.txt`: Summary of recommended fixed-point parameters

3. **C++ Header File**:
   - `fixed_point_types.h`: HLS type definitions for FPGA implementation

4. **Serialized Data**:
   - `fixed_point_analysis.pkl`: Complete analysis results that can be loaded later

## Interpreting Results

The analysis recommends fixed-point parameters based on Signal-to-Noise Ratio (SNR) and relative error metrics. Higher SNR values indicate better precision.

For each operator, the recommended type is in the format:

```
ap_fixed<total_bits, int_bits>
```

Where:
- `total_bits`: Total number of bits in the fixed-point representation
- `int_bits`: Number of bits for the integer part (including sign bit)

The remaining bits (`total_bits - int_bits`) are used for the fractional part.

## Example Recommendations

Typical recommendations might look like:

```
ADI_X: ap_fixed<24, 4> (SNR: 85.32dB)
ADI_Y: ap_fixed<24, 4> (SNR: 84.76dB)
Nonlinear: ap_fixed<20, 3> (SNR: 92.15dB)
LinearAbsorption: ap_fixed<16, 2> (SNR: 98.43dB)
TwoPhotonAbsorption: ap_fixed<20, 3> (SNR: 95.67dB)
```

These recommendations can be directly used in your HLS implementation to optimize resource usage while maintaining precision.

## Advanced Usage

For more advanced analysis, you can modify the `fixed_point_analyzer.py` script to customize:

- Error metrics used for optimization
- Bit-width ranges to test
- Visualization options
- Custom operators to analyze