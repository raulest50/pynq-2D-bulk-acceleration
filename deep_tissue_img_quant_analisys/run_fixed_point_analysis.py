#!/usr/bin/env python
"""
Command-line script to run fixed-point analysis on the deep tissue imaging algorithm.
This script analyzes the algorithm and recommends optimal fixed-point parameters
for FPGA implementation to balance precision and resource usage.
"""

import argparse
import os
import sys
import numpy as np
from deep_tissue_img_quant_analisys.fixed_point_analyzer import FixedPointAnalyzer, main as run_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze deep tissue imaging algorithm and recommend fixed-point parameters for FPGA implementation."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./fixed_point_analysis",
        help="Directory to save analysis results (default: ./fixed_point_analysis)"
    )
    
    parser.add_argument(
        "--bit-widths", 
        type=str, 
        default="16,20,24,28,32",
        help="Comma-separated list of total bit widths to test (default: 16,20,24,28,32)"
    )
    
    parser.add_argument(
        "--short-run", 
        action="store_true",
        help="Run a shorter propagation for faster analysis (less accurate)"
    )
    
    parser.add_argument(
        "--load-existing", 
        action="store_true",
        help="Load existing analysis results if available"
    )
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run the analysis."""
    args = parse_args()
    
    # Parse bit widths
    bit_widths = [int(x.strip()) for x in args.bit_widths.split(",")]
    
    print(f"=== Deep Tissue Imaging Fixed-Point Analysis ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Testing bit widths: {bit_widths}")
    
    # Check if we should load existing analysis
    if args.load_existing and os.path.exists(os.path.join(args.output_dir, "fixed_point_analysis.pkl")):
        print("Loading existing analysis results...")
        analyzer = FixedPointAnalyzer(save_dir=args.output_dir)
        analyzer.load_analysis()
        
        print("\n=== Loaded Analysis Results ===")
        print("Recommended fixed-point types:")
        
        for op_name, rec in analyzer.recommendations.items():
            print(f"  {op_name}: ap_fixed<{rec['total_bits']}, {rec['int_bits']}> (SNR: {rec['snr_db']:.2f}dB)")
        
        # Generate HLS type definitions
        hls_types = analyzer.generate_hls_types()
        with open(os.path.join(args.output_dir, "fixed_point_types.h"), 'w') as f:
            f.write(hls_types)
        
        print(f"\nHLS type definitions written to {os.path.join(args.output_dir, 'fixed_point_types.h')}")
    else:
        # Run the full analysis
        print("Running fixed-point analysis...")
        run_analysis()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())