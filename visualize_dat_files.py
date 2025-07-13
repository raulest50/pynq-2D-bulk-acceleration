import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def load_complex_matrix(filename, shape=None):
    """
    Load a complex matrix from a .dat file.
    
    Args:
        filename (str): Path to the .dat file
        shape (tuple, optional): Shape to reshape the matrix. If None, will try to determine a square shape.
    
    Returns:
        ndarray: Complex matrix loaded from the file
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                real, imag = float(parts[0]), float(parts[1])
                data.append(complex(real, imag))
    
    # If shape is not provided, try to determine a square shape
    if shape is None:
        n = int(np.sqrt(len(data)))
        if n * n != len(data):
            raise ValueError(f"Cannot determine shape for {len(data)} elements. Please provide shape explicitly.")
        shape = (n, n)
    
    # Reshape the data into the original matrix shape
    return np.array(data, dtype=np.complex64).reshape(shape)

def visualize_complex_field(matrix, output_file, title=None, cmap='viridis'):
    """
    Visualize a complex field and save as PNG.
    
    Args:
        matrix (ndarray): Complex matrix to visualize
        output_file (str): Path to save the PNG file
        title (str, optional): Title for the plot
        cmap (str, optional): Colormap to use
    """
    # Calculate intensity (|E|²)
    intensity = np.abs(matrix)**2
    
    # Normalize intensity for better visualization
    intensity_norm = intensity / np.max(intensity)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot intensity
    plt.imshow(intensity_norm, cmap=cmap, origin='lower')
    plt.colorbar(label='Normalized Intensity')
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {output_file}")

def visualize_phase(matrix, output_file, title=None, cmap='twilight'):
    """
    Visualize the phase of a complex field and save as PNG.
    
    Args:
        matrix (ndarray): Complex matrix to visualize
        output_file (str): Path to save the PNG file
        title (str, optional): Title for the plot
        cmap (str, optional): Colormap to use
    """
    # Calculate phase
    phase = np.angle(matrix)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot phase
    plt.imshow(phase, cmap=cmap, origin='lower')
    plt.colorbar(label='Phase (radians)')
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved phase visualization to {output_file}")

def process_dat_files(input_dir, output_dir, shape=None):
    """
    Process all .dat files in the input directory and generate PNG visualizations.
    
    Args:
        input_dir (str): Directory containing .dat files
        output_dir (str): Directory to save PNG files
        shape (tuple, optional): Shape to reshape the matrices. If None, will try to determine a square shape.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .dat files in the input directory
    dat_files = list(Path(input_dir).glob('*.dat'))
    
    if not dat_files:
        print(f"No .dat files found in {input_dir}")
        return
    
    print(f"Found {len(dat_files)} .dat files in {input_dir}")
    
    # Process each .dat file
    for dat_file in dat_files:
        try:
            # Load complex matrix
            matrix = load_complex_matrix(dat_file, shape)
            
            # Generate base output filename
            base_name = os.path.splitext(os.path.basename(dat_file))[0]
            
            # Visualize intensity
            intensity_output = os.path.join(output_dir, f"{base_name}_intensity.png")
            visualize_complex_field(matrix, intensity_output, title=f"Intensity: {base_name}")
            
            # Visualize phase
            phase_output = os.path.join(output_dir, f"{base_name}_phase.png")
            visualize_phase(matrix, phase_output, title=f"Phase: {base_name}")
            
        except Exception as e:
            print(f"Error processing {dat_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize .dat files as PNG images")
    parser.add_argument("--input-dir", default="validation_data_main", help="Directory containing .dat files")
    parser.add_argument("--output-dir", default="visualization", help="Directory to save PNG files")
    parser.add_argument("--shape", type=int, nargs=2, help="Shape to reshape the matrices (rows cols)")
    args = parser.parse_args()
    
    shape = tuple(args.shape) if args.shape else None
    
    process_dat_files(args.input_dir, args.output_dir, shape)

if __name__ == "__main__":
    main()