#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

def compare_transmittance(file1: str, file2: str, tol: float = 1e-8):
    """
    Load two CSVs with columns ['z','T'], align them, and report on differences.
    """
    # Read the data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check that z-vectors match
    if not np.allclose(df1['z'], df2['z'], atol=tol):
        diffs = df1['z'] - df2['z']
        max_z_diff = np.abs(diffs).max()
        print(f"⚠️  z arrays differ! max |Δz| = {max_z_diff:.3e}")
    else:
        print("✅ z arrays match within tolerance.")

    # Compute T differences
    diff = df1['T'] - df2['T']
    max_diff = diff.abs().max()
    mean_diff = diff.mean()
    std_diff = diff.std()

    print(f"Max ΔT = {max_diff:.3e}")
    print(f"Mean ΔT = {mean_diff:.3e}")
    print(f"Std  ΔT = {std_diff:.3e}")

    if max_diff <= tol:
        print("🎉 Transmittance curves are identical within the given tolerance.")
    else:
        print("❌ Transmittance curves differ beyond tolerance!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_transmittance.py T_n.csv T_s.csv")
        sys.exit(1)
    file_n, file_s = sys.argv[1], sys.argv[2]
    compare_transmittance(file_n, file_s)

    # run from terminal (with env activated):
    # python .\T_comparison.py T_n.csv T_s.csv

