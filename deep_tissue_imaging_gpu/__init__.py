"""
Deep Tissue Imaging GPU package.

This package provides GPU-accelerated functions for deep tissue imaging simulations.
It uses a hybrid CPU-GPU approach where only the halfsteps are accelerated with GPU,
while the ADI operations and phase mask management remain on CPU.
"""