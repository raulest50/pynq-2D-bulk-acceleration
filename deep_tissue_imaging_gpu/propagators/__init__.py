"""
Propagators module for deep tissue imaging GPU package.

This module provides GPU-accelerated propagation functions for deep tissue imaging.
It uses a hybrid CPU-GPU approach where only the halfsteps are accelerated with GPU,
while the ADI operations and phase mask management remain on CPU.
"""