# Import functions to expose at the package level
from .matrix_utils import create_tridiagonal_matrix
from .wave_functions import u_gaussian, ut_gaussian, utt_gaussian, uttt_gaussian

# Version information
__version__ = '0.1.0'