import numpy as np

# Analytical solution
def u_gaussian(x, t, σ=0.3, c=1):
    """
    Calculates a Gaussian wave function.
    
    Args:
        x (float or numpy.ndarray): Spatial coordinate(s)
        t (float): Time
        σ (float, optional): Width parameter. Defaults to 0.3.
        c (float, optional): Wave speed. Defaults to 1.
        
    Returns:
        float or numpy.ndarray: Gaussian function value(s)
    """
    return np.exp(-((x-4)-c*t)**2/(2*σ**2))


def ut_gaussian(x, t, σ=0.3, c=1):
    """
    Calculates the first time derivative of a Gaussian wave function.
    
    Args:
        x (float or numpy.ndarray): Spatial coordinate(s)
        t (float): Time
        σ (float, optional): Width parameter. Defaults to 0.3.
        c (float, optional): Wave speed. Defaults to 1.
        
    Returns:
        float or numpy.ndarray: First time derivative value(s)
    """
    return c * ((x - 4) - c * t) / (σ ** 2) * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))


def utt_gaussian(x, t, σ=0.3, c=1):
    """
    Calculates the second time derivative of a Gaussian wave function.
    
    Args:
        x (float or numpy.ndarray): Spatial coordinate(s)
        t (float): Time
        σ (float, optional): Width parameter. Defaults to 0.3.
        c (float, optional): Wave speed. Defaults to 1.
        
    Returns:
        float or numpy.ndarray: Second time derivative value(s)
    """
    term1 = -c ** 2 / (σ ** 2)
    term2 = 1 - ((x - 4) - c * t) ** 2 / (σ ** 2)
    return term1 * term2 * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))


def uttt_gaussian(x, t, σ=0.3, c=1):
    """
    Calculates the third time derivative of a Gaussian wave function.
    
    Args:
        x (float or numpy.ndarray): Spatial coordinate(s)
        t (float): Time
        σ (float, optional): Width parameter. Defaults to 0.3.
        c (float, optional): Wave speed. Defaults to 1.
        
    Returns:
        float or numpy.ndarray: Third time derivative value(s)
    """
    term1 = c ** 3 / (σ ** 4)
    term2 = 3 * ((x - 4) - c * t) - ((x - 4) - c * t) ** 3 / (σ ** 2)
    return term1 * term2 * np.exp(-((x - 4) - c * t) ** 2 / (2 * σ ** 2))