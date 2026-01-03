"""
Mathematical utility functions
"""

import numpy as np


def moody(Re:  float, roughness_ratio: float) -> float:
    """
    Calculate Moody friction factor using Colebrook-White equation
    
    Parameters
    ----------
    Re : float
        Reynolds number
    roughness_ratio : float
        Relative roughness (epsilon/D)
    
    Returns
    -------
    float
        Friction factor
    """
    if Re < 2300:
        # Laminar flow
        return 64.0 / Re
    else:
        # Turbulent flow - use Swamee-Jain approximation
        # More stable than iterative Colebrook-White
        if roughness_ratio < 1e-8:
            # Smooth pipe (Blasius)
            return 0.316 / (Re ** 0.25)
        else:
            num = 0.25
            denom = (np.log10(roughness_ratio/3.7 + 5.74/(Re**0.9)))**2
            return num / denom


def cosd(angle_deg: float) -> float:
    """Cosine of angle in degrees"""
    return np.cos(np.radians(angle_deg))


def sind(angle_deg: float) -> float:
    """Sine of angle in degrees"""
    return np. sin(np.radians(angle_deg))


def tand(angle_deg: float) -> float:
    """Tangent of angle in degrees"""
    return np.tan(np.radians(angle_deg))


def arctand(value: float) -> float:
    """Arctangent returning degrees"""
    return np.degrees(np.arctan(value))


def arccosd(value: float) -> float:
    """Arccosine returning degrees"""
    return np.degrees(np.arccos(value))


def arcsind(value: float) -> float:
    """Arcsine returning degrees"""
    return np.degrees(np.arcsin(value))