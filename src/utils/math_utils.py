"""Mathematical utility functions (aligned with RadComp correlations)."""

import math
import numpy as np
from scipy import optimize


def moody(Re: float, roughness_ratio: float) -> float:
    """Calculate Moody friction factor using Colebrook-White (RadComp parity)."""
    if Re < 2300.0:
        return 64.0 / Re / 4.0

    def colebrook(x: float) -> float:
        # fsolve passes ndarray; coerce scalar to avoid array math issues
        f = float(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else float(x)
        return -2 * math.log10(roughness_ratio / 3.72 + 2.51 / Re / f**0.5) - 1 / f**0.5

    return optimize.fsolve(colebrook, 0.02)[0] / 4.0


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