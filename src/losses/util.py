"""
Loss Model Utility Functions

Shared helper functions used across multiple loss models.
Avoids duplication and ensures consistent calculations.
"""

import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...geometry.geometry import Geometry
    from ...components.impeller import ImpellerState

def compute_W_avg_for_sf(st2: "ImpellerState", st4: "ImpellerState") -> float:
    """
    Jansen/Oh weighted average relative velocity for skin friction:
    W̅ = (W2t + W2h + 2*W4) / 4
    
    Uses available fields; graceful fallback if hub/tip components missing.
    """
    terms = []
    weights = []
    
    # Inlet tip (shroud) relative velocity
    if hasattr(st2, "Ws"):
        terms.append(abs(st2.Ws))
        weights.append(1.0)
    
    # Inlet hub relative velocity (or fallback to exit)
    if hasattr(st2, "Wh") and st2.Wh is not None:
        terms.append(abs(st2.Wh))
        weights.append(1.0)
    else:
        terms.append(abs(st4.W))
        weights.append(1.0)  # fallback
    
    # Exit relative velocity (double weight)
    if hasattr(st4, "W"):
        terms.append(abs(st4.W))
        weights.append(2.0)
    
    return sum(terms) / 4.0

def compute_Lb_and_Dhyd_Jansen(geom: "Geometry") -> tuple[float, float]:
    """
    Zhang (2019) Eqs. (8)-(9) for impeller flow length Lb and hydraulic diameter D_hyd.
    
    Returns:
        (Lb, D_hyd): Flow length and hydraulic diameter in meters
    """
    # Extract geometry parameters with fallbacks
    d4   = 2.0 * geom.r4 if hasattr(geom, "r4") else geom.d4
    d2t  = 2.0 * geom.r2s if hasattr(geom, "r2s") else geom.d2t
    d2h  = 2.0 * geom.r2h if hasattr(geom, "r2h") else geom.d2h
    b4   = geom.b4
    Z1   = geom.n_blades if hasattr(geom, "n_blades") else geom.Z
    Z2   = geom.n_splits if hasattr(geom, "n_splits") else 0
    Z    = Z1 + Z2
    Lz   = geom.l_comp if hasattr(geom, "l_comp") else (geom.r4 - geom.r2s + geom.b4)
    
    # Blade angles (with fallbacks)
    beta2s = np.radians(getattr(geom, "beta2s", getattr(geom, "beta2", 0.0)))
    beta2  = np.radians(getattr(geom, "beta2", 0.0))
    beta4  = np.radians(getattr(geom, "beta4", 0.0))
    
    # Flow length Lb (Zhang Eq. 8)
    cos_avg_inlet = 0.5 * (np.cos(beta2s) + np.cos(beta2))
    cos_term = 2.0 / (cos_avg_inlet + np.cos(beta4))
    Lb = (np.pi/8.0) * (d4 - (d2t + d2h)/2.0 - b4 + 2.0*Lz) * cos_term
    
    # Hydraulic diameter D_hyd (Zhang Eq. 9)
    term1 = d4 * np.cos(beta4) * (Z/np.pi + (d4*np.cos(beta4))/b4)
    
    d_ratio = 0.5 * (d2t/d4 + d2h/d4)
    cos_avg_normalized = cos_avg_inlet / 2.0
    denom = Z/np.pi + (d2t + d2h)/(d2t - d2h) * cos_avg_inlet
    term2 = d_ratio * cos_avg_normalized / denom
    
    Dhyd = term1 + term2
    Dhyd = max(Dhyd, 1e-6)  # Guard against numerical issues
    
    return Lb, Dhyd

def compute_Cf_Jansen(Dhyd: float, U4: float, nu02: float) -> float:
    """
    Jansen-style friction coefficient per Zhang (2019):
    Cf = 0.0412 * Re^(-0.1925), where Re = U4 * Dhyd / nu02
    """
    Re = max(1.0, U4 * Dhyd / max(1e-12, nu02))
    return 0.0412 * (Re ** -0.1925)

def compute_coppage_diffusion_factor(Z: int, r2s: float, r4: float, 
                                   W2s: float, W4: float, 
                                   dH_Euler: float, U4: float) -> float:
    """
    Coppage diffusion factor with geometry bracket:
    Df = 1 - (W4/W2s) + 0.75 * (dH_Euler/U4²) * (W4/W2s) * geometry_bracket
    """
    geom_bracket = (Z/np.pi) * (1.0 - r2s/r4) + 2.0 * (r2s/r4)
    Df = 1.0 - (W4 / W2s) + 0.75 * (dH_Euler / U4**2) * (W4 / W2s) * geom_bracket
    return max(Df, 0.0)

def compute_reynolds_disc_friction(U: float, r: float, nu: float) -> float:
    """Compute Reynolds number for disc friction: Re = U * r / nu"""
    return max(1.0, U * r / max(1e-12, nu))

def compute_daily_nece_friction_factor(Re: float) -> float:
    """
    Daily & Nece (1960) piecewise friction factor:
    f_df = 2.67 / Re^0.5     if Re < 3e5
    f_df = 0.0622 / Re^0.2   if Re >= 3e5
    """
    if Re < 3.0e5:
        return 2.67 / (Re**0.5)
    else:
        return 0.0622 / (Re**0.2)

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safe division with fallback for numerical stability."""
    return numerator / max(abs(denominator), 1e-12) if denominator != 0.0 else fallback

def ensure_positive(value: float, minimum: float = 0.0) -> float:
    """Ensure value is non-negative (loss models must return >= 0)."""
    return max(value, minimum)

def compute_meridional_velocity(V: float, alpha_deg: float) -> float:
    """Compute meridional component: V_m = V * cos(alpha)"""
    return V * math.cos(math.radians(alpha_deg))

def compute_tangential_velocity(V: float, alpha_deg: float) -> float:
    """Compute tangential component: V_u = V * sin(alpha)"""
    return V * math.sin(math.radians(alpha_deg))

def compute_Cf_Jansen(Dhyd: float, U4: float, nu02: float) -> float:
    """
    Jansen-style Cf per Zhang (2019):
    Cf = 0.0412 * Re^{-0.1925}, Re = U4 * Dhyd / nu02
    """
    Re = max(1.0, U4 * Dhyd / max(1e-12, nu02))
    return 0.0412 * (Re ** -0.1925)