"""
Incidence Loss Models

Models for losses due to flow incidence at impeller inlet.
These models only require inlet conditions (Pass A timing).
"""

import math
from ..registry import register_impeller
from ..context import ImpellerContext

@register_impeller("incidence", "galvas")
def incidence_galvas(ctx: ImpellerContext, **kwargs) -> float:
    """Galvas (1973) incidence loss model."""
    # blockage at RMS dia (B2)
    Z = ctx.geom.n_blades + ctx.geom.n_splits
    t = getattr(ctx.geom, "blade_le", getattr(ctx.geom, "blade_e", 0.0))
    d2 = 2 * ctx.geom.r2rms
    beta_2b = math.radians(ctx.geom.beta2)
    
    B2 = 1.0 - Z * t / (math.pi * d2 * math.sin(beta_2b))
    
    # epsilon_2
    eps2 = math.atan((1 - B2) * math.tan(beta_2b) / (1 + B2 * math.tan(beta_2b)**2))
    beta_opt = beta_2b - eps2
    
    # Meridional velocity at station 2
    W2m = ctx.V2 * math.cos(math.radians(ctx.geom.alpha2))
    
    # Specific heat at constant pressure (fallback if not available)
    Cp = getattr(ctx.st2.static, 'CP', kwargs.get('Cp', 1005.0))
    
    WL = W2m * math.cos(abs(beta_opt - beta_2b))
    return max(0.0, WL**2 / (2.0 * Cp))

@register_impeller("incidence", "aungier")
def incidence_aungier(ctx: ImpellerContext, **kwargs) -> float:
    """Aungier (1995) incidence loss model with compressibility effects."""
    beta_2b = math.radians(ctx.geom.beta2)  # inlet blade angle
    V2m = ctx.V2 * math.cos(math.radians(ctx.geom.alpha2))
    return max(0.0, 0.4 * (ctx.W2 - V2m/math.cos(beta_2b))**2)

@register_impeller("incidence", "conrad")
def incidence_conrad(ctx: ImpellerContext, **kwargs) -> float:
    """Conrad (1980) incidence loss model."""
    W2u = ctx.U2 - ctx.V2u  # tangential component of relative velocity
    return max(0.0, 0.6 * (W2u**2) / 2.0)

@register_impeller("incidence", "galvas_stanitz")
def incidence_galvas_stanitz(ctx: ImpellerContext, **kwargs) -> float:
    """Galvas-Stanitz incidence loss model."""
    beta2f = kwargs.get('beta2f', math.radians(ctx.geom.beta2))
    beta2_opt = kwargs.get('beta2_opt', beta2f * 0.9)  # approximation
    return max(0.0, 0.5 * (ctx.W2**2) * (math.sin(abs(beta2f - beta2_opt)))**2)